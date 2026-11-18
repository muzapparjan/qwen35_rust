//! The whole forward pass, recorded.
//!
//! [`Model::forward`] calls one kernel at a time on host slices. That is the right
//! shape for a CPU, where a call is a function call, and the wrong shape for a
//! device, where a call is a submission: a forward pass is a few hundred operators
//! and each submission costs about half a millisecond.
//!
//! [`Model::record`] builds the same arithmetic as a [`Spec`] — a dataflow graph
//! over buffers — so one executor can run the entire pass. The two paths are
//! gated against each other bit for bit on the CPU, which is what says the
//! recording is faithful rather than merely similar.

use crate::backend::{Arena, Flow, Slot, Spec};
use crate::weights::{self, manifest, Matrix};

use super::{layer_base, vector_at, Cache, Model};

/// A recorded pass and the state it needs.
pub struct Recorded {
    /// The pass itself.
    pub spec: Spec,
    /// One buffer per caller-owned state, in the order the recorder asked for
    /// them: for each full-attention layer its keys and values, then for each
    /// linear-attention layer its convolution window and its recurrent state.
    ///
    /// Empty for a [`Model::record_step`], which copies nothing.
    pub state: Vec<Vec<f32>>,
    /// The arena index of the logits.
    pub logits: Slot,
    /// The arena index of the last position's hidden state.
    pub hidden: Slot,
}

impl Model {
    /// Record a forward pass over `ids` starting from `cache`.
    ///
    /// `positions` are the rope positions, one row per token; the caller builds
    /// them because an image prompt carries a grid rather than token indices.
    pub fn record(
        &self,
        ids: &[u32],
        sequence: usize,
        positions: &[[u32; 3]],
        cache: &Cache,
    ) -> Recorded {
        let mut state = Vec::new();
        for layer_slot in 0..manifest::FULL_LAYERS.len() {
            state.push(gather_history(
                &cache.attention[layer_slot].keys,
                cache.length,
                manifest::HEAD_DIM,
            ));
            state.push(gather_history(
                &cache.attention[layer_slot].values,
                cache.length,
                manifest::HEAD_DIM,
            ));
        }
        for layer_slot in 0..manifest::LINEAR_LAYERS.len() {
            state.push(cache.linear[layer_slot].conv.clone());
            state.push(cache.linear[layer_slot].state.clone());
        }
        self.record_with_state(ids, sequence, positions, cache.length, &state)
    }

    /// Record a forward pass from state the caller carries itself.
    ///
    /// `past` is how many positions the state already holds; `state` is laid out as
    /// [`Model::record`] returns it, so a generation loop can hand back what the
    /// last step left without going through [`Cache`].
    pub fn record_with_state(
        &self,
        ids: &[u32],
        sequence: usize,
        positions: &[[u32; 3]],
        past: usize,
        state: &[Vec<f32>],
    ) -> Recorded {
        self.record_pass(ids, sequence, positions, past, Some(state))
    }

    /// Record a step whose state the caller already holds, without copying it.
    ///
    /// [`Model::record_with_state`] copies every state buffer out of the caller's
    /// slices so the recording can seed a fresh executor. A decode step does not
    /// need that: the state is already on the device, and the copy is tens of
    /// megabytes — several times what running the step costs. This records the same
    /// pass and reports an empty [`Recorded::state`].
    pub fn record_step(
        &self,
        ids: &[u32],
        sequence: usize,
        positions: &[[u32; 3]],
        past: usize,
    ) -> Recorded {
        self.record_pass(ids, sequence, positions, past, None)
    }

    fn record_pass(
        &self,
        ids: &[u32],
        sequence: usize,
        positions: &[[u32; 3]],
        past: usize,
        state: Option<&[Vec<f32>]>,
    ) -> Recorded {
        let hidden = manifest::HIDDEN_SIZE;
        let eps = manifest::RMS_NORM_EPS;
        let mut flow = Flow::new();

        // --- the caller's state, in a fixed order. Its buffers are declared by
        // shape either way; the values are copied only when the caller asked for a
        // recording that can seed an executor on its own.
        let mut carried: Vec<Vec<f32>> = Vec::new();
        let mut next_state = 0usize;
        let mut attention_slots = Vec::new();
        let total = past + sequence;
        for layer_slot in 0..manifest::FULL_LAYERS.len() {
            let width = total * manifest::NUM_KEY_VALUE_HEADS * manifest::HEAD_DIM;
            let key_slot = flow.external(next_state, width);
            next_state += 1;
            let value_slot = flow.external(next_state, width);
            next_state += 1;
            if let Some(state) = state {
                carried.push(state[2 * layer_slot].clone());
                carried.push(state[2 * layer_slot + 1].clone());
            }
            attention_slots.push((key_slot, value_slot));
        }
        let mut linear_slots = Vec::new();
        let linear_base = 2 * manifest::FULL_LAYERS.len();
        for layer_slot in 0..manifest::LINEAR_LAYERS.len() {
            let conv_slot = flow.external(
                next_state,
                manifest::LINEAR_CONV_DIM * manifest::LINEAR_CONV_KERNEL,
            );
            next_state += 1;
            let state_slot = flow.external(
                next_state,
                manifest::LINEAR_VALUE_HEADS
                    * manifest::LINEAR_KEY_HEAD_DIM
                    * manifest::LINEAR_VALUE_HEAD_DIM,
            );
            next_state += 1;
            if let Some(state) = state {
                carried.push(state[linear_base + 2 * layer_slot].clone());
                carried.push(state[linear_base + 2 * layer_slot + 1].clone());
            }
            linear_slots.push((conv_slot, state_slot));
        }

        // --- the rope tables, laid out one row per rotated band: the query has
        // four heads per key head, so the two tables differ in length
        let (cos_query, sin_query) = roped_positions(positions, manifest::NUM_ATTENTION_HEADS);
        let cos_rows = flow.bind(cos_query);
        let sin_rows = flow.bind(sin_query);
        let (cos_key, sin_key) = roped_positions(positions, manifest::NUM_KEY_VALUE_HEADS);
        let cos_keys = flow.bind(cos_key);
        let sin_keys = flow.bind(sin_key);

        // --- the prompt
        let id_values: Vec<f32> = ids.iter().map(|id| *id as f32).collect();
        let id_slot = flow.bind(id_values);
        let x = flow.buffer(sequence * hidden);
        let embed_table = flow.tensor("model.language_model.embed_tokens.weight");
        flow.embed(embed_table, id_slot, sequence, hidden, x);

        // --- the layers
        let normalized = flow.buffer(sequence * hidden);
        let mixed = flow.buffer(sequence * hidden);
        let mut current = x;
        for layer in 0..manifest::NUM_HIDDEN_LAYERS {
            let base = layer_base(layer);
            let norm = flow.bound(vector_at(&base, "input_layernorm.weight"));
            flow.rms_norm(current, sequence, hidden, norm, eps, normalized);

            if manifest::LINEAR_LAYERS.contains(&layer) {
                let slot = manifest::LINEAR_LAYERS
                    .iter()
                    .position(|value| *value == layer)
                    .unwrap();
                self.record_linear_attention(
                    &mut flow,
                    &base,
                    slot,
                    &linear_slots[slot],
                    normalized,
                    sequence,
                    eps,
                    mixed,
                );
            } else {
                let slot = manifest::FULL_LAYERS.iter().position(|value| *value == layer).unwrap();
                let (keys, values) = attention_slots[slot];
                self.record_full_attention(
                    &mut flow,
                    &base,
                    normalized,
                    sequence,
                    past,
                    cos_rows,
                    sin_rows,
                    cos_keys,
                    sin_keys,
                    keys,
                    values,
                    mixed,
                );
            }
            let summed = flow.buffer(sequence * hidden);
            flow.add(current, mixed, summed);

            let norm = flow.bound(vector_at(&base, "post_attention_layernorm.weight"));
            flow.rms_norm(summed, sequence, hidden, norm, eps, normalized);
            self.record_mlp(&mut flow, &base, normalized, sequence, mixed);
            let next = flow.buffer(sequence * hidden);
            flow.add(summed, mixed, next);
            current = next;
        }

        // --- the final norm and the output head
        let final_norm = flow.bound(weights::expect("model.language_model.norm.weight").values());
        let final_hidden = flow.buffer(sequence * hidden);
        flow.rms_norm(current, sequence, hidden, final_norm, eps, final_hidden);
        let last = flow.buffer(hidden);
        flow.gather(final_hidden, 1, 1, hidden, (sequence - 1) * hidden, hidden, last);
        let logits = flow.buffer(manifest::VOCAB_SIZE);
        let head = flow.tensor("model.language_model.embed_tokens.weight");
        flow.linear(last, 1, head, logits);

        Recorded {
            spec: flow.finish(),
            state: carried,
            logits,
            hidden: last,
        }
    }

    /// Run a recorded pass on this machine.
    pub fn execute(&self, recorded: &Recorded) -> Arena {
        let mut arena = Arena::with_state(&recorded.spec, recorded.state.clone());
        crate::backend::execute(&Container, &recorded.spec, &mut arena);
        arena
    }

    #[allow(clippy::too_many_arguments)]
    fn record_mlp(&self, flow: &mut Flow, base: &str, x: Slot, sequence: usize, out: Slot) {
        let intermediate = manifest::INTERMEDIATE_SIZE;
        let gate = flow.buffer(sequence * intermediate);
        let up = flow.buffer(sequence * intermediate);
        let activated = flow.buffer(sequence * intermediate);
        let gate_weight = flow.tensor(&format!("{base}.mlp.gate_proj.weight"));
        flow.linear(x, sequence, gate_weight, gate);
        let up_weight = flow.tensor(&format!("{base}.mlp.up_proj.weight"));
        flow.linear(x, sequence, up_weight, up);
        flow.silu_mul(gate, up, activated);
        let down_weight = flow.tensor(&format!("{base}.mlp.down_proj.weight"));
        flow.linear(activated, sequence, down_weight, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_full_attention(
        &self,
        flow: &mut Flow,
        base: &str,
        x: Slot,
        sequence: usize,
        past: usize,
        cos: Slot,
        sin: Slot,
        cos_key: Slot,
        sin_key: Slot,
        keys: Slot,
        values: Slot,
        out: Slot,
    ) {
        let heads = manifest::NUM_ATTENTION_HEADS;
        let kv_heads = manifest::NUM_KEY_VALUE_HEADS;
        let width = manifest::HEAD_DIM;
        let rotary = manifest::ROTARY_DIM;

        let packed = flow.buffer(sequence * heads * 2 * width);
        let q_weight = flow.tensor(&format!("{base}.self_attn.q_proj.weight"));
        flow.linear(x, sequence, q_weight, packed);
        let query = flow.buffer(sequence * heads * width);
        let gate = flow.buffer(sequence * heads * width);
        flow.split(packed, sequence * heads, width, query, gate);

        let q_norm = flow.tensor(&format!("{base}.self_attn.q_norm.weight"));
        let query_normed = flow.buffer(sequence * heads * width);
        flow.rms_norm(query, sequence * heads, width, q_norm, manifest::RMS_NORM_EPS, query_normed);

        let key = flow.buffer(sequence * kv_heads * width);
        let k_weight = flow.tensor(&format!("{base}.self_attn.k_proj.weight"));
        flow.linear(x, sequence, k_weight, key);
        let k_norm = flow.tensor(&format!("{base}.self_attn.k_norm.weight"));
        let key_normed = flow.buffer(sequence * kv_heads * width);
        flow.rms_norm(key, sequence * kv_heads, width, k_norm, manifest::RMS_NORM_EPS, key_normed);

        let value = flow.buffer(sequence * kv_heads * width);
        let v_weight = flow.tensor(&format!("{base}.self_attn.v_proj.weight"));
        flow.linear(x, sequence, v_weight, value);

        let query_roped = flow.buffer(sequence * heads * width);
        flow.rope(query_normed, cos, sin, sequence * heads, width, rotary, query_roped);
        let key_roped = flow.buffer(sequence * kv_heads * width);
        flow.rope(key_normed, cos_key, sin_key, sequence * kv_heads, width, rotary, key_roped);

        let attended = flow.buffer(sequence * heads * width);
        flow.attention(
            query_roped, key_roped, value, keys, values, sequence, heads, kv_heads, width, past,
            attended,
        );
        let gated = flow.buffer(sequence * heads * width);
        flow.gate_mul(attended, gate, gated);
        let o_weight = flow.tensor(&format!("{base}.self_attn.o_proj.weight"));
        flow.linear(gated, sequence, o_weight, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_linear_attention(
        &self,
        flow: &mut Flow,
        base: &str,
        _slot: usize,
        state_slots: &(Slot, Slot),
        x: Slot,
        sequence: usize,
        eps: f32,
        out: Slot,
    ) {
        let conv_dim = manifest::LINEAR_CONV_DIM;
        let key_width = manifest::LINEAR_KEY_HEAD_DIM;
        let value_width = manifest::LINEAR_VALUE_HEAD_DIM;
        let taps = manifest::LINEAR_CONV_KERNEL;
        let heads = manifest::LINEAR_KEY_HEADS;

        let mixed = flow.buffer(sequence * conv_dim);
        let qkv = flow.tensor(&format!("{base}.linear_attn.in_proj_qkv.weight"));
        flow.linear(x, sequence, qkv, mixed);

        let convolved = flow.buffer(sequence * conv_dim);
        let conv_weight = flow.bound(vector_at(base, "linear_attn.conv1d.weight"));
        flow.convolve(mixed, conv_weight, state_slots.0, taps, conv_dim, sequence, convolved);

        let query = flow.buffer(sequence * heads * key_width);
        let key = flow.buffer(sequence * heads * key_width);
        let value = flow.buffer(sequence * heads * value_width);
        flow.gather(convolved, sequence * heads, heads, conv_dim, 0, key_width, query);
        flow.gather(
            convolved,
            sequence * heads,
            heads,
            conv_dim,
            manifest::LINEAR_KEY_DIM,
            key_width,
            key,
        );
        flow.gather(
            convolved,
            sequence * heads,
            heads,
            conv_dim,
            2 * manifest::LINEAR_KEY_DIM,
            value_width,
            value,
        );

        let scale = 1.0 / (key_width as f32).sqrt();
        let query_normed = flow.buffer(sequence * heads * key_width);
        flow.l2_normalize(query, sequence * heads, key_width, 1e-6, scale, query_normed);
        let key_normed = flow.buffer(sequence * heads * key_width);
        flow.l2_normalize(key, sequence * heads, key_width, 1e-6, 1.0, key_normed);

        let z = flow.buffer(sequence * manifest::LINEAR_VALUE_DIM);
        let z_weight = flow.tensor(&format!("{base}.linear_attn.in_proj_z.weight"));
        flow.linear(x, sequence, z_weight, z);
        let b = flow.buffer(sequence * heads);
        let b_weight = flow.tensor(&format!("{base}.linear_attn.in_proj_b.weight"));
        flow.linear(x, sequence, b_weight, b);
        let a = flow.buffer(sequence * heads);
        let a_weight = flow.tensor(&format!("{base}.linear_attn.in_proj_a.weight"));
        flow.linear(x, sequence, a_weight, a);

        let beta = flow.buffer(sequence * heads);
        let decay = flow.buffer(sequence * heads);
        let a_log = flow.bound(vector_at(base, "linear_attn.A_log"));
        let dt_bias = flow.bound(vector_at(base, "linear_attn.dt_bias"));
        flow.gate_decay(b, a, a_log, dt_bias, sequence * heads, beta, decay);

        let core = flow.buffer(sequence * manifest::LINEAR_VALUE_DIM);
        flow.delta_rule(
            query_normed,
            key_normed,
            value,
            beta,
            decay,
            state_slots.1,
            sequence,
            heads,
            key_width,
            value_width,
            core,
        );

        let norm = flow.bound(vector_at(base, "linear_attn.norm.weight"));
        let normalized = flow.buffer(sequence * manifest::LINEAR_VALUE_DIM);
        flow.gated_rms_norm(
            core,
            z,
            sequence * heads,
            value_width,
            norm,
            eps,
            normalized,
        );
        let out_weight = flow.tensor(&format!("{base}.linear_attn.out_proj.weight"));
        flow.linear(normalized, sequence, out_weight, out);
    }
}

/// The weights a recorded pass reads.
struct Container;

impl crate::backend::Weights for Container {
    fn matrix(&self, name: &str) -> Matrix {
        weights::expect(name).matrix()
    }
}

/// The cache's per-head buffers as one flat `[position][head][width]` block.
fn gather_history(buffers: &[Vec<f32>], positions: usize, width: usize) -> Vec<f32> {
    let heads = buffers.len();
    let mut flat = vec![0.0f32; positions.max(1) * heads * width];
    if positions == 0 {
        return flat;
    }
    for (head, buffer) in buffers.iter().enumerate() {
        for position in 0..positions {
            let source = position * width;
            if source + width > buffer.len() {
                break;
            }
            let target = (position * heads + head) * width;
            flat[target..target + width].copy_from_slice(&buffer[source..source + width]);
        }
    }
    flat
}

/// The rope tables, laid out one row per `(token, head)`.
///
/// Every head of a token shares its angle, so the row is repeated: the kernel
/// indexes a row per rotated band and does not know about heads.
fn roped_positions(positions: &[[u32; 3]], heads: usize) -> (Vec<f32>, Vec<f32>) {
    let (cos, sin) = super::rope_tables(positions);
    let half = manifest::ROTARY_DIM / 2;
    let rows = positions.len() * heads;
    let mut cos_rows = vec![0.0f32; rows * half];
    let mut sin_rows = vec![0.0f32; rows * half];
    for token in 0..positions.len() {
        for head in 0..heads {
            let row = token * heads + head;
            cos_rows[row * half..(row + 1) * half].copy_from_slice(&cos[token]);
            sin_rows[row * half..(row + 1) * half].copy_from_slice(&sin[token]);
        }
    }
    (cos_rows, sin_rows)
}

/// Run a recorded pass on this machine, keeping the state.
///
/// Only the logits are asked for, so the arena shares storage between the
/// intermediates: a 512-token prefill declares a gigabyte of slots and holds a few
/// megabytes of them at a time.
pub fn execute_recorded(recorded: &Recorded) -> (Vec<f32>, Vec<Vec<f32>>) {
    let logits_slot = match recorded.logits {
        Slot::Buffer(index) => index,
        _ => panic!("the logits are an arena buffer"),
    };
    let mut arena = Arena::with_state_reading(
        &recorded.spec,
        recorded.state.clone(),
        &[logits_slot],
    );
    crate::backend::execute(&Container, &recorded.spec, &mut arena);
    let logits = arena.buffer(logits_slot).to_vec();
    let state = arena.take_state();
    (logits, state)
}

impl Model {
    /// [`Model::forward`], run through the recorded pass instead of one kernel at
    /// a time.
    ///
    /// The two have to agree bit for bit: the recording is the same arithmetic in
    /// the same order, so a difference is a defect in the recording, not a
    /// rounding.
    pub fn forward_recorded(&self, ids: &[u32], cache: &mut Cache) -> Vec<f32> {
        let sequence = ids.len();
        assert!(sequence > 0, "forward needs at least one token");
        let base = cache.length as i64 + cache.position_delta;
        let positions: Vec<[u32; 3]> = (0..sequence)
            .map(|token| {
                let position = (base + token as i64).max(0) as u32;
                [position; 3]
            })
            .collect();
        let recorded = self.record(ids, sequence, &positions, cache);
        execute_recorded(&recorded).0
    }

    /// One recorded step, carrying the state forward.
    ///
    /// `past` is how many positions the state holds, and the returned state is what
    /// the next step has to be given, so a generation loop never rebuilds from
    /// scratch.
    pub fn forward_recorded_step(
        &self,
        ids: &[u32],
        past: usize,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let sequence = ids.len();
        assert!(sequence > 0, "forward needs at least one token");
        let positions: Vec<[u32; 3]> = (0..sequence)
            .map(|token| [(past + token) as u32; 3])
            .collect();
        let recorded = self.record_with_state(ids, sequence, &positions, past, state);
        execute_recorded(&recorded)
    }
}

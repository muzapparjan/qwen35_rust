//! The text model.
//!
//! 24 decoder layers: 18 run Gated DeltaNet and 6 run full attention, in the
//! pattern the checkpoint's `layer_types` gives. The operator order and the
//! shapes are the ones in `docs/op-spec.md`; every step there names the line of
//! the reference it came from.
//!
//! The same `forward` serves a whole prompt and a single token, because both go
//! through the same cached state: prefill and decode cannot drift apart when
//! there is one implementation of each layer.

use crate::ops;
use crate::weights::{self, manifest, Matrix};

/// Something that records intermediate tensors.
///
/// `NoTrace` is what a normal call passes: the recording is compiled in but
/// switched off, so the layer code has one shape rather than two.
pub trait Trace {
    /// Whether `record` will do anything; callers skip building names when it
    /// will not.
    fn enabled(&self) -> bool {
        true
    }

    fn record(&mut self, name: &str, shape: &[usize], values: &[f32]);
}

/// A recorder that keeps nothing.
pub struct NoTrace;

impl Trace for NoTrace {
    fn enabled(&self) -> bool {
        false
    }

    fn record(&mut self, _name: &str, _shape: &[usize], _values: &[f32]) {}
}

impl<F: FnMut(&str, &[usize], &[f32])> Trace for F {
    fn record(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        self(name, shape, values)
    }
}

macro_rules! record {
    ($trace:expr, $shape:expr, $values:expr, $($name:tt)*) => {
        if $trace.enabled() {
            $trace.record(&format!($($name)*), $shape, $values);
        }
    };
}

/// The state a generation carries between calls: one KV pair per full-attention
/// layer and one convolution plus recurrent state per linear one.
pub struct Cache {
    pub attention: Vec<AttentionCache>,
    pub linear: Vec<LinearCache>,
    /// Tokens already processed.
    pub length: usize,
}

/// Key and value for one full-attention layer: one buffer per head, laid out
/// `[position][dim]`.
///
/// Per head rather than one flat buffer: the stride of a flat buffer would be the
/// current token count, so growing the cache would move every head's data.
pub struct AttentionCache {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub positions: usize,
}

/// State for one linear-attention layer.
pub struct LinearCache {
    /// The last `kernel` inputs of the causal convolution, per channel:
    /// `conv[channel * kernel + offset]`, oldest first.
    pub conv: Vec<f32>,
    /// The recurrent state, `[value head][key dim][value dim]`.
    pub state: Vec<f32>,
}

impl Cache {
    pub fn new() -> Cache {
        Cache {
            attention: (0..manifest::FULL_LAYERS.len()).map(|_| AttentionCache::new()).collect(),
            linear: (0..manifest::LINEAR_LAYERS.len()).map(|_| LinearCache::new()).collect(),
            length: 0,
        }
    }

    /// Forget everything, as if no token had been processed.
    pub fn reset(&mut self) {
        for cache in &mut self.attention {
            for buffer in cache.keys.iter_mut().chain(cache.values.iter_mut()) {
                buffer.clear();
            }
            cache.positions = 0;
        }
        for cache in &mut self.linear {
            cache.conv.fill(0.0);
            cache.state.fill(0.0);
        }
        self.length = 0;
    }
}

impl Default for Cache {
    fn default() -> Cache {
        Cache::new()
    }
}

impl AttentionCache {
    fn new() -> AttentionCache {
        AttentionCache {
            keys: vec![Vec::new(); manifest::NUM_KEY_VALUE_HEADS],
            values: vec![Vec::new(); manifest::NUM_KEY_VALUE_HEADS],
            positions: 0,
        }
    }

    fn reserve(&mut self, positions: usize) {
        let needed = positions * manifest::HEAD_DIM;
        for buffer in self.keys.iter_mut().chain(self.values.iter_mut()) {
            if buffer.len() < needed {
                buffer.resize(needed, 0.0);
            }
        }
    }
}

impl LinearCache {
    fn new() -> LinearCache {
        LinearCache {
            conv: vec![0.0; manifest::LINEAR_CONV_DIM * manifest::LINEAR_CONV_KERNEL],
            state: vec![0.0; manifest::LINEAR_VALUE_HEADS * manifest::LINEAR_KEY_HEAD_DIM * manifest::LINEAR_VALUE_HEAD_DIM],
        }
    }
}

/// The model. The weights are embedded, so this holds nothing.
pub struct Model;

impl Default for Model {
    fn default() -> Model {
        Model
    }
}

/// One embedding row, for the lookup and for the tied output head.
fn embedding_matrix() -> Matrix {
    weights::expect("model.language_model.embed_tokens.weight").matrix()
}

fn prefix(layer: usize, suffix: &str) -> String {
    format!("model.language_model.layers.{layer}.{suffix}")
}

fn matrix(layer: usize, suffix: &str) -> Matrix {
    weights::expect(&prefix(layer, suffix)).matrix()
}

fn vector(layer: usize, suffix: &str) -> Vec<f32> {
    weights::expect(&prefix(layer, suffix)).values()
}

impl Model {
    pub fn new() -> Model {
        // Fail here rather than inside the first layer when the container and
        // the code disagree about what a layer is called.
        for layer in manifest::FULL_LAYERS {
            matrix(layer, "self_attn.q_proj.weight");
        }
        for layer in manifest::LINEAR_LAYERS {
            matrix(layer, "linear_attn.in_proj_qkv.weight");
        }
        Model
    }

    /// Run `ids` after whatever is already in `cache` and return the logits of
    /// the last position.
    pub fn forward<T: Trace>(&self, ids: &[u32], cache: &mut Cache, trace: &mut T) -> Vec<f32> {
        let sequence = ids.len();
        assert!(sequence > 0, "forward needs at least one token");
        let hidden = manifest::HIDDEN_SIZE;
        let eps = manifest::RMS_NORM_EPS;

        // --- embeddings
        let embed = embedding_matrix();
        let mut x = vec![0.0f32; sequence * hidden];
        for (token, id) in ids.iter().enumerate() {
            ops::row_values(&embed, *id as usize, &mut x[token * hidden..(token + 1) * hidden]);
        }
        record!(trace, &[sequence, hidden], &x, "embed");

        // --- layers
        let mut normalized = vec![0.0f32; sequence * hidden];
        let mut mixed = vec![0.0f32; sequence * hidden];
        for layer in 0..manifest::NUM_HIDDEN_LAYERS {
            if trace.enabled() {
                trace.record(&format!("layers.{layer}.in"), &[sequence, hidden], &x);
            }
            ops::rms_norm_zero_centered(
                &x,
                sequence,
                hidden,
                &vector(layer, "input_layernorm.weight"),
                eps,
                &mut normalized,
            );
            record!(trace, &[sequence, hidden], &normalized, "layers.{}.input_layernorm.out", layer);

            if manifest::LINEAR_LAYERS.contains(&layer) {
                let slot = manifest::LINEAR_LAYERS.iter().position(|value| *value == layer).unwrap();
                self.linear_attention(layer, &mut cache.linear[slot], &normalized, sequence, trace, &mut mixed);
                record!(trace, &[sequence, hidden], &mixed, "layers.{}.linear_attn.out", layer);
            } else {
                let slot = manifest::FULL_LAYERS.iter().position(|value| *value == layer).unwrap();
                self.full_attention(
                    layer,
                    &mut cache.attention[slot],
                    &normalized,
                    sequence,
                    cache.length,
                    trace,
                    &mut mixed,
                );
                record!(trace, &[sequence, hidden], &mixed, "layers.{}.self_attn.out", layer);
            }

            for index in 0..x.len() {
                x[index] += mixed[index];
            }

            ops::rms_norm_zero_centered(
                &x,
                sequence,
                hidden,
                &vector(layer, "post_attention_layernorm.weight"),
                eps,
                &mut normalized,
            );
            record!(
                trace,
                &[sequence, hidden],
                &normalized,
                "layers.{}.post_attention_layernorm.out",
                layer
            );

            self.mlp(layer, &normalized, sequence, &mut mixed);
            for index in 0..x.len() {
                x[index] += mixed[index];
            }
            record!(trace, &[sequence, hidden], &x, "layers.{}.out", layer);
        }
        cache.length += sequence;

        // --- final norm and the tied output head
        let mut final_hidden = vec![0.0f32; sequence * hidden];
        ops::rms_norm_zero_centered(
            &x,
            sequence,
            hidden,
            &weights::expect("model.language_model.norm.weight").values(),
            eps,
            &mut final_hidden,
        );
        record!(trace, &[sequence, hidden], &final_hidden, "final_norm.out");

        let mut logits = vec![0.0f32; manifest::VOCAB_SIZE];
        ops::linear(
            &final_hidden[(sequence - 1) * hidden..],
            1,
            &embed,
            None,
            &mut logits,
        );
        record!(trace, &[manifest::VOCAB_SIZE], &logits, "logits_last");
        logits
    }

    /// `mlp(x) = down(silu(gate(x)) * up(x))`; writes into `out`.
    fn mlp(&self, layer: usize, x: &[f32], sequence: usize, out: &mut [f32]) {
        let hidden = manifest::HIDDEN_SIZE;
        let intermediate = manifest::INTERMEDIATE_SIZE;
        let gate = matrix(layer, "mlp.gate_proj.weight");
        let up = matrix(layer, "mlp.up_proj.weight");
        let down = matrix(layer, "mlp.down_proj.weight");
        let mut gate_values = vec![0.0f32; sequence * intermediate];
        let mut up_values = vec![0.0f32; sequence * intermediate];
        ops::linear(x, sequence, &gate, None, &mut gate_values);
        ops::linear(x, sequence, &up, None, &mut up_values);
        for index in 0..gate_values.len() {
            gate_values[index] = ops::silu(gate_values[index]) * up_values[index];
        }
        ops::linear(&gate_values, sequence, &down, None, out[..sequence * hidden].as_mut());
    }

    /// Gated DeltaNet, one token at a time.
    #[allow(clippy::too_many_arguments)]
    fn linear_attention<T: Trace>(
        &self,
        layer: usize,
        cache: &mut LinearCache,
        x: &[f32],
        sequence: usize,
        trace: &mut T,
        out: &mut [f32],
    ) {
        let hidden = manifest::HIDDEN_SIZE;
        let conv_dim = manifest::LINEAR_CONV_DIM;
        let key_dim = manifest::LINEAR_KEY_DIM;
        let value_dim = manifest::LINEAR_VALUE_DIM;
        let heads = manifest::LINEAR_VALUE_HEADS;
        let key_head_dim = manifest::LINEAR_KEY_HEAD_DIM;
        let value_head_dim = manifest::LINEAR_VALUE_HEAD_DIM;
        let kernel = manifest::LINEAR_CONV_KERNEL;

        // --- projections, then the causal depthwise convolution
        let mut mixed = vec![0.0f32; sequence * conv_dim];
        ops::linear(x, sequence, &matrix(layer, "linear_attn.in_proj_qkv.weight"), None, &mut mixed);

        let conv_weight = vector(layer, "linear_attn.conv1d.weight"); // [conv_dim, 1, kernel]
        let mut convolved = vec![0.0f32; sequence * conv_dim];
        // The window of token `t` reaches back `kernel - 1` places, so the first
        // tokens of a sequence take part of their input from the cache. Both that
        // history and the weights are laid out tap-major here, which makes every
        // read of a tap a contiguous run and lets the tokens be computed
        // independently.
        let mut history = vec![0.0f32; kernel * conv_dim];
        for channel in 0..conv_dim {
            for slot in 0..kernel {
                history[slot * conv_dim + channel] = cache.conv[channel * kernel + slot];
            }
        }
        let mut planes = vec![0.0f32; kernel * conv_dim];
        for channel in 0..conv_dim {
            for tap in 0..kernel {
                planes[tap * conv_dim + channel] = conv_weight[channel * kernel + tap];
            }
        }
        crate::parallel::pool().chunks_mut(conv_dim, &mut convolved, &|index, chunk| {
            let token = index;
            chunk.fill(0.0);
            for tap in 0..kernel {
                let back = kernel as isize - 1 - tap as isize;
                let source: &[f32] = if token as isize >= back {
                    let start = (token as isize - back) as usize * conv_dim;
                    &mixed[start..start + conv_dim]
                } else {
                    let slot = (token as isize + tap as isize + 1) as usize;
                    &history[slot * conv_dim..(slot + 1) * conv_dim]
                };
                let plane = &planes[tap * conv_dim..(tap + 1) * conv_dim];
                for (out, (value, weight)) in chunk.iter_mut().zip(source.iter().zip(plane)) {
                    *out += value * weight;
                }
            }
            for value in chunk.iter_mut() {
                *value = ops::silu(*value);
            }
        });
        // Carry the window forward: the cache keeps the last `kernel` inputs.
        for channel in 0..conv_dim {
            let state = &mut cache.conv[channel * kernel..(channel + 1) * kernel];
            if sequence >= kernel {
                for slot in 0..kernel {
                    state[slot] = mixed[(sequence - kernel + slot) * conv_dim + channel];
                }
            } else {
                let kept = kernel - sequence;
                for slot in 0..kept {
                    state[slot] = history[(sequence + slot) * conv_dim + channel];
                }
                for slot in 0..sequence {
                    state[kept + slot] = mixed[slot * conv_dim + channel];
                }
            }
        }
        // The reference records the convolution's output before it drops the
        // left context, so this one is the piece the comparison can use.
        record!(
            trace,
            &[sequence, conv_dim],
            &convolved,
            "layers.{}.linear_attn.conv1d.out",
            layer
        );

        // --- gates
        let mut z = vec![0.0f32; sequence * value_dim];
        ops::linear(x, sequence, &matrix(layer, "linear_attn.in_proj_z.weight"), None, &mut z);
        record!(trace, &[sequence, heads, value_head_dim], &z, "layers.{}.linear_attn.in_proj_z.out", layer);
        let mut b = vec![0.0f32; sequence * heads];
        ops::linear(x, sequence, &matrix(layer, "linear_attn.in_proj_b.weight"), None, &mut b);
        let mut a = vec![0.0f32; sequence * heads];
        ops::linear(x, sequence, &matrix(layer, "linear_attn.in_proj_a.weight"), None, &mut a);
        let a_log = vector(layer, "linear_attn.A_log");
        let dt_bias = vector(layer, "linear_attn.dt_bias");

        // --- the recurrence
        //
        // The heads are independent: each has its own state, and the token axis
        // of one head depends on nothing but that head. Walking tokens outside
        // heads would make every head wait for its predecessor's token, which is
        // what left this the one section of prefill that did not answer to more
        // workers. The work per head is written head-major and transposed
        // afterwards.
        let scale = 1.0 / (key_head_dim as f32).sqrt();
        let mut by_head = vec![0.0f32; heads * sequence * value_head_dim];
        let state_len = key_head_dim * value_head_dim;
        let head_len = sequence * value_head_dim;
        crate::parallel::pool().chunks_mut_pair(
            state_len,
            &mut cache.state,
            head_len,
            &mut by_head,
            &|head, state, out| {
                let mut query = vec![0.0f32; key_head_dim];
                let mut key = vec![0.0f32; key_head_dim];
                let mut value = vec![0.0f32; value_head_dim];
                let mut memory = vec![0.0f32; value_head_dim];
                let mut delta = vec![0.0f32; value_head_dim];
                for token in 0..sequence {
                    let base = token * conv_dim;
                    let decay_base = token * heads;
                    for index in 0..key_head_dim {
                        query[index] = convolved[base + head * key_head_dim + index];
                        key[index] = convolved[base + key_dim + head * key_head_dim + index];
                    }
                    for index in 0..value_head_dim {
                        value[index] = convolved[base + 2 * key_dim + head * value_head_dim + index];
                    }
                    ops::l2_normalize(&mut query, 1e-6);
                    ops::l2_normalize(&mut key, 1e-6);
                    for index in 0..key_head_dim {
                        query[index] *= scale;
                    }

                    let beta = ops::sigmoid(b[decay_base + head]);
                    let decay = (-(a_log[head].exp())
                        * ops::softplus(a[decay_base + head] + dt_bias[head]))
                    .exp();
                    ops::scale_all(state, decay);
                    // memory = S^T k
                    ops::transposed_matvec(state, key_head_dim, value_head_dim, &key, &mut memory);
                    for index in 0..value_head_dim {
                        delta[index] = (value[index] - memory[index]) * beta;
                    }
                    ops::outer_add(state, key_head_dim, value_head_dim, &key, &delta);
                    // output = S^T q
                    ops::transposed_matvec(state, key_head_dim, value_head_dim, &query, &mut memory);
                    out[token * value_head_dim..(token + 1) * value_head_dim]
                        .copy_from_slice(&memory);
                }
            },
        );
        let mut core = vec![0.0f32; sequence * value_dim];
        for head in 0..heads {
            for token in 0..sequence {
                let source = (head * sequence + token) * value_head_dim;
                let target = (token * heads + head) * value_head_dim;
                core[target..target + value_head_dim]
                    .copy_from_slice(&by_head[source..source + value_head_dim]);
            }
        }
        record!(
            trace,
            &[sequence, heads, value_head_dim],
            &core,
            "layers.{}.linear_attn.delta.out",
            layer
        );

        // --- gated normalisation over the value head, then the output projection
        let mut normalized = vec![0.0f32; sequence * value_dim];
        ops::rms_norm_gated(
            &core,
            &z,
            sequence * heads,
            value_head_dim,
            &vector(layer, "linear_attn.norm.weight"),
            manifest::RMS_NORM_EPS,
            &mut normalized,
        );
        record!(
            trace,
            &[sequence, heads, value_head_dim],
            &normalized,
            "layers.{}.linear_attn.norm.out",
            layer
        );
        ops::linear(
            &normalized,
            sequence,
            &matrix(layer, "linear_attn.out_proj.weight"),
            None,
            &mut out[..sequence * hidden],
        );
    }

    /// Full attention with a KV cache, four query heads per key/value head.
    #[allow(clippy::too_many_arguments)]
    fn full_attention<T: Trace>(
        &self,
        layer: usize,
        cache: &mut AttentionCache,
        x: &[f32],
        sequence: usize,
        past: usize,
        trace: &mut T,
        out: &mut [f32],
    ) {
        let hidden = manifest::HIDDEN_SIZE;
        let heads = manifest::NUM_ATTENTION_HEADS;
        let kv_heads = manifest::NUM_KEY_VALUE_HEADS;
        let head_dim = manifest::HEAD_DIM;
        let rotary = manifest::ROTARY_DIM;
        let groups = heads / kv_heads;

        // --- query and gate come out of one projection, interleaved per head
        let mut packed = vec![0.0f32; sequence * heads * 2 * head_dim];
        ops::linear(x, sequence, &matrix(layer, "self_attn.q_proj.weight"), None, &mut packed);
        let mut query = vec![0.0f32; sequence * heads * head_dim];
        let mut gate = vec![0.0f32; sequence * heads * head_dim];
        for token in 0..sequence {
            for head in 0..heads {
                let source = (token * heads + head) * 2 * head_dim;
                let target = (token * heads + head) * head_dim;
                query[target..target + head_dim].copy_from_slice(&packed[source..source + head_dim]);
                gate[target..target + head_dim]
                    .copy_from_slice(&packed[source + head_dim..source + 2 * head_dim]);
            }
        }

        let query_norm = vector(layer, "self_attn.q_norm.weight");
        let mut normalized_query = vec![0.0f32; query.len()];
        ops::rms_norm_zero_centered(
            &query,
            sequence * heads,
            head_dim,
            &query_norm,
            manifest::RMS_NORM_EPS,
            &mut normalized_query,
        );
        let mut query = normalized_query;
        record!(
            trace,
            &[sequence, heads, head_dim],
            &query,
            "layers.{}.self_attn.q_norm.out",
            layer
        );

        let mut key = vec![0.0f32; sequence * kv_heads * head_dim];
        ops::linear(x, sequence, &matrix(layer, "self_attn.k_proj.weight"), None, &mut key);
        let mut normalized_key = vec![0.0f32; key.len()];
        ops::rms_norm_zero_centered(
            &key,
            sequence * kv_heads,
            head_dim,
            &vector(layer, "self_attn.k_norm.weight"),
            manifest::RMS_NORM_EPS,
            &mut normalized_key,
        );
        let mut key = normalized_key;
        record!(
            trace,
            &[sequence, kv_heads, head_dim],
            &key,
            "layers.{}.self_attn.k_norm.out",
            layer
        );

        let mut value = vec![0.0f32; sequence * kv_heads * head_dim];
        ops::linear(x, sequence, &matrix(layer, "self_attn.v_proj.weight"), None, &mut value);

        // --- rotation
        let (cos, sin) = rope_tables(past, sequence);
        for token in 0..sequence {
            for head in 0..heads {
                let base = (token * heads + head) * head_dim;
                ops::apply_rope(&mut query[base..base + head_dim], rotary, &cos[token], &sin[token]);
            }
            for head in 0..kv_heads {
                let base = (token * kv_heads + head) * head_dim;
                ops::apply_rope(&mut key[base..base + head_dim], rotary, &cos[token], &sin[token]);
            }
        }

        // --- append to the cache
        let total = past + sequence;
        cache.reserve(total);
        for token in 0..sequence {
            for head in 0..kv_heads {
                let source = (token * kv_heads + head) * head_dim;
                let target = (past + token) * head_dim;
                cache.keys[head][target..target + head_dim].copy_from_slice(&key[source..source + head_dim]);
                cache.values[head][target..target + head_dim]
                    .copy_from_slice(&value[source..source + head_dim]);
            }
        }
        cache.positions = total;

        // --- attention over the cached prefix, causal
        //
        // A (token, head) pair depends on nothing else: the prefix is already in
        // the cache and the pairs only read it. Chunking `attended` by head width
        // gives exactly that pairing, so the split needs no transpose.
        let scaling = 1.0 / (head_dim as f32).sqrt();
        let mut attended = vec![0.0f32; sequence * heads * head_dim];
        crate::parallel::pool().chunks_mut(head_dim, &mut attended, &|index, target| {
            let token = index / heads;
            let head = index % heads;
            let visible = past + token + 1;
            let kv_head = head / groups;
            let base = (token * heads + head) * head_dim;
            let mut scores = vec![0.0f32; visible];
            for position in 0..visible {
                let other = position * head_dim;
                scores[position] = ops::dot(
                    &query[base..base + head_dim],
                    &cache.keys[kv_head][other..other + head_dim],
                ) * scaling;
            }
            ops::softmax(&mut scores);
            target.fill(0.0);
            for position in 0..visible {
                let weight = scores[position];
                let other = position * head_dim;
                let values = &cache.values[kv_head][other..other + head_dim];
                for (slot, value) in target.iter_mut().zip(values) {
                    *slot += weight * value;
                }
            }
        });

        // --- the gate, then the output projection
        for index in 0..attended.len() {
            attended[index] *= ops::sigmoid(gate[index]);
        }
        ops::linear(
            &attended,
            sequence,
            &matrix(layer, "self_attn.o_proj.weight"),
            None,
            &mut out[..sequence * hidden],
        );
    }
}

/// `cos`/`sin` for `sequence` positions starting at `past`, 32 values each.
///
/// With text input the three position rows are identical, so the mRoPE
/// interleaving reduces to one angle per frequency; `rope_tables` still takes the
/// three rows because an image carries different ones.
fn rope_tables(past: usize, sequence: usize) -> (Vec<[f32; 32]>, Vec<[f32; 32]>) {
    let rotary = manifest::ROTARY_DIM;
    let half = rotary / 2;
    let mut frequencies = Vec::with_capacity(half);
    for index in 0..half {
        let exponent = (2 * index) as f64 / rotary as f64;
        frequencies.push((1.0 / manifest::ROPE_THETA.powf(exponent)) as f32);
    }
    let mut cos = Vec::with_capacity(sequence);
    let mut sin = Vec::with_capacity(sequence);
    for token in 0..sequence {
        let position = (past + token) as f32;
        let mut cosine = [0.0f32; 32];
        let mut sine = [0.0f32; 32];
        for index in 0..half {
            let angle = position * frequencies[index];
            cosine[index] = angle.cos();
            sine[index] = angle.sin();
        }
        cos.push(cosine);
        sin.push(sine);
    }
    (cos, sin)
}

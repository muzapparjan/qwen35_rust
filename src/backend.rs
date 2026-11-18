//! A recorded sequence of kernels, and the portable executor for it.
//!
//! A backend that runs on another device has to be handed the whole pass, not one
//! operator at a time: each round trip to a device costs about half a millisecond
//! in submission and read-back, and one forward pass is roughly two hundred
//! operators, so a per-operator backend spends more time waiting than computing.
//! [`Flow`] therefore records the pass — which kernels, on which buffers, with
//! which weights — and a backend executes the recording as one submission,
//! reading results back once at the end.
//!
//! The kernels are the ones this model needs and no others: a linear projection,
//! a zero-centred RMS norm, the gate of an MLP, and the argmax over a row.
//!
//! Intermediate values live in buffers the pass allocates as it records, so a
//! recorded pass is a dataflow graph rather than a list of calls on host slices.
//! A weight is either a tensor of the container, addressed by name, or a dense
//! vector the caller bound once — the same reuse the host path gets by holding a
//! norm weight across calls.
//!
//! A slot is not a buffer: [`layout`] hands out storage by lifetime, so a pass costs
//! its high-water mark and not the sum of everything it names. A prefill of two
//! thousand tokens declares seven gigabytes of intermediates and runs in two
//! hundred megabytes. The slots the caller reads back keep storage of their own,
//! because a shared one would be overwritten by a later operator before the caller
//! looked at it.

use crate::weights::Matrix;

/// Where a kernel reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// A buffer in the pass' arena.
    Buffer(usize),
    /// A dense vector the caller bound; read-only.
    Bound(usize),
    /// A buffer the caller owns across calls: a cache, or a recurrent state.
    External(usize),
}

/// A matrix or vector operand that is a weight rather than a data value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    /// A quantised tensor of the container.
    Tensor(&'static str),
    /// A dense vector the caller bound.
    Bound(usize),
}

/// One kernel.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    /// `out = weight` expanded from four-bit values, in row-major order.
    Dequantize { weight: Weight, out: Slot },
    /// `out = weight · input`, one token per row of `input`.
    Linear {
        input: Slot,
        tokens: usize,
        weight: Weight,
        out: Slot,
    },
    /// `out = x / rms(x) * (1 + weight)`, one row per `width` values.
    RmsNorm {
        input: Slot,
        rows: usize,
        width: usize,
        weight: Weight,
        eps: f32,
        out: Slot,
    },
    /// `out = silu(gate) * up`, elementwise.
    SiluMul { gate: Slot, up: Slot, out: Slot },
    /// `out[i] = in[i] * factor`.
    Scale { input: Slot, factor: f32, out: Slot },
    /// Copy one band out of a `[token][band][head][width]` block.
    ///
    /// `out[row][i] = input[(row / group) * stride + offset + (row % group) * width + i]`,
    /// where `row` runs over `(token, head)` pairs. The bands of a token are
    /// interleaved per head, so a single stride cannot express the copy.
    Gather {
        input: Slot,
        rows: usize,
        group: usize,
        stride: usize,
        offset: usize,
        width: usize,
        out: Slot,
    },
    /// `out = a + b`, elementwise.
    Add { a: Slot, b: Slot, out: Slot },
    /// `out = a * sigmoid(b)`, elementwise; the output gate of attention.
    GateMul { a: Slot, b: Slot, out: Slot },
    /// The embedding rows `ids` names, in order.
    Embed {
        table: Weight,
        ids: Slot,
        rows: usize,
        width: usize,
        out: Slot,
    },
    /// Split every `2 * width` row of `input` into its first and second halves.
    ///
    /// The query and its gate come out of one projection interleaved per head, so
    /// a row is `[query head | gate head]`.
    Split {
        input: Slot,
        rows: usize,
        width: usize,
        first: Slot,
        second: Slot,
    },
    /// `out = weight * x / rms(x) * silu(gate)`, one row per `width` values.
    ///
    /// This is the gate of a linear-attention layer: a per-row reduction, then a
    /// per-element multiply.
    GatedRmsNorm {
        input: Slot,
        gate: Slot,
        rows: usize,
        width: usize,
        weight: Weight,
        eps: f32,
        out: Slot,
    },
    /// `out = (x - mean) / sqrt(variance + eps) * (1 + weight) + bias`.
    LayerNorm {
        input: Slot,
        rows: usize,
        width: usize,
        weight: Weight,
        bias: Weight,
        eps: f32,
        out: Slot,
    },
    /// `out = softmax(input)` over each row of `width`.
    Softmax {
        input: Slot,
        rows: usize,
        width: usize,
        out: Slot,
    },
    /// The Gated DeltaNet recurrence.
    ///
    /// Each of `query`, `key` and `value` holds one row per `(token, head)` pair,
    /// already normalised; `beta` and `decay` hold one value per pair; `state` is
    /// the carried state, `heads x key_width x value_width`, read and written in
    /// place exactly as the host path does.
    #[allow(clippy::too_many_arguments)]
    DeltaRule {
        query: Slot,
        key: Slot,
        value: Slot,
        beta: Slot,
        decay: Slot,
        state: Slot,
        tokens: usize,
        heads: usize,
        key_width: usize,
        value_width: usize,
        out: Slot,
    },
    /// `l2(x) * scale` over each row of `width`; the delta rule's key and query
    /// normalisation.
    L2Normalize {
        input: Slot,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
        out: Slot,
    },
    /// `rotate_half` with the pairing `(i, i + half)`.
    Rope {
        input: Slot,
        cos: Slot,
        sin: Slot,
        rows: usize,
        width: usize,
        rotary: usize,
        out: Slot,
    },
    /// `out[0, lane] = sum over rows`; a reduction over one contiguous run.
    Sum { input: Slot, length: usize, out: Slot },
    /// The causal depthwise convolution of a linear-attention layer.
    ///
    /// `weights` is `[channels, kernel]` and `state` holds the last `kernel`
    /// inputs of every channel, oldest first, laid out `[channel][kernel]`; the
    /// kernel carries the window forward in place.
    #[allow(clippy::too_many_arguments)]
    Convolve {
        input: Slot,
        weights: Weight,
        state: Slot,
        taps: usize,
        channels: usize,
        tokens: usize,
        out: Slot,
    },
    /// The beta and decay of a linear-attention layer.
    ///
    /// `beta = sigmoid(b)` and `decay = exp(-exp(a_log) * softplus(a + dt_bias))`,
    /// one value per `(token, head)`.
    GateDecay {
        b: Slot,
        a: Slot,
        a_log: Weight,
        dt_bias: Weight,
        rows: usize,
        beta: Slot,
        decay: Slot,
    },
    /// Attention over a cached prefix, causal, with an output gate.
    ///
    /// `keys` and `values` hold `[position][head][width]` with `capacity`
    /// positions allocated; `past` is how many are filled.
    #[allow(clippy::too_many_arguments)]
    Attention {
        query: Slot,
        key: Slot,
        value: Slot,
        keys: Slot,
        values: Slot,
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        width: usize,
        past: usize,
        out: Slot,
    },
    /// `out[row] = argmax(input[row])`, one value per row.
    ///
    /// A token index is below 2^24, so it is exact in `f32` and needs no second
    /// kind of buffer.
    Argmax {
        input: Slot,
        rows: usize,
        width: usize,
        out: Slot,
    },
}

impl Op {
    /// The slots this operator reads.
    ///
    /// Exhaustive on purpose: an operator added without a line here would look, to
    /// [`layout`], like it reads nothing, and its input's storage could then be
    /// handed to somebody else while it is still needed.
    pub fn reads(&self) -> Vec<Slot> {
        match *self {
            Op::Dequantize { weight: _, out: _ } => Vec::new(),
            Op::Linear { input, .. } => vec![input],
            Op::RmsNorm { input, .. } => vec![input],
            Op::SiluMul { gate, up, .. } => vec![gate, up],
            Op::Scale { input, .. } => vec![input],
            Op::Gather { input, .. } => vec![input],
            Op::Add { a, b, .. } => vec![a, b],
            Op::GateMul { a, b, .. } => vec![a, b],
            Op::Embed { ids, .. } => vec![ids],
            Op::Split { input, .. } => vec![input],
            Op::GatedRmsNorm { input, gate, .. } => vec![input, gate],
            Op::LayerNorm { input, .. } => vec![input],
            Op::Softmax { input, .. } => vec![input],
            Op::DeltaRule {
                query,
                key,
                value,
                beta,
                decay,
                state,
                ..
            } => vec![query, key, value, beta, decay, state],
            Op::L2Normalize { input, .. } => vec![input],
            Op::Rope {
                input, cos, sin, ..
            } => vec![input, cos, sin],
            Op::Sum { input, .. } => vec![input],
            Op::Convolve { input, state, .. } => vec![input, state],
            Op::GateDecay { b, a, .. } => vec![b, a],
            Op::Attention {
                query,
                key,
                value,
                keys,
                values,
                ..
            } => vec![query, key, value, keys, values],
            Op::Argmax { input, .. } => vec![input],
        }
    }

    /// The slots this operator writes, including the ones it also reads.
    pub fn writes(&self) -> Vec<Slot> {
        match *self {
            Op::Dequantize { out, .. } => vec![out],
            Op::Linear { out, .. } => vec![out],
            Op::RmsNorm { out, .. } => vec![out],
            Op::SiluMul { out, .. } => vec![out],
            Op::Scale { out, .. } => vec![out],
            Op::Gather { out, .. } => vec![out],
            Op::Add { out, .. } => vec![out],
            Op::GateMul { out, .. } => vec![out],
            Op::Embed { out, .. } => vec![out],
            Op::Split { first, second, .. } => vec![first, second],
            Op::GatedRmsNorm { out, .. } => vec![out],
            Op::LayerNorm { out, .. } => vec![out],
            Op::Softmax { out, .. } => vec![out],
            Op::DeltaRule { state, out, .. } => vec![state, out],
            Op::L2Normalize { out, .. } => vec![out],
            Op::Rope { out, .. } => vec![out],
            Op::Sum { out, .. } => vec![out],
            Op::Convolve { state, out, .. } => vec![state, out],
            Op::GateDecay { beta, decay, .. } => vec![beta, decay],
            Op::Attention {
                keys, values, out, ..
            } => vec![keys, values, out],
            Op::Argmax { out, .. } => vec![out],
        }
    }
}

/// Which storage each of a pass' slots runs in.
///
/// A recording declares a slot for every intermediate, and a pass only reads a few
/// of them at a time: the arena a 512-token prefill declares is 1.9 GB, while what
/// is live at any instant is a few megabytes. This hands a slot the storage of one
/// that nothing will touch again, which is why a pass costs its high-water mark and
/// not its sum.
///
/// Slots in `wanted` keep storage of their own and are never reused: a caller that
/// asks for a slot wants the value that slot ends with, and reusing its storage
/// after its last read would leave a later operator's output there instead.
pub struct Layout {
    storage: Vec<usize>,
    lengths: Vec<usize>,
}

impl Layout {
    /// The storage a slot runs in.
    pub fn storage(&self, slot: usize) -> usize {
        self.storage[slot]
    }

    /// The length of each storage, in storage-index order.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }
}

/// Lay out a pass' slots over as few storages as its live ranges allow.
pub fn layout(spec: &Spec, wanted: &[bool]) -> Layout {
    let count = spec.buffers.len();
    assert_eq!(wanted.len(), count, "one wanted flag per slot");

    // The last operator that touches each slot, read or written.
    let mut last = vec![usize::MAX; count];
    for (position, op) in spec.ops.iter().enumerate() {
        for slot in op.reads().into_iter().chain(op.writes()) {
            if let Slot::Buffer(index) = slot {
                last[index] = position;
            }
        }
    }

    let mut storage = vec![usize::MAX; count];
    let mut lengths: Vec<usize> = Vec::new();
    // The slot each storage currently belongs to.
    let mut holder: Vec<usize> = Vec::new();
    for (position, op) in spec.ops.iter().enumerate() {
        for slot in op.writes() {
            let Slot::Buffer(index) = slot else {
                continue;
            };
            if storage[index] != usize::MAX && holder[storage[index]] == index {
                // Written before and still holding its storage: an operator that
                // updates a buffer in place must keep working on the same one.
                continue;
            }
            let needed = spec.buffers[index].max(1);
            // The smallest storage whose owner nothing will touch again. Smallest
            // first, so a row's worth is not spent on a one-element slot.
            let mut pick: Option<usize> = None;
            for (at, owner) in holder.iter().enumerate() {
                if wanted[*owner] || last[*owner] >= position || lengths[at] < needed {
                    continue;
                }
                if pick.is_none_or(|best| lengths[at] < lengths[best]) {
                    pick = Some(at);
                }
            }
            let at = match pick {
                Some(at) => at,
                None => {
                    lengths.push(needed);
                    holder.push(index);
                    lengths.len() - 1
                }
            };
            storage[index] = at;
            holder[at] = index;
        }
    }

    // A slot no operator writes is still a slot; give it storage of its own rather
    // than a panic in the backend that asks for it.
    for index in 0..count {
        if storage[index] == usize::MAX {
            lengths.push(spec.buffers[index].max(1));
            storage[index] = lengths.len() - 1;
            holder.push(index);
        }
    }
    Layout { storage, lengths }
}

/// A recorded pass.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Quantised tensors the pass reads, by container name, deduplicated.
    pub tensors: Vec<String>,
    /// Dense vectors the caller bound, in binding order.
    pub vectors: Vec<Vec<f32>>,
    /// Buffer lengths, in arena-index order.
    pub buffers: Vec<usize>,
    /// State buffers the caller owns, in index order.
    pub externals: Vec<usize>,
    pub ops: Vec<Op>,
}

/// The weights a pass can name.
pub trait Weights {
    fn matrix(&self, name: &str) -> Matrix;
}

/// Records a pass, then hands it to a backend as one submission.
pub struct Flow {
    tensors: Vec<String>,
    vectors: Vec<Vec<f32>>,
    buffers: Vec<usize>,
    externals: Vec<usize>,
    ops: Vec<Op>,
}

impl Flow {
    pub fn new() -> Flow {
        Flow {
            tensors: Vec::new(),
            vectors: Vec::new(),
            buffers: Vec::new(),
            externals: Vec::new(),
            ops: Vec::new(),
        }
    }

    /// A scratch buffer of `length` values.
    pub fn buffer(&mut self, length: usize) -> Slot {
        self.buffers.push(length);
        Slot::Buffer(self.buffers.len() - 1)
    }

    /// A dense vector the pass reads, bound once.
    pub fn bind(&mut self, values: Vec<f32>) -> Slot {
        self.vectors.push(values);
        Slot::Bound(self.vectors.len() - 1)
    }

    /// The tensor operand for a container weight, recorded once.
    pub fn tensor(&mut self, name: &str) -> Weight {
        if !self.tensors.iter().any(|known| known == name) {
            self.tensors.push(name.to_string());
        }
        // The name comes from the manifest's static table, which is what makes the
        // operand `'static` rather than borrowed from the caller.
        let name: &'static str = crate::weights::spec(name)
            .map(|spec| spec.name)
            .unwrap_or_else(|| panic!("the container has no tensor called {name:?}"));
        Weight::Tensor(name)
    }

    /// The bound operand for a dense weight.
    pub fn bound(&mut self, values: Vec<f32>) -> Weight {
        self.vectors.push(values);
        Weight::Bound(self.vectors.len() - 1)
    }

    pub fn linear(&mut self, input: Slot, tokens: usize, weight: Weight, out: Slot) {
        self.ops.push(Op::Linear {
            input,
            tokens,
            weight,
            out,
        });
    }

    /// Expand a quantised tensor into `out`, whole rows at a time.
    pub fn dequantize(&mut self, weight: Weight, out: Slot) {
        self.ops.push(Op::Dequantize { weight, out });
    }

    pub fn rms_norm(
        &mut self,
        input: Slot,
        rows: usize,
        width: usize,
        weight: Weight,
        eps: f32,
        out: Slot,
    ) {
        self.ops.push(Op::RmsNorm {
            input,
            rows,
            width,
            weight,
            eps,
            out,
        });
    }

    pub fn silu_mul(&mut self, gate: Slot, up: Slot, out: Slot) {
        self.ops.push(Op::SiluMul { gate, up, out });
    }

    pub fn argmax(&mut self, input: Slot, rows: usize, width: usize, out: Slot) {
        self.ops.push(Op::Argmax {
            input,
            rows,
            width,
            out,
        });
    }

    pub fn scale(&mut self, input: Slot, factor: f32, out: Slot) {
        self.ops.push(Op::Scale {
            input,
            factor,
            out,
        });
    }

    /// A buffer the caller keeps across calls, such as a cache or a recurrent
    /// state. The first call registers the length.
    pub fn external(&mut self, index: usize, length: usize) -> Slot {
        if self.externals.len() <= index {
            self.externals.resize(index + 1, 0);
        }
        self.externals[index] = length;
        Slot::External(index)
    }

    pub fn add(&mut self, a: Slot, b: Slot, out: Slot) {
        self.ops.push(Op::Add { a, b, out });
    }

    pub fn gather(
        &mut self,
        input: Slot,
        rows: usize,
        group: usize,
        stride: usize,
        offset: usize,
        width: usize,
        out: Slot,
    ) {
        self.ops.push(Op::Gather {
            input,
            rows,
            group,
            stride,
            offset,
            width,
            out,
        });
    }

    pub fn gate_mul(&mut self, a: Slot, b: Slot, out: Slot) {
        self.ops.push(Op::GateMul { a, b, out });
    }

    pub fn embed(&mut self, table: Weight, ids: Slot, rows: usize, width: usize, out: Slot) {
        self.ops.push(Op::Embed {
            table,
            ids,
            rows,
            width,
            out,
        });
    }

    pub fn split(&mut self, input: Slot, rows: usize, width: usize, first: Slot, second: Slot) {
        self.ops.push(Op::Split {
            input,
            rows,
            width,
            first,
            second,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn convolve(
        &mut self,
        input: Slot,
        weights: Weight,
        state: Slot,
        taps: usize,
        channels: usize,
        tokens: usize,
        out: Slot,
    ) {
        self.ops.push(Op::Convolve {
            input,
            weights,
            state,
            taps,
            channels,
            tokens,
            out,
        });
    }

    pub fn gate_decay(
        &mut self,
        b: Slot,
        a: Slot,
        a_log: Weight,
        dt_bias: Weight,
        rows: usize,
        beta: Slot,
        decay: Slot,
    ) {
        self.ops.push(Op::GateDecay {
            b,
            a,
            a_log,
            dt_bias,
            rows,
            beta,
            decay,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        query: Slot,
        key: Slot,
        value: Slot,
        keys: Slot,
        values: Slot,
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        width: usize,
        past: usize,
        out: Slot,
    ) {
        self.ops.push(Op::Attention {
            query,
            key,
            value,
            keys,
            values,
            tokens,
            heads,
            kv_heads,
            width,
            past,
            out,
        });
    }

    pub fn gated_rms_norm(
        &mut self,
        input: Slot,
        gate: Slot,
        rows: usize,
        width: usize,
        weight: Weight,
        eps: f32,
        out: Slot,
    ) {
        self.ops.push(Op::GatedRmsNorm {
            input,
            gate,
            rows,
            width,
            weight,
            eps,
            out,
        });
    }

    pub fn layer_norm(
        &mut self,
        input: Slot,
        rows: usize,
        width: usize,
        weight: Weight,
        bias: Weight,
        eps: f32,
        out: Slot,
    ) {
        self.ops.push(Op::LayerNorm {
            input,
            rows,
            width,
            weight,
            bias,
            eps,
            out,
        });
    }

    pub fn softmax(&mut self, input: Slot, rows: usize, width: usize, out: Slot) {
        self.ops.push(Op::Softmax {
            input,
            rows,
            width,
            out,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn delta_rule(
        &mut self,
        query: Slot,
        key: Slot,
        value: Slot,
        beta: Slot,
        decay: Slot,
        state: Slot,
        tokens: usize,
        heads: usize,
        key_width: usize,
        value_width: usize,
        out: Slot,
    ) {
        self.ops.push(Op::DeltaRule {
            query,
            key,
            value,
            beta,
            decay,
            state,
            tokens,
            heads,
            key_width,
            value_width,
            out,
        });
    }

    pub fn l2_normalize(&mut self, input: Slot, rows: usize, width: usize, eps: f32, scale: f32, out: Slot) {
        self.ops.push(Op::L2Normalize {
            input,
            rows,
            width,
            eps,
            scale,
            out,
        });
    }

    pub fn rope(
        &mut self,
        input: Slot,
        cos: Slot,
        sin: Slot,
        rows: usize,
        width: usize,
        rotary: usize,
        out: Slot,
    ) {
        self.ops.push(Op::Rope {
            input,
            cos,
            sin,
            rows,
            width,
            rotary,
            out,
        });
    }

    pub fn sum(&mut self, input: Slot, length: usize, out: Slot) {
        self.ops.push(Op::Sum { input, length, out });
    }

    pub fn finish(self) -> Spec {
        Spec {
            tensors: self.tensors,
            vectors: self.vectors,
            buffers: self.buffers,
            externals: self.externals,
            ops: self.ops,
        }
    }
}

impl Default for Flow {
    fn default() -> Flow {
        Flow::new()
    }
}

/// The arena a pass runs in.
///
/// The storage is the pass' high-water mark, not the sum of its slots: a slot whose
/// value nothing will touch again hands its space to the next operator that needs
/// some. A slot the caller asked for keeps storage of its own, because the value it
/// ends with is what the caller reads.
pub struct Arena {
    storages: Vec<Vec<f32>>,
    /// The storage each slot runs in.
    mapping: Vec<usize>,
    /// The length each slot declared; a shared storage is usually longer.
    lengths: Vec<usize>,
    externals: Vec<Vec<f32>>,
}

impl Arena {
    /// An arena in which every slot keeps storage of its own, with room for the
    /// pass' state beside it.
    pub fn new(spec: &Spec) -> Arena {
        Arena::of(spec, Arena::all(spec), true)
    }

    /// An arena for a pass whose results are `outputs`.
    ///
    /// Only those slots keep storage of their own; a slot the caller never reads
    /// shares storage with a dead neighbour, so a prefill that declares gigabytes
    /// of intermediates costs its widest few megabytes. There is no room for state:
    /// a caller that wants the state back asks for it separately.
    pub fn reading(spec: &Spec, outputs: &[usize]) -> Arena {
        Arena::of(spec, Arena::wanted(spec, outputs), false)
    }

    /// An arena that carries the caller's state instead of zeroing it.
    ///
    /// Each buffer is padded to the length the pass declared, because a cache grows
    /// as the pass appends to it and the caller's copy is only as long as what it
    /// has seen so far.
    pub fn with_state(spec: &Spec, externals: Vec<Vec<f32>>) -> Arena {
        Arena::stateful(spec, externals, Arena::all(spec))
    }

    /// A state-carrying arena in which only `outputs` keep storage of their own.
    pub fn with_state_reading(spec: &Spec, externals: Vec<Vec<f32>>, outputs: &[usize]) -> Arena {
        Arena::stateful(spec, externals, Arena::wanted(spec, outputs))
    }

    fn all(spec: &Spec) -> Vec<bool> {
        vec![true; spec.buffers.len()]
    }

    fn wanted(spec: &Spec, outputs: &[usize]) -> Vec<bool> {
        let mut wanted = vec![false; spec.buffers.len()];
        for index in outputs {
            assert!(
                *index < spec.buffers.len(),
                "output slot {index} is past the pass' {} buffers",
                spec.buffers.len()
            );
            wanted[*index] = true;
        }
        wanted
    }

    fn stateful(spec: &Spec, externals: Vec<Vec<f32>>, wanted: Vec<bool>) -> Arena {
        let mut arena = Arena::of(spec, wanted, true);
        for (index, mut values) in externals.into_iter().enumerate() {
            if index >= arena.externals.len() {
                arena.externals.push(Vec::new());
            }
            let length = arena.externals[index].len().max(spec.externals[index]);
            values.resize(length, 0.0);
            arena.externals[index] = values;
        }
        arena
    }

    fn of(spec: &Spec, wanted: Vec<bool>, keep_state: bool) -> Arena {
        let laid = layout(spec, &wanted);
        Arena {
            storages: laid
                .lengths()
                .iter()
                .map(|length| vec![0.0f32; *length])
                .collect(),
            mapping: (0..spec.buffers.len()).map(|slot| laid.storage(slot)).collect(),
            lengths: spec.buffers.clone(),
            externals: if keep_state {
                spec.externals
                    .iter()
                    .map(|length| vec![0.0f32; *length])
                    .collect()
            } else {
                Vec::new()
            },
        }
    }

    /// The values of a slot.
    ///
    /// A slot that was not in the arena's `outputs` shares its storage with another
    /// slot, so what comes back is that storage's contents and not a value this
    /// slot ever held.
    pub fn buffer(&self, index: usize) -> &[f32] {
        &self.storages[self.mapping[index]][..self.lengths[index]]
    }

    pub fn state(&self, index: usize) -> &[f32] {
        &self.externals[index]
    }

    /// Take the caller's state back out, for the next call.
    pub fn take_state(self) -> Vec<Vec<f32>> {
        self.externals
    }

    /// Fill a buffer with results a backend computed.
    pub fn set_buffer(&mut self, index: usize, values: &[f32]) {
        let length = self.lengths[index];
        assert_eq!(length, values.len(), "buffer {index} length");
        let storage = self.mapping[index];
        self.storages[storage][..length].copy_from_slice(values);
    }

    /// How much memory the arena holds, in bytes, slot storage and state together.
    pub fn held_bytes(&self) -> usize {
        let slots: usize = self.storages.iter().map(|values| values.len() * 4).sum();
        let state: usize = self.externals.iter().map(|values| values.len() * 4).sum();
        slots + state
    }

    /// Fill a state buffer with what a backend wrote into it.
    pub fn set_state(&mut self, index: usize, values: &[f32]) {
        let target = &mut self.externals[index];
        let length = target.len().min(values.len());
        target[..length].copy_from_slice(&values[..length]);
    }
}

/// Run a recorded pass on this machine.
pub fn execute<W: Weights>(weights: &W, spec: &Spec, arena: &mut Arena) {
    // Two scratch buffers reused across operators: a pass holds a few hundred of
    // them, and allocating each one would cost more than the arithmetic.
    let mut scratch = Vec::new();
    let mut other = Vec::new();
    for op in &spec.ops {
        match *op {
            Op::Dequantize { weight, out } => {
                let values = match weight {
                    Weight::Tensor(name) => crate::weights::expect(name).values(),
                    Weight::Bound(_) => panic!("a bound value is not quantised"),
                };
                write(arena, out, &values);
            }
            Op::Linear {
                input,
                tokens,
                weight,
                out,
            } => {
                let matrix = matrix_of(weights, weight);
                let columns = matrix.columns;
                let values: Vec<f32> = read(spec, arena, input, tokens * columns).to_vec();
                scratch.resize(tokens * matrix.rows, 0.0);
                crate::ops::linear(&values, tokens, &matrix, None, &mut scratch);
                write(arena, out, &scratch);
            }
            Op::RmsNorm {
                input,
                rows,
                width,
                weight,
                eps,
                out,
            } => {
                let gain = weight_values(weights, spec, weight);
                let values: Vec<f32> = read(spec, arena, input, rows * width).to_vec();
                scratch.resize(rows * width, 0.0);
                crate::ops::rms_norm_zero_centered(&values, rows, width, &gain, eps, &mut scratch);
                write(arena, out, &scratch);
            }
            Op::SiluMul { gate, up, out } => {
                let gate_values: Vec<f32> = read(spec, arena, gate, usize::MAX).to_vec();
                let up_values = read(spec, arena, up, gate_values.len());
                other.resize(gate_values.len(), 0.0);
                for index in 0..gate_values.len() {
                    other[index] = crate::ops::silu(gate_values[index]) * up_values[index];
                }
                write(arena, out, &other);
            }
            Op::Argmax {
                input,
                rows,
                width,
                out,
            } => {
                let values: Vec<f32> = read(spec, arena, input, rows * width).to_vec();
                other.resize(rows, 0.0);
                for row in 0..rows {
                    let slice = &values[row * width..(row + 1) * width];
                    let mut best = 0usize;
                    for (index, value) in slice.iter().enumerate() {
                        if *value > slice[best] {
                            best = index;
                        }
                    }
                    other[row] = best as f32;
                }
                write(arena, out, &other);
            }
            Op::Scale {
                input,
                factor,
                out,
            } => {
                let values = read(spec, arena, input, usize::MAX);
                other.clear();
                other.extend(values.iter().map(|value| value * factor));
                write(arena, out, &other);
            }
            Op::GatedRmsNorm {
                input,
                gate,
                rows,
                width,
                weight,
                eps,
                out,
            } => {
                let gain = weight_values(weights, spec, weight);
                let values: Vec<f32> = read(spec, arena, input, rows * width).to_vec();
                let gates: Vec<f32> = read(spec, arena, gate, rows * width).to_vec();
                scratch.resize(rows * width, 0.0);
                crate::ops::rms_norm_gated(&values, &gates, rows, width, &gain, eps, &mut scratch);
                write(arena, out, &scratch);
            }
            Op::LayerNorm {
                input,
                rows,
                width,
                weight,
                bias,
                eps,
                out,
            } => {
                let gain = weight_values(weights, spec, weight);
                let shift = weight_values(weights, spec, bias);
                let values: Vec<f32> = read(spec, arena, input, rows * width).to_vec();
                scratch.resize(rows * width, 0.0);
                for row in 0..rows {
                    let source = &values[row * width..(row + 1) * width];
                    let mean = source.iter().sum::<f32>() / width as f32;
                    let variance = source
                        .iter()
                        .map(|value| {
                            let centred = value - mean;
                            centred * centred
                        })
                        .sum::<f32>()
                        / width as f32;
                    let inverse = 1.0 / (variance + eps).sqrt();
                    let target = &mut scratch[row * width..(row + 1) * width];
                    for index in 0..width {
                        target[index] =
                            (source[index] - mean) * inverse * (1.0 + gain[index]) + shift[index];
                    }
                }
                write(arena, out, &scratch);
            }
            Op::Softmax {
                input,
                rows,
                width,
                out,
            } => {
                scratch.resize(rows * width, 0.0);
                scratch.copy_from_slice(read(spec, arena, input, rows * width));
                for row in 0..rows {
                    crate::ops::softmax(&mut scratch[row * width..(row + 1) * width]);
                }
                write(arena, out, &scratch);
            }
            Op::DeltaRule {
                query,
                key,
                value,
                beta,
                decay,
                state,
                tokens,
                heads,
                key_width,
                value_width,
                out,
            } => {
                let queries: Vec<f32> =
                    read(spec, arena, query, tokens * heads * key_width).to_vec();
                let keys: Vec<f32> = read(spec, arena, key, tokens * heads * key_width).to_vec();
                let values: Vec<f32> =
                    read(spec, arena, value, tokens * heads * value_width).to_vec();
                let betas: Vec<f32> = read(spec, arena, beta, tokens * heads).to_vec();
                let decays: Vec<f32> = read(spec, arena, decay, tokens * heads).to_vec();
                let state_index = state_index_of(state);
                let mut carried = arena.externals[state_index].clone();
                other.resize(tokens * heads * value_width, 0.0);
                let band = key_width * value_width;
                let mut memory = vec![0.0f32; value_width];
                let mut delta = vec![0.0f32; value_width];
                for head in 0..heads {
                    let band_state = &mut carried[head * band..(head + 1) * band];
                    for token in 0..tokens {
                        let pair = token * heads + head;
                        let query_row = &queries[pair * key_width..(pair + 1) * key_width];
                        let key_row = &keys[pair * key_width..(pair + 1) * key_width];
                        let value_row = &values[pair * value_width..(pair + 1) * value_width];
                        crate::ops::scale_all(band_state, decays[pair]);
                        crate::ops::transposed_matvec(
                            band_state,
                            key_width,
                            value_width,
                            key_row,
                            &mut memory,
                        );
                        for index in 0..value_width {
                            delta[index] = (value_row[index] - memory[index]) * betas[pair];
                        }
                        crate::ops::outer_add(band_state, key_width, value_width, key_row, &delta);
                        crate::ops::transposed_matvec(
                            band_state,
                            key_width,
                            value_width,
                            query_row,
                            &mut memory,
                        );
                        let target = pair * value_width;
                        other[target..target + value_width].copy_from_slice(&memory);
                    }
                }
                arena.externals[state_index] = carried;
                write(arena, out, &other);
            }
            Op::L2Normalize {
                input,
                rows,
                width,
                eps,
                scale,
                out,
            } => {
                let values: Vec<f32> = read(spec, arena, input, rows * width).to_vec();
                scratch.resize(rows * width, 0.0);
                for row in 0..rows {
                    let source = &values[row * width..(row + 1) * width];
                    let mut target = source.to_vec();
                    crate::ops::l2_normalize(&mut target, eps);
                    for value in target.iter_mut() {
                        *value *= scale;
                    }
                    scratch[row * width..(row + 1) * width].copy_from_slice(&target);
                }
                write(arena, out, &scratch);
            }
            Op::Rope {
                input,
                cos,
                sin,
                rows,
                width,
                rotary,
                out,
            } => {
                let values: Vec<f32> = read(spec, arena, input, rows * width).to_vec();
                let cosines: Vec<f32> = read(spec, arena, cos, usize::MAX).to_vec();
                let sines: Vec<f32> = read(spec, arena, sin, usize::MAX).to_vec();
                scratch.resize(rows * width, 0.0);
                scratch.copy_from_slice(&values);
                for row in 0..rows {
                    let offset = (row % cosines.len().max(1)) * rotary / 2;
                    let half = rotary / 2;
                    crate::ops::apply_rope(
                        &mut scratch[row * width..row * width + width],
                        rotary,
                        &cosines[offset..offset + half],
                        &sines[offset..offset + half],
                    );
                }
                write(arena, out, &scratch);
            }
            Op::Sum { input, length, out } => {
                let values = read(spec, arena, input, length);
                other.clear();
                other.push(values.iter().sum());
                write(arena, out, &other);
            }
            Op::Add { a, b, out } => {
                let left = read(spec, arena, a, usize::MAX);
                let right = read(spec, arena, b, usize::MAX);
                other.clear();
                other.extend(left.iter().zip(right).map(|(left, right)| left + right));
                write(arena, out, &other);
            }
            Op::Gather {
                input,
                rows,
                group,
                stride,
                offset,
                width,
                out,
            } => {
                let values: Vec<f32> = read(spec, arena, input, usize::MAX).to_vec();
                other.resize(rows * width, 0.0);
                for row in 0..rows {
                    let start = (row / group) * stride + offset + (row % group) * width;
                    other[row * width..(row + 1) * width]
                        .copy_from_slice(&values[start..start + width]);
                }
                write(arena, out, &other);
            }
            Op::GateMul { a, b, out } => {
                let values = read(spec, arena, a, usize::MAX);
                let gates = read(spec, arena, b, usize::MAX);
                other.clear();
                other.extend(
                    values
                        .iter()
                        .zip(gates)
                        .map(|(value, gate)| value * crate::ops::sigmoid(*gate)),
                );
                write(arena, out, &other);
            }
            Op::Embed {
                table,
                ids,
                rows,
                width,
                out,
            } => {
                let matrix = matrix_of(weights, table);
                let indices: Vec<f32> = read(spec, arena, ids, rows).to_vec();
                other.resize(rows * width, 0.0);
                for (row, id) in indices.iter().enumerate() {
                    crate::ops::row_values(
                        &matrix,
                        *id as usize,
                        &mut other[row * width..(row + 1) * width],
                    );
                }
                write(arena, out, &other);
            }
            Op::Split {
                input,
                rows,
                width,
                first,
                second,
            } => {
                let values: Vec<f32> = read(spec, arena, input, rows * 2 * width).to_vec();
                let mut left = vec![0.0f32; rows * width];
                let mut right = vec![0.0f32; rows * width];
                for row in 0..rows {
                    let source = &values[row * 2 * width..(row + 1) * 2 * width];
                    left[row * width..(row + 1) * width].copy_from_slice(&source[..width]);
                    right[row * width..(row + 1) * width].copy_from_slice(&source[width..]);
                }
                write(arena, first, &left);
                write(arena, second, &right);
            }
            Op::Convolve {
                input,
                weights: weight,
                state,
                taps,
                channels,
                tokens,
                out,
            } => {
                let values: Vec<f32> = read(spec, arena, input, tokens * channels).to_vec();
                let planes = weight_values(weights, spec, weight);
                let state_index = state_index_of(state);
                let mut history = arena.externals[state_index].clone();
                let mut convolved = vec![0.0f32; tokens * channels];
                for token in 0..tokens {
                    let target = &mut convolved[token * channels..(token + 1) * channels];
                    for tap in 0..taps {
                        let back = taps as isize - 1 - tap as isize;
                        for channel in 0..channels {
                            let source = if token as isize >= back {
                                values[(token as isize - back) as usize * channels + channel]
                            } else {
                                let slot = (token as isize + tap as isize + 1) as usize;
                                history[channel * taps + slot]
                            };
                            target[channel] += source * planes[channel * taps + tap];
                        }
                    }
                    for value in target.iter_mut() {
                        *value = crate::ops::silu(*value);
                    }
                }
                // The window carries the last `taps` inputs of every channel.
                for channel in 0..channels {
                    let keep = &mut history[channel * taps..(channel + 1) * taps];
                    if tokens >= taps {
                        for slot in 0..taps {
                            keep[slot] = values[(tokens - taps + slot) * channels + channel];
                        }
                    } else {
                        let held = taps - tokens;
                        for slot in 0..held {
                            keep[slot] = keep[tokens + slot];
                        }
                        for slot in 0..tokens {
                            keep[held + slot] = values[slot * channels + channel];
                        }
                    }
                }
                arena.externals[state_index] = history;
                write(arena, out, &convolved);
            }
            Op::GateDecay {
                b,
                a,
                a_log,
                dt_bias,
                rows,
                beta,
                decay,
            } => {
                let bets: Vec<f32> = read(spec, arena, b, rows).to_vec();
                let advances: Vec<f32> = read(spec, arena, a, rows).to_vec();
                let log = weight_values(weights, spec, a_log);
                let bias = weight_values(weights, spec, dt_bias);
                let heads = log.len();
                let mut betas = vec![0.0f32; rows];
                let mut decays = vec![0.0f32; rows];
                for row in 0..rows {
                    let head = row % heads;
                    betas[row] = crate::ops::sigmoid(bets[row]);
                    decays[row] = (-(log[head].exp())
                        * crate::ops::softplus(advances[row] + bias[head]))
                    .exp();
                }
                write(arena, beta, &betas);
                write(arena, decay, &decays);
            }
            Op::Attention {
                query,
                key,
                value,
                keys,
                values,
                tokens,
                heads,
                kv_heads,
                width,
                past,
                out,
            } => {
                let queries: Vec<f32> = read(spec, arena, query, tokens * heads * width).to_vec();
                let fresh_keys: Vec<f32> = read(spec, arena, key, tokens * kv_heads * width).to_vec();
                let fresh_values: Vec<f32> =
                    read(spec, arena, value, tokens * kv_heads * width).to_vec();
                let keys_index = state_index_of(keys);
                let values_index = state_index_of(values);
                let total = past + tokens;
                {
                    // Both caches at once: `split_at_mut` keeps the borrow checker
                    // honest without a second copy of the buffers.
                    let (low, high) = if keys_index < values_index {
                        let (left, right) = arena.externals.split_at_mut(values_index);
                        (&mut left[keys_index], &mut right[0])
                    } else {
                        let (left, right) = arena.externals.split_at_mut(keys_index);
                        (&mut right[0], &mut left[values_index])
                    };
                    let key_cache = low;
                    let value_cache = high;
                    for token in 0..tokens {
                        for head in 0..kv_heads {
                            let source = (token * kv_heads + head) * width;
                            let target = ((past + token) * kv_heads + head) * width;
                            key_cache[target..target + width]
                                .copy_from_slice(&fresh_keys[source..source + width]);
                            value_cache[target..target + width]
                                .copy_from_slice(&fresh_values[source..source + width]);
                        }
                    }
                }
                let groups = heads / kv_heads;
                let scaling = 1.0 / (width as f32).sqrt();
                let mut attended = vec![0.0f32; tokens * heads * width];
                // Taken out of the arena so the loop can read the keys and values
                // while writing the output.
                let cached_keys = std::mem::take(&mut arena.externals[keys_index]);
                let cached_values = std::mem::take(&mut arena.externals[values_index]);
                for token in 0..tokens {
                    let visible = past + token + 1;
                    for head in 0..heads {
                        let source = (token * heads + head) * width;
                        let kv_head = head / groups;
                        let mut scores = vec![0.0f32; visible];
                        for position in 0..visible {
                            let other = (position * kv_heads + kv_head) * width;
                            scores[position] = crate::ops::dot(
                                &queries[source..source + width],
                                &cached_keys[other..other + width],
                            ) * scaling;
                        }
                        crate::ops::softmax(&mut scores);
                        let target = &mut attended[source..source + width];
                        for position in 0..visible {
                            let weight = scores[position];
                            let other = (position * kv_heads + kv_head) * width;
                            for (slot, value) in
                                target.iter_mut().zip(&cached_values[other..other + width])
                            {
                                *slot += weight * value;
                            }
                        }
                    }
                }
                arena.externals[keys_index] = cached_keys;
                arena.externals[values_index] = cached_values;
                let _ = total;
                write(arena, out, &attended);
            }
        }
    }
}

/// The index of a caller-owned state buffer.
fn state_index_of(slot: Slot) -> usize {
    match slot {
        Slot::External(index) => index,
        _ => panic!("a state buffer has to be external"),
    }
}

fn matrix_of<W: Weights>(weights: &W, weight: Weight) -> Matrix {
    match weight {
        Weight::Tensor(name) => weights.matrix(name),
        Weight::Bound(_) => panic!("a bound value has no matrix shape"),
    }
}

fn weight_values<W: Weights>(weights: &W, spec: &Spec, weight: Weight) -> Vec<f32> {
    match weight {
        Weight::Tensor(name) => {
            let _ = weights;
            crate::weights::expect(name).values()
        }
        Weight::Bound(index) => spec.vectors[index].clone(),
    }
}

/// The values of a slot. `usize::MAX` asks for the whole slot.
fn read<'a>(spec: &'a Spec, arena: &'a Arena, slot: Slot, length: usize) -> &'a [f32] {
    let values = match slot {
        Slot::Buffer(index) => arena.buffer(index),
        Slot::External(index) => arena.externals[index].as_slice(),
        Slot::Bound(index) => spec.vectors[index].as_slice(),
    };
    if length == usize::MAX {
        values
    } else {
        &values[..length]
    }
}

fn write(arena: &mut Arena, slot: Slot, values: &[f32]) {
    let slot = match slot {
        Slot::Buffer(index) => Slot::Buffer(index),
        Slot::External(index) => {
            // A state buffer is written in place; its length is the caller's.
            let target = &mut arena.externals[index];
            target.fill(0.0);
            let count = values.len().min(target.len());
            target[..count].copy_from_slice(&values[..count]);
            return;
        }
        Slot::Bound(index) => panic!("bound slot {index} is not writable"),
    };
    match slot {
        Slot::Buffer(index) => arena.set_buffer(index, values),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three slots in a line: each operator reads the one before it, so the first
    /// is dead by the time the third is written.
    fn chain() -> Spec {
        let step = |input, out| Op::Scale {
            input,
            factor: 1.0,
            out,
        };
        Spec {
            tensors: Vec::new(),
            vectors: vec![vec![1.0; 4]],
            buffers: vec![4, 4, 4],
            externals: Vec::new(),
            ops: vec![
                step(Slot::Bound(0), Slot::Buffer(0)),
                step(Slot::Buffer(0), Slot::Buffer(1)),
                step(Slot::Bound(0), Slot::Buffer(2)),
            ],
        }
    }

    #[test]
    fn a_slot_that_is_dead_shares_its_storage() {
        let spec = chain();
        let laid = layout(&spec, &[false, false, false]);
        assert_eq!(laid.lengths().len(), 2, "three slots, two live at once");
        assert_eq!(laid.storage(0), laid.storage(2));
        assert_ne!(laid.storage(0), laid.storage(1));
    }

    #[test]
    fn a_slot_that_is_read_back_keeps_its_own_storage() {
        let spec = chain();
        let laid = layout(&spec, &[true, false, false]);
        // Slot 0 is wanted, so neither 1 nor 2 may land in its storage; they may
        // share one between themselves.
        assert_ne!(laid.storage(0), laid.storage(1));
        assert_ne!(laid.storage(0), laid.storage(2));
        assert_eq!(laid.lengths().len(), 2);
    }

    #[test]
    fn a_buffer_updated_in_place_keeps_one_storage() {
        // The same slot read and written by one operator: it must stay on the
        // storage it started on, or the update would land somewhere else.
        let mut spec = chain();
        spec.buffers = vec![4, 4];
        spec.ops = vec![
            Op::Scale {
                input: Slot::Bound(0),
                factor: 1.0,
                out: Slot::Buffer(0),
            },
            Op::Scale {
                input: Slot::Buffer(0),
                factor: 2.0,
                out: Slot::Buffer(0),
            },
            Op::Scale {
                input: Slot::Buffer(0),
                factor: 3.0,
                out: Slot::Buffer(1),
            },
        ];
        let laid = layout(&spec, &[false, false]);
        assert_eq!(laid.storage(0), 0, "the updated buffer moved");
        assert_eq!(laid.lengths().len(), 2, "slot 1 is live while 0 is read");
    }
}

//! The GPU backend.
//!
//! The same arithmetic as the CPU backend, expressed as WGSL compute shaders and
//! run on whatever adapter `wgpu` finds. The CPU path is the specification: a
//! kernel here computes the values that path computes, and the gates compare the
//! two.
//!
//! Two kinds of comparison, because there are two kinds of result:
//!
//! - elementwise work has no reduction in it, so it must agree bit for bit;
//! - a reduction leaves the association order (and, for a multiply-add, the
//!   fusion) to the implementation, so it agrees to within a relative tolerance.
//!
//! The backend is behind the `wgpu` feature, which is off by default: the default
//! build has no dependencies at all.
//!
//! # A few submissions per pass
//!
//! A pass is recorded with [`crate::backend::Flow`] and handed over whole.
//! [`Device::execute`] records its dispatches in chunks, submits each chunk as it
//! is recorded, and reads the results back in one mapping. One round trip per
//! operator would cost about half a millisecond each, and a forward pass is a few
//! hundred operators, so per-operator submission spends more time waiting than
//! computing. Chunking rather than one command buffer is a measurement: the device
//! starts on the first chunk while the host is still recording the rest, which is
//! worth about two milliseconds of a decode step, and the queue keeps the order.
//!
//! # State between calls
//!
//! A generation carries key/value caches, convolution windows and recurrent states
//! from one step to the next. Those arrive as [`crate::backend::Slot::External`],
//! and [`Device::execute_stateful`] keeps them in device buffers that survive the
//! call, so a decode loop does not re-upload a cache it just wrote.
//!
//! # The arena is a high-water mark
//!
//! A recording declares a slot for every intermediate the pass makes, and a prefill
//! makes one per operator per token: a five-hundred-token prompt declares nearly two
//! gigabytes. What is live at any instant is a few buffers. [`crate::backend::layout`]
//! hands a slot the storage of one that nothing will touch again, so the device holds
//! the pass' peak instead of its sum — twelve buffers instead of four hundred and
//! forty-four at that prompt length. Slots the caller asked to read back keep storage
//! of their own, because their value has to survive to the end of the pass.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use crate::backend::{Arena, Op, Slot, Spec, Weight};

/// A device and the queue that feeds it.
pub struct Device {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<Pipelines>,
    /// Tensors already on the device, keyed by the container's name for them.
    tensors: Mutex<HashMap<String, Uploaded>>,
    /// Dense weights already on the device, keyed by name. The container keeps the
    /// two small projections of a linear-attention layer in `bf16`.
    dense: Mutex<HashMap<String, wgpu::Buffer>>,
    /// The submission's uniform parameters, one slot per dispatch.
    parameters: Mutex<ParameterSlab>,
    /// Arena buffers, reused across calls. A recorded pass has a few hundred of
    /// them and their contents are always produced before they are read, so a
    /// buffer that is big enough can simply be used again.
    slots: Mutex<Vec<Option<(wgpu::Buffer, usize)>>>,
    /// Bound vectors, reused across calls, with the contents the device last
    /// received.
    ///
    /// A bound vector is read by the kernels and never written by them: a pass that
    /// wrote one would leave the spec's own copy stale, and running the same spec
    /// twice would give two answers. So the copy here is the device's contents, and
    /// a step whose vectors are unchanged costs a comparison instead of a transfer.
    /// A decode loop re-records its pass every token and all but a few of the
    /// vectors — the norm weights and the rope tables — are the same each time.
    vectors_cached: Mutex<Vec<Option<Bound>>>,
    /// The readback staging buffer, and the three distinct buffers a kernel binds
    /// where it reads nothing.
    staging: Mutex<Option<(wgpu::Buffer, usize)>>,
    spares: Mutex<Option<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>>,
    /// The attention weights a pass writes in its first pass and reads in its
    /// second, kept between calls because a decode loop needs one the size of the
    /// cache every step.
    attention: Mutex<Option<(wgpu::Buffer, usize)>>,
}

/// A bound vector's buffer and the values it holds.
struct Bound {
    buffer: wgpu::Buffer,
    capacity: usize,
    values: Vec<f32>,
}

/// One uniform buffer for a whole submission.
///
/// A forward pass is a few hundred dispatches, and creating and writing a buffer
/// for each of them costs more than the dispatch does. The slots are addressed by
/// dynamic offset, at the alignment the device requires.
#[derive(Default)]
struct ParameterSlab {
    buffer: Option<wgpu::Buffer>,
    bytes: Vec<u8>,
    stride: usize,
    next: u32,
}

/// The state a pass carries between calls, in device buffers.
///
/// Indices are the ones the recorder assigned with [`crate::backend::Flow::external`],
/// so one `DeviceState` belongs to one recording of a pass. A buffer grows by being
/// replaced: a cache that gains positions needs more room, and the old contents are
/// copied into the new buffer before the pass writes past them.
#[derive(Default)]
pub struct DeviceState {
    buffers: Vec<Option<(wgpu::Buffer, usize)>>,
    /// Values to write into a buffer the first time it is created. A pass that
    /// starts from a cache the caller already filled seeds them here; a buffer
    /// that already exists keeps what it holds.
    pending: Vec<Option<Vec<f32>>>,
}

impl DeviceState {
    pub fn new() -> DeviceState {
        DeviceState::default()
    }

    /// A state that starts from what the caller carries.
    ///
    /// `values` is laid out as [`crate::model::Recorded::state`] is: one entry per
    /// state buffer, in the order the recorder declared them. A shorter entry is
    /// left as the rest of the buffer is: the pass writes the positions a cache
    /// gains at the end, so the earlier ones are the ones that must arrive.
    pub fn seeded(spec: &Spec, values: &[Vec<f32>]) -> DeviceState {
        let mut states: Vec<Option<Vec<f32>>> = Vec::with_capacity(spec.externals.len());
        for index in 0..spec.externals.len() {
            states.push(values.get(index).cloned());
        }
        DeviceState {
            buffers: vec![None; spec.externals.len()],
            pending: states,
        }
    }
}

/// What the backend is running on.
#[derive(Clone, Debug)]
pub struct Description {
    pub name: String,
    pub backend: String,
    /// The largest storage buffer the device will bind, in bytes.
    pub binding_limit: u64,
    /// The largest buffer the device will allocate, in bytes.
    pub buffer_limit: u64,
}

/// Pipelines are built once: creating one compiles a shader, which costs more
/// than the first call is worth paying again.
struct Pipelines {
    /// The quantised layout.
    quantised: wgpu::BindGroupLayout,
    dequantize: wgpu::ComputePipeline,
    matvec: wgpu::ComputePipeline,
    /// The same arithmetic with one token's accumulators, which is what a decode
    /// step needs.
    matvec_single: wgpu::ComputePipeline,
    embed: wgpu::ComputePipeline,
    /// The float layout.
    floats: wgpu::BindGroupLayout,
    dense_matvec: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    gate_mul: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
    split: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    gated_rms_norm: wgpu::ComputePipeline,
    layer_norm: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    argmax: wgpu::ComputePipeline,
    l2_normalize: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    convolve: wgpu::ComputePipeline,
    convolve_carry: wgpu::ComputePipeline,
    gate_decay: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    attention_scores: wgpu::ComputePipeline,
    attention_softmax: wgpu::ComputePipeline,
    attention_mix: wgpu::ComputePipeline,
    delta_rule: wgpu::ComputePipeline,
}

/// One tensor as the shaders read it: packed integers, and the group scales
/// already widened to `f32`.
///
/// `wgpu::Buffer` is a reference-counted handle, so cloning this keeps the device
/// buffers alive without holding the map's lock.
#[derive(Clone)]
struct Uploaded {
    integers: wgpu::Buffer,
    scales: wgpu::Buffer,
    rows: usize,
    columns: usize,
    group: usize,
}

/// Block on a future.
///
/// The alternative is a dependency whose only job is this function, and the
/// protocol is small enough to write: park the thread, unpark it when the waker
/// fires.
fn block_on<F: Future>(future: F) -> F::Output {
    struct Signal {
        thread: std::thread::Thread,
    }
    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            self.thread.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.thread.unpark();
        }
    }
    let mut future = std::pin::pin!(future);
    let signal = Arc::new(Signal {
        thread: std::thread::current(),
    });
    let waker = Waker::from(signal);
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// A buffer entry for a storage binding.
fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        // One buffer holds every dispatch's parameters, so the pass addresses them
        // by offset rather than allocating one buffer per dispatch.
        has_dynamic_offset: true,
        min_binding_size: wgpu::BufferSize::new(64),
    }
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

impl Device {
    /// Open the default adapter.
    ///
    /// The limits come from the adapter rather than from a default: each tensor
    /// is one buffer and the largest of them is past the downlevel defaults, so
    /// asking for less than the device offers would make the model fail to bind
    /// for no reason.
    pub fn new() -> Result<Device, String> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            // Report the adapter's own limits. Bucketing exists to stop a page from
            // fingerprinting a device; this is a library on a machine its caller
            // owns, and a bucketed limit would understate what is available.
            apply_limit_buckets: false,
        }))
        .map_err(|error| format!("no adapter: {error}"))?;
        let limits = adapter.limits();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("qwen35"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|error| format!("no device: {error}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qwen35 kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });

        // The quantised layout: packed integers, group scales, one input row, one
        // output row, the parameters.
        let quantised = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen35 quantised"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                // The input again, read four values at a time.
                entry(4, storage(true)),
                entry(5, uniform()),
            ],
        });
        // The float layout: four read operands, two read-write operands, a read
        // operand, the output, the parameters.
        let floats = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen35 floats"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, storage(false)),
                entry(5, storage(true)),
                entry(6, storage(true)),
                entry(7, storage(false)),
                entry(8, uniform()),
            ],
        });

        let quantised_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen35 quantised pipeline"),
            bind_group_layouts: &[Some(&quantised)],
            // No immediates: a dispatch's parameters are a uniform buffer, which a
            // bind group addresses by offset.
            immediate_size: 0,
        });
        let float_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen35 float pipeline"),
            bind_group_layouts: &[Some(&floats)],
            immediate_size: 0,
        });
        let quantised_pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&quantised_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let float_pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&float_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipelines = Pipelines {
            quantised,
            dequantize: quantised_pipeline("dequantize_one"),
            matvec: quantised_pipeline("quantised_matvec_one"),
            matvec_single: quantised_pipeline("quantised_matvec_single"),
            embed: quantised_pipeline("embed_rows"),
            floats,
            dense_matvec: float_pipeline("dense_matvec"),
            scale: float_pipeline("scale_all"),
            add: float_pipeline("add_rows"),
            gate_mul: float_pipeline("gate_mul"),
            silu_mul: float_pipeline("silu_mul_rows"),
            split: float_pipeline("split_bands"),
            gather: float_pipeline("gather_bands"),
            rms_norm: float_pipeline("rms_norm_rows"),
            gated_rms_norm: float_pipeline("gated_rms_norm_rows"),
            layer_norm: float_pipeline("layer_norm_rows"),
            softmax: float_pipeline("softmax_rows"),
            argmax: float_pipeline("argmax_rows"),
            l2_normalize: float_pipeline("l2_normalize_rows"),
            rope: float_pipeline("rope_rows"),
            sum: float_pipeline("sum_all"),
            convolve: float_pipeline("convolve_rows"),
            convolve_carry: float_pipeline("convolve_carry"),
            gate_decay: float_pipeline("gate_decay"),
            kv_append: float_pipeline("kv_append"),
            attention_scores: float_pipeline("attention_scores"),
            attention_softmax: float_pipeline("attention_softmax"),
            attention_mix: float_pipeline("attention_mix"),
            delta_rule: float_pipeline("delta_rule"),
        };

        Ok(Device {
            adapter,
            device,
            queue,
            pipelines: Mutex::new(pipelines),
            tensors: Mutex::new(HashMap::new()),
            dense: Mutex::new(HashMap::new()),
            parameters: Mutex::new(ParameterSlab::default()),
            slots: Mutex::new(Vec::new()),
            vectors_cached: Mutex::new(Vec::new()),
            staging: Mutex::new(None),
            spares: Mutex::new(None),
            attention: Mutex::new(None),
        })
    }

    /// What the backend is running on.
    pub fn description(&self) -> Description {
        let info = self.adapter.get_info();
        Description {
            name: info.name,
            backend: format!("{:?}", info.backend),
            binding_limit: u64::from(self.device.limits().max_storage_buffer_binding_size),
            buffer_limit: self.device.limits().max_buffer_size,
        }
    }

    /// Expand a quantised tensor to `f32`, in row-major order.
    ///
    /// This is the device half of [`crate::weights::Tensor::values`].
    pub fn dequantize(&self, name: &str, out: &mut [f32]) {
        let mut flow = crate::backend::Flow::new();
        let target = flow.buffer(out.len());
        let weight = flow.tensor(name);
        flow.dequantize(weight, target);
        let spec = flow.finish();
        let arena = self.execute(&spec);
        out.copy_from_slice(arena.buffer(buffer_index(target)));
    }

    /// `out[row] = weight[row] · input` for a four-bit weight.
    pub fn quantised_matvec(&self, name: &str, input: &[f32], out: &mut [f32]) {
        let mut flow = crate::backend::Flow::new();
        let source = flow.bind(input.to_vec());
        let target = flow.buffer(out.len());
        let weight = flow.tensor(name);
        flow.linear(source, 1, weight, target);
        let spec = flow.finish();
        let arena = self.execute(&spec);
        out.copy_from_slice(arena.buffer(buffer_index(target)));
    }

    /// Run a pass that keeps no state between calls.
    pub fn execute(&self, spec: &Spec) -> Arena {
        assert!(
            spec.externals.is_empty(),
            "this pass carries state; use `execute_stateful` so the state survives the call"
        );
        let mut state = DeviceState::new();
        self.execute_stateful(spec, &mut state)
    }

    /// Run a pass, keeping the state in `state`, and read back every arena buffer.
    ///
    /// Every slot has to keep storage of its own, because the caller may look at
    /// any of them, so this asks the device for the pass' whole declaration and the
    /// staging buffer for the same. A prefill that declares gigabytes will not fit:
    /// a caller that wants a few slots should name them with
    /// [`Device::execute_stateful_reading`].
    pub fn execute_stateful(&self, spec: &Spec, state: &mut DeviceState) -> Arena {
        let all: Vec<usize> = (0..spec.buffers.len()).collect();
        self.execute_stateful_reading(spec, state, &all)
    }

    /// Run a pass, keeping the state in `state`, and read back only the slots in
    /// `outputs`.
    ///
    /// A recorded pass has a few hundred intermediates and a caller usually wants
    /// one of them. Every slot read back costs a copy command and its bytes on the
    /// wire, and the whole model's pass produces about twenty megabytes that nobody
    /// looks at.
    ///
    /// A slot that is not in `outputs` comes back zero: its value is on the device
    /// and is dropped here, so this is for a caller that knows which slots it wants.
    /// An index named twice is copied once.
    pub fn execute_stateful_reading(
        &self,
        spec: &Spec,
        state: &mut DeviceState,
        outputs: &[usize],
    ) -> Arena {
        let device = &self.device;
        // Every operand is resolved before the encoder is opened: a tensor that is
        // not uploaded yet costs an upload, and that must not happen in the middle
        // of recording the pass.
        let mut uploads: HashMap<&str, Uploaded> = HashMap::new();
        for name in &spec.tensors {
            // A dense weight is not packed integers and has its own kernel.
            if crate::weights::expect(name).matrix().kind == "q4" {
                uploads.insert(name.as_str(), self.tensor(name));
            }
        }
        let mut bound: Vec<wgpu::Buffer> = Vec::with_capacity(spec.vectors.len());
        {
            let mut pool = self.vectors_cached.lock().expect("vector pool");
            pool.resize_with(spec.vectors.len(), || None);
            for (index, values) in spec.vectors.iter().enumerate() {
                let needed = (values.len() * 4).max(4);
                let held = match pool[index].take() {
                    Some(held) if held.capacity >= needed => {
                        if held.values == *values {
                            held
                        } else {
                            self.queue.write_buffer(&held.buffer, 0, as_bytes(values));
                            Bound {
                                values: values.clone(),
                                ..held
                            }
                        }
                    }
                    _ => {
                        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("bound vector"),
                            size: needed as u64,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        self.queue.write_buffer(&buffer, 0, as_bytes(values));
                        Bound {
                            buffer,
                            capacity: needed,
                            values: values.clone(),
                        }
                    }
                };
                bound.push(held.buffer.clone());
                pool[index] = Some(held);
            }
        }
        // Which slots this call reads back. They are computed here because the
        // layout has to know: a wanted slot keeps storage of its own, or a later
        // operator's output would be sitting in the buffer it is read from.
        let mut wanted = vec![false; spec.buffers.len()];
        for index in outputs {
            assert!(
                *index < spec.buffers.len(),
                "output slot {index} is past the pass' {} buffers",
                spec.buffers.len()
            );
            wanted[*index] = true;
        }
        // Slots whose lifetimes do not overlap share storage, so the device holds
        // the pass' high-water mark rather than the sum of every intermediate.
        let layout = crate::backend::layout(spec, &wanted);
        let arena: Vec<wgpu::Buffer> = {
            let mut pool = self.slots.lock().expect("slot pool");
            pool.resize_with(layout.lengths().len(), || None);
            layout
                .lengths()
                .iter()
                .enumerate()
                .map(|(index, length)| {
                    lease(
                        device,
                        &mut pool,
                        index,
                        length.max(&1) * 4,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    )
                })
                .collect()
        };
        // A binding has to exist even when a kernel reads none of it, and a buffer
        // used with two access modes in one dispatch is a validation error, so the
        // unused operands are distinct buffers.
        let (spare, state_spare, other_spare) = {
            let mut cached = self.spares.lock().expect("spares");
            match cached.as_ref() {
                Some((spare, state_spare, other_spare)) => {
                    (spare.clone(), state_spare.clone(), other_spare.clone())
                }
                None => {
                    let spare = upload_f32(device, &self.queue, "spare", &[0.0; 8]);
                    let state_spare =
                        upload_f32(device, &self.queue, "state spare", &[0.0; 8]);
                    let other_spare =
                        upload_f32(device, &self.queue, "other spare", &[0.0; 8]);
                    *cached = Some((spare.clone(), state_spare.clone(), other_spare.clone()));
                    (spare, state_spare, other_spare)
                }
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35 pass"),
        });

        // The state a pass declares, in buffers that outlive the call.
        let mut externals: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(spec.externals.len());
        if state.buffers.len() < spec.externals.len() {
            state.buffers.resize_with(spec.externals.len(), || None);
        }
        for (index, length) in spec.externals.iter().enumerate() {
            let needed = (*length).max(1) * 4;
            if state.pending.len() <= index {
                state.pending.resize_with(index + 1, || None);
            }
            let existing = state.buffers[index].take();
            let buffer = match existing {
                Some((buffer, capacity)) if capacity >= needed => buffer,
                Some((old, capacity)) => {
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("state"),
                        size: needed as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    // The old contents are a prefix of the new ones: a cache only
                    // ever grows at its end.
                    encoder.copy_buffer_to_buffer(&old, 0, &buffer, 0, capacity as u64);
                    buffer
                }
                None => {
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("state"),
                        size: needed as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    if let Some(values) = state.pending[index].take() {
                        let bytes: Vec<u8> = values
                            .iter()
                            .take(needed / 4)
                            .flat_map(|value| value.to_le_bytes())
                            .collect();
                        self.queue.write_buffer(&buffer, 0, &bytes);
                    }
                    buffer
                }
            };
            state.buffers[index] = Some((buffer.clone(), needed));
            externals.push(Some(buffer));
        }
        let external = |index: usize| -> &wgpu::Buffer {
            externals[index]
                .as_ref()
                .expect("every declared state buffer was created above")
        };

        let slot = |slot: Slot| -> &wgpu::Buffer {
            match slot {
                Slot::Buffer(index) => &arena[layout.storage(index)],
                Slot::Bound(index) => &bound[index],
                Slot::External(index) => external(index),
            }
        };
        let length = |slot: Slot| -> usize {
            match slot {
                Slot::Buffer(index) => spec.buffers[index],
                Slot::Bound(index) => spec.vectors[index].len(),
                Slot::External(index) => spec.externals[index],
            }
        };

        // One parameter buffer for the whole submission, sized for the dispatches
        // the pass will record. The pass lock is held for the whole encode, so a
        // second pass cannot reset the slab between a dispatch and the write.
        let dispatches: usize = spec
            .ops
            .iter()
            .map(|op| match op {
                Op::Convolve { .. } => 2,
                Op::Attention { .. } => 4,
                _ => 1,
            })
            .sum();
        let pipelines = self.pipelines.lock().expect("pipelines");
        self.prepare_parameters(dispatches);
        // The pass is submitted in chunks rather than as one command buffer. The
        // device starts on the first chunk while the host is still recording the
        // rest, and recording five hundred dispatches costs about as much as
        // running them. The queue preserves submission order, and a chunk boundary
        // is a memory barrier, so a later chunk sees everything an earlier one
        // wrote.
        const CHUNK: usize = 64;
        let mut position = 0usize;
        let mut recorded = 0usize;
        while position < spec.ops.len() {
            let end = (position + CHUNK).min(spec.ops.len());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("qwen35 pass"),
                    timestamp_writes: None,
                });
                for op in &spec.ops[position..end] {
                    self.encode(
                        &mut pass,
                        &pipelines,
                        spec,
                        &uploads,
                        &bound,
                        &spare,
                        &state_spare,
                        &other_spare,
                        *op,
                        &slot,
                        &length,
                    );
                }
            }
            // This chunk's parameters, written before it is executed.
            {
                let slab = self.parameters.lock().expect("parameters");
                let buffer = slab.buffer.as_ref().expect("prepared above");
                let from = recorded * slab.stride;
                recorded = slab.next as usize;
                let to = recorded * slab.stride;
                self.queue.write_buffer(buffer, from as u64, &slab.bytes[from..to]);
            }
            let finished = std::mem::replace(
                &mut encoder,
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("qwen35 pass"),
                }),
            );
            self.queue.submit(Some(finished.finish()));
            position = end;
        }
        drop(pipelines);

        // One staging buffer for the slots this call wants, one copy each, one
        // mapping.
        let arena_sizes: Vec<u64> = spec
            .buffers
            .iter()
            .enumerate()
            .map(|(index, length)| {
                if wanted[index] {
                    (*length * 4).max(4) as u64
                } else {
                    0
                }
            })
            .collect();
        let arena_total: u64 = arena_sizes.iter().sum::<u64>().max(4);
        let needed_bytes = arena_total as usize;
        let staging = {
            let mut cached = self.staging.lock().expect("staging");
            match cached.as_ref() {
                Some((buffer, capacity)) if *capacity >= needed_bytes => buffer.clone(),
                _ => {
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("staging"),
                        size: needed_bytes as u64,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    *cached = Some((buffer.clone(), needed_bytes));
                    buffer
                }
            }
        };
        let mut offset = 0u64;
        for (index, size) in arena_sizes.iter().enumerate() {
            if *size == 0 {
                continue;
            }
            // Each wanted slot has storage of its own, so its slice of the staging
            // buffer can be copied straight out of it.
            let source = &arena[layout.storage(index)];
            encoder.copy_buffer_to_buffer(source, 0, &staging, offset, *size);
            offset += *size;
        }
        // The state is not copied back. It is the largest thing in a pass — the
        // recurrent state of a linear layer is a megabyte and there are eighteen of
        // them — and a decode loop hands it straight back to the next call, so a
        // copy per step is a megabyte-scale transfer for a value nobody reads.
        // `download_state` is for the caller that does want it.
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the device to finish the pass");
        receiver
            .recv()
            .expect("the mapping callback to run")
            .expect("the result to map");

        // The result carries only the slots that were asked for; the rest share
        // storage on the device and have nothing to come back to.
        let mut result = Arena::reading(spec, outputs);
        {
            let view = slice
                .get_mapped_range()
                .expect("the pass' staging buffer to be mapped");
            let mut offset = 0usize;
            for (index, length) in spec.buffers.iter().enumerate() {
                if !wanted[index] {
                    continue;
                }
                let bytes = &view[offset..offset + length * 4];
                // The bytes are `f32` values the device wrote, and a mapping is
                // page-aligned, so this is a reinterpretation and not a conversion:
                // reading them one `from_le_bytes` at a time costs more than the
                // transfer does. A recorded pass has hundreds of these slots.
                let (_, values, _) = unsafe { bytes.align_to::<f32>() };
                assert_eq!(
                    values.len(),
                    *length,
                    "buffer {index} did not map to whole `f32` values"
                );
                result.set_buffer(index, values);
                offset += length * 4;
            }
        }
        staging.unmap();
        result
    }

    /// Read a pass' state back out of the device.
    ///
    /// This is a megabyte-scale transfer, so it is a separate call: a decode loop
    /// keeps the state where it is, and only a caller that wants to look at it — to
    /// hand it to the host executor, or to compare against it — pays for it.
    pub fn download_state(&self, spec: &Spec, state: &mut DeviceState) -> Vec<Vec<f32>> {
        let sizes: Vec<usize> = spec
            .externals
            .iter()
            .map(|length| (*length).max(1) * 4)
            .collect();
        let total: usize = sizes.iter().sum();
        if total == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state staging"),
            size: total as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("state readback"),
            });
        let mut offset = 0u64;
        for (index, size) in sizes.iter().enumerate() {
            let buffer = state
                .buffers
                .get(index)
                .and_then(|entry| entry.as_ref())
                .map(|(buffer, _)| buffer)
                .expect("the state buffer was created by the pass");
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, *size as u64);
            offset += *size as u64;
        }
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the device to finish the readback");
        receiver
            .recv()
            .expect("the mapping callback to run")
            .expect("the state to map");
        let mut values = Vec::with_capacity(sizes.len());
        {
            let view = slice
                .get_mapped_range()
                .expect("the state staging buffer to be mapped");
            let mut offset = 0usize;
            for size in &sizes {
                let bytes = &view[offset..offset + size];
                let (_, state, _) = unsafe { bytes.align_to::<f32>() };
                values.push(state.to_vec());
                offset += size;
            }
        }
        staging.unmap();
        values
    }
}

/// The uniform block, laid out as the shader declares it.
#[repr(C)]
#[derive(Clone, Copy)]
struct Parameters {
    rows: u32,
    columns: u32,
    groups: u32,
    group: u32,
    grid_x: u32,
    eps: f32,
    extra: u32,
    kind: u32,
    stride: u32,
    offset: u32,
    past: u32,
    taps: u32,
    heads: u32,
    kv_heads: u32,
    tokens: u32,
    padding: u32,
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            rows: 1,
            columns: 1,
            groups: 0,
            group: 32,
            grid_x: 0,
            eps: 0.0,
            extra: 0,
            kind: 0,
            stride: 0,
            offset: 0,
            past: 0,
            taps: 0,
            heads: 0,
            kv_heads: 0,
            tokens: 0,
            padding: 0,
        }
    }

    fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        for (index, value) in [
            self.rows,
            self.columns,
            self.groups,
            self.group,
            self.grid_x,
            self.eps.to_bits(),
            self.extra,
            self.kind,
            self.stride,
            self.offset,
            self.past,
            self.taps,
            self.heads,
            self.kv_heads,
            self.tokens,
            self.padding,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// Workgroups to cover `count` invocations at 64 threads each.
fn elementwise(count: usize) -> (u32, u32) {
    let groups = count.div_ceil(64).max(1) as u32;
    let width = groups.min(65535);
    (width, groups.div_ceil(width))
}

/// Workgroups to cover `rows` rows, one workgroup per row.
fn per_row(rows: usize) -> (u32, u32) {
    let width = rows.max(1).min(65535) as u32;
    (width, rows.max(1).div_ceil(width as usize) as u32)
}

/// Workgroups for `count` dispatch groups that each carry a fixed thread count.
fn grouped(count: usize) -> (u32, u32) {
    let groups = count.max(1) as u32;
    let width = groups.min(65535);
    (width, groups.div_ceil(width))
}

/// Warps in one matvec workgroup, which is also the number of weight rows it
/// computes. `kernels.wgsl` declares the same number as `WARPS`; a disagreement
/// mis-addresses the rows and the row-projection gates fail.
const MATVEC_WARPS: usize = 4;

/// Scores one attention workgroup computes at once. `kernels.wgsl` declares the
/// same number as `SCORES_PER_GROUP`.
const SCORES_PER_GROUP: usize = 4;

impl Device {
    /// Record one kernel into the pass' command buffer.
    #[allow(clippy::too_many_arguments)]
    fn encode<'a>(
        &'a self,
        encoder: &mut wgpu::ComputePass<'_>,
        pipelines: &'a Pipelines,
        spec: &Spec,
        uploads: &'a HashMap<&'a str, Uploaded>,
        bound: &'a [wgpu::Buffer],
        spare: &'a wgpu::Buffer,
        state_spare: &'a wgpu::Buffer,
        other_spare: &'a wgpu::Buffer,
        op: Op,
        slot: &dyn Fn(Slot) -> &'a wgpu::Buffer,
        length: &dyn Fn(Slot) -> usize,
    ) {
        let upload_of = |weight: Weight| -> &'a Uploaded {
            let name = match weight {
                Weight::Tensor(name) => name,
                Weight::Bound(_) => panic!("a bound value has no matrix shape"),
            };
            uploads
                .get(name)
                .unwrap_or_else(|| panic!("the pass reads {name:?} as a quantised weight"))
        };
        // A per-element weight is either a vector the recorder bound or a tensor the
        // container keeps dense; both arrive as `f32` storage.
        let bound_of = |weight: Weight| -> wgpu::Buffer {
            match weight {
                Weight::Bound(index) => bound[index].clone(),
                Weight::Tensor(name) => self.dense_tensor(name),
            }
        };

        let mut parameters = Parameters::new();
        match op {
            Op::Dequantize { weight, out } => {
                let uploaded = upload_of(weight);
                let total = length(out);
                parameters.rows = (total / uploaded.columns).max(1) as u32;
                parameters.columns = uploaded.columns as u32;
                parameters.group = uploaded.group as u32;
                self.step(
                    encoder,
                    &pipelines.dequantize,
                    &pipelines.quantised,
                    &[
                        uploaded.integers.as_entire_binding(),
                        uploaded.scales.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                        spare.as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(total),
                );
            }
            Op::Embed {
                table,
                ids,
                rows,
                width,
                out,
            } => {
                let uploaded = upload_of(table);
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.groups = (uploaded.columns / uploaded.group) as u32;
                parameters.group = uploaded.group as u32;
                self.step(
                    encoder,
                    &pipelines.embed,
                    &pipelines.quantised,
                    &[
                        uploaded.integers.as_entire_binding(),
                        uploaded.scales.as_entire_binding(),
                        slot(ids).as_entire_binding(),
                        slot(out).as_entire_binding(),
                        spare.as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(rows * width),
                );
            }
            Op::Linear {
                input,
                tokens,
                weight,
                out,
            } => {
                parameters.tokens = tokens as u32;
                // One workgroup per `(weight row group, chunk of eight tokens)`.
                let chunks = tokens.div_ceil(8);
                parameters.grid_x = 0;
                match weight {
                    Weight::Tensor(name) if crate::weights::expect(name).matrix().kind != "q4" => {
                        let matrix = crate::weights::expect(name).matrix();
                        parameters.columns = matrix.columns as u32;
                        parameters.extra = matrix.rows as u32;
                        // The dense kernel still runs one workgroup per output row.
                        parameters.rows = (tokens * matrix.rows) as u32;
                        parameters.grid_x = parameters.rows.min(65535);
                        let groups = per_row(parameters.rows as usize);
                        let dense = self.dense_tensor(name);
                        self.step(
                            encoder,
                            &pipelines.dense_matvec,
                            &pipelines.floats,
                            &[
                                dense.as_entire_binding(),
                                slot(input).as_entire_binding(),
                                spare.as_entire_binding(),
                                state_spare.as_entire_binding(),
                                other_spare.as_entire_binding(),
                                spare.as_entire_binding(),
                                spare.as_entire_binding(),
                                slot(out).as_entire_binding(),
                            ],
                            &parameters,
                            groups,
                        );
                    }
                    _ => {
                        let uploaded = upload_of(weight);
                        parameters.columns = uploaded.columns as u32;
                        parameters.groups = (uploaded.columns / uploaded.group) as u32;
                        parameters.group = uploaded.group as u32;
                        // One workgroup per four weight rows and chunk; the kernel
                        // splits the dispatch index by the row-group count.
                        let row_groups = uploaded.rows.div_ceil(MATVEC_WARPS);
                        parameters.rows = (chunks * row_groups) as u32;
                        parameters.extra = uploaded.rows as u32;
                        parameters.grid_x = parameters.rows.min(65535);
                        let groups = per_row(parameters.rows as usize);
                        // A single token has one set of accumulators, and a step
                        // held to the chunk's eight carries seven dead ones.
                        let pipeline = if tokens == 1 {
                            &pipelines.matvec_single
                        } else {
                            &pipelines.matvec
                        };
                        self.step(
                            encoder,
                            pipeline,
                            &pipelines.quantised,
                            &[
                                uploaded.integers.as_entire_binding(),
                                uploaded.scales.as_entire_binding(),
                                slot(input).as_entire_binding(),
                                slot(out).as_entire_binding(),
                                vec4_binding(slot(input), length(input)),
                            ],
                            &parameters,
                            groups,
                        );
                    }
                }
            }
            Op::RmsNorm {
                input,
                rows,
                width,
                weight,
                eps,
                out,
            } => {
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.eps = eps;
                self.step(
                    encoder,
                    &pipelines.rms_norm,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        bound_of(weight).as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    per_row(rows),
                );
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
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.eps = eps;
                self.step(
                    encoder,
                    &pipelines.gated_rms_norm,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(gate).as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        bound_of(weight).as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    per_row(rows),
                );
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
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.eps = eps;
                self.step(
                    encoder,
                    &pipelines.layer_norm,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        bound_of(weight).as_entire_binding(),
                        bound_of(bias).as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    per_row(rows),
                );
            }
            Op::SiluMul { gate, up, out } => {
                parameters.rows = length(out) as u32;
                self.step(
                    encoder,
                    &pipelines.silu_mul,
                    &pipelines.floats,
                    &[
                        slot(gate).as_entire_binding(),
                        slot(up).as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(length(out)),
                );
            }
            Op::Add { a, b, out } => {
                parameters.rows = length(out) as u32;
                self.step(
                    encoder,
                    &pipelines.add,
                    &pipelines.floats,
                    &[
                        slot(a).as_entire_binding(),
                        slot(b).as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(length(out)),
                );
            }
            Op::GateMul { a, b, out } => {
                parameters.rows = length(out) as u32;
                self.step(
                    encoder,
                    &pipelines.gate_mul,
                    &pipelines.floats,
                    &[
                        slot(a).as_entire_binding(),
                        slot(b).as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(length(out)),
                );
            }
            Op::Scale {
                input,
                factor,
                out,
            } => {
                let scale = upload_f32(&self.device, &self.queue, "factor", &[factor]);
                parameters.rows = length(out) as u32;
                self.step(
                    encoder,
                    &pipelines.scale,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        scale.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(length(out)),
                );
            }
            Op::Split {
                input,
                rows,
                width,
                first,
                second,
            } => {
                parameters.rows = rows as u32;
                parameters.extra = width as u32;
                self.step(
                    encoder,
                    &pipelines.split,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(first).as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(second).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(rows * width),
                );
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
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.extra = group as u32;
                parameters.stride = stride as u32;
                parameters.offset = offset as u32;
                self.step(
                    encoder,
                    &pipelines.gather,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(rows * width),
                );
            }
            Op::Softmax {
                input,
                rows,
                width,
                out,
            } => {
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                self.step(
                    encoder,
                    &pipelines.softmax,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    per_row(rows),
                );
            }
            Op::Argmax {
                input,
                rows,
                width,
                out,
            } => {
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                self.step(
                    encoder,
                    &pipelines.argmax,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    per_row(rows),
                );
            }
            Op::L2Normalize {
                input,
                rows,
                width,
                eps,
                scale,
                out,
            } => {
                let factor = upload_f32(&self.device, &self.queue, "scale", &[scale]);
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.eps = eps;
                self.step(
                    encoder,
                    &pipelines.l2_normalize,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        factor.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    per_row(rows),
                );
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
                parameters.rows = rows as u32;
                parameters.columns = width as u32;
                parameters.extra = rotary as u32;
                self.step(
                    encoder,
                    &pipelines.rope,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        slot(cos).as_entire_binding(),
                        slot(sin).as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(rows * width),
                );
            }
            Op::Sum { input, length: len, out } => {
                parameters.columns = len as u32;
                self.step(
                    encoder,
                    &pipelines.sum,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        state_spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    (1, 1),
                );
            }
            Op::Convolve {
                input,
                weights,
                state,
                taps,
                channels,
                tokens,
                out,
            } => {
                parameters.rows = tokens as u32;
                parameters.columns = channels as u32;
                parameters.taps = taps as u32;
                self.step(
                    encoder,
                    &pipelines.convolve,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(state).as_entire_binding(),
                        other_spare.as_entire_binding(),
                        bound_of(weights).as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(tokens * channels),
                );
                // The window moves only after every token has read it.
                self.step(
                    encoder,
                    &pipelines.convolve_carry,
                    &pipelines.floats,
                    &[
                        slot(input).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(state).as_entire_binding(),
                        other_spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(channels),
                );
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
                parameters.rows = rows as u32;
                parameters.extra = match a_log {
                    Weight::Bound(index) => spec.vectors[index].len() as u32,
                    Weight::Tensor(name) => crate::weights::expect(name).numel() as u32,
                };
                self.step(
                    encoder,
                    &pipelines.gate_decay,
                    &pipelines.floats,
                    &[
                        slot(b).as_entire_binding(),
                        slot(a).as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(decay).as_entire_binding(),
                        other_spare.as_entire_binding(),
                        bound_of(a_log).as_entire_binding(),
                        bound_of(dt_bias).as_entire_binding(),
                        slot(beta).as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(rows),
                );
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
                // Append the fresh keys and values first: the score pass reads the
                // cache this dispatch writes. Both caches are writable operands, so
                // one dispatch fills both.
                parameters.rows = tokens as u32;
                parameters.columns = width as u32;
                parameters.extra = kv_heads as u32;
                parameters.past = past as u32;
                self.step(
                    encoder,
                    &pipelines.kv_append,
                    &pipelines.floats,
                    &[
                        slot(key).as_entire_binding(),
                        slot(value).as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(keys).as_entire_binding(),
                        slot(values).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(tokens * kv_heads * width),
                );
                parameters.heads = heads as u32;
                parameters.kv_heads = kv_heads as u32;
                // The weights of the longest row, so both passes can find a row
                // without being told each one's length.
                parameters.stride = (past + tokens) as u32;
                let weights = self.attention_weights((tokens * heads * (past + tokens)) as usize);
                // Four scores share a workgroup, so the score count is grouped into
                // chunks of four and the kernel reaches its row through `groups`.
                let chunks = (past + tokens).div_ceil(SCORES_PER_GROUP);
                parameters.groups = chunks as u32;
                parameters.rows = (tokens * heads * chunks) as u32;
                parameters.grid_x = parameters.rows.min(65535);
                self.step(
                    encoder,
                    &pipelines.attention_scores,
                    &pipelines.floats,
                    &[
                        slot(query).as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(keys).as_entire_binding(),
                        weights.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                    ],
                    &parameters,
                    // One workgroup per four scores.
                    per_row(tokens * heads * chunks),
                );
                // One thread per row.
                parameters.rows = (tokens * heads) as u32;
                self.step(
                    encoder,
                    &pipelines.attention_softmax,
                    &pipelines.floats,
                    &[
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        weights.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                    ],
                    &parameters,
                    elementwise(tokens * heads),
                );
                // The mix pass splits each row's positions across a workgroup: one
                // workgroup per (row, block of thirty-two columns).
                let blocks = width.div_ceil(32);
                parameters.rows = (tokens * heads) as u32;
                parameters.groups = blocks as u32;
                let dispatch = grouped(tokens * heads * blocks);
                parameters.grid_x = dispatch.0;
                self.step(
                    encoder,
                    &pipelines.attention_mix,
                    &pipelines.floats,
                    &[
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        spare.as_entire_binding(),
                        other_spare.as_entire_binding(),
                        slot(values).as_entire_binding(),
                        weights.as_entire_binding(),
                        spare.as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    dispatch,
                );
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
                parameters.rows = heads as u32;
                parameters.columns = tokens as u32;
                parameters.groups = key_width as u32;
                parameters.extra = value_width as u32;
                // One workgroup per (head, block of value columns); its 256 threads
                // split the key axis.
                let groups = heads * value_width.div_ceil(32);
                let dispatch = grouped(groups);
                parameters.grid_x = dispatch.0;
                self.step(
                    encoder,
                    &pipelines.delta_rule,
                    &pipelines.floats,
                    &[
                        slot(query).as_entire_binding(),
                        slot(key).as_entire_binding(),
                        slot(value).as_entire_binding(),
                        slot(state).as_entire_binding(),
                        other_spare.as_entire_binding(),
                        slot(beta).as_entire_binding(),
                        slot(decay).as_entire_binding(),
                        slot(out).as_entire_binding(),
                    ],
                    &parameters,
                    dispatch,
                );
            }
        }
    }

    /// Record one dispatch with `layout`.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        encoder: &mut wgpu::ComputePass<'_>,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        buffers: &[wgpu::BindingResource<'_>],
        parameters: &Parameters,
        dispatch: (u32, u32),
    ) {
        let device = &self.device;
        // This dispatch's parameters, in the slot the pass reserved for it.
        let (parameter_buffer, offset) = {
            let mut slab = self.parameters.lock().expect("parameters");
            let slot = slab.next as usize;
            slab.next += 1;
            let start = slot * slab.stride;
            assert!(
                start + 64 <= slab.bytes.len(),
                "the pass declared fewer dispatches than it recorded"
            );
            slab.bytes[start..start + 64].copy_from_slice(&parameters.to_bytes());
            (
                slab.buffer.clone().expect("the slab is prepared before encoding"),
                start as u32,
            )
        };
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, resource)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource,
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &parameter_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(64),
            }),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step"),
            layout,
            entries: &entries,
        });
        encoder.set_pipeline(pipeline);
        encoder.set_bind_group(0, &bind_group, &[offset]);
        encoder.dispatch_workgroups(dispatch.0.max(1), dispatch.1.max(1), 1);
    }

    /// A scratch buffer for the attention weights, at least `floats` long.
    fn attention_weights(&self, floats: usize) -> wgpu::Buffer {
        let needed = floats.max(4) * 4;
        let mut cached = self.attention.lock().expect("attention weights");
        match cached.as_ref() {
            Some((buffer, capacity)) if *capacity >= needed => buffer.clone(),
            _ => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("attention weights"),
                    size: needed as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                *cached = Some((buffer.clone(), needed));
                buffer
            }
        }
    }

    /// Size the parameter slab for `dispatches` and reset it.
    fn prepare_parameters(&self, dispatches: usize) {
        let stride = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(64) as usize;
        let needed = stride * dispatches.max(1);
        let mut slab = self.parameters.lock().expect("parameters");
        slab.stride = stride;
        slab.next = 0;
        let large_enough = slab
            .buffer
            .as_ref()
            .is_some_and(|buffer| buffer.size() >= needed as u64);
        if !large_enough {
            slab.buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("parameters"),
                size: needed as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        slab.bytes.clear();
        slab.bytes.resize(needed, 0);
    }
}

/// A buffer bound as `array<vec4<f32>>`: the size has to be a whole number of
/// vectors, and a pooled buffer can be a little larger than the value it holds.
fn vec4_binding(buffer: &wgpu::Buffer, floats: usize) -> wgpu::BindingResource<'_> {
    let wanted = (floats * 4).div_ceil(16) * 16;
    let bytes = (wanted as u64).min(buffer.size()).max(16);
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(bytes.max(16)),
    })
}

/// The buffer index a slot names.
fn buffer_index(slot: Slot) -> usize {
    match slot {
        Slot::Buffer(index) => index,
        Slot::Bound(_) => panic!("a bound slot has no arena buffer"),
        Slot::External(index) => index,
    }
}

/// A `f32` slice as bytes, for the queue's copy.
///
/// Building a `Vec<u8>` instead would allocate and convert two megabytes on every
/// decode step, which costs more than the transfer does.
fn as_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and every bit pattern is a value; a byte slice
    // carries no alignment requirement.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

/// A buffer of at least `needed` bytes, from the pool when one is big enough.
///
/// A pass allocates a few hundred of these. The contents of an arena slot are
/// always written before they are read, so a pooled buffer needs no clearing; a
/// bound vector is rewritten by its caller.
fn lease(
    device: &wgpu::Device,
    pool: &mut [Option<(wgpu::Buffer, usize)>],
    index: usize,
    needed: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    if let Some((buffer, capacity)) = &pool[index] {
        if *capacity >= needed {
            return buffer.clone();
        }
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("slot"),
        size: needed as u64,
        usage,
        mapped_at_creation: false,
    });
    pool[index] = Some((buffer.clone(), needed));
    buffer
}

fn upload_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let mut padded = bytes.to_vec();
    while padded.len() % 4 != 0 {
        padded.push(0);
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: padded.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &padded);
    buffer
}

fn upload_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    values: &[f32],
) -> wgpu::Buffer {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    upload_bytes(device, queue, label, &bytes, wgpu::BufferUsages::STORAGE)
}

impl Device {
    /// A dense weight as `f32`, uploaded the first time it is asked for.
    fn dense_tensor(&self, name: &str) -> wgpu::Buffer {
        let mut dense = self.dense.lock().expect("dense tensors");
        if let Some(buffer) = dense.get(name) {
            return buffer.clone();
        }
        let values = crate::weights::expect(name).values();
        let buffer = upload_f32(&self.device, &self.queue, name, &values);
        dense.insert(name.to_string(), buffer.clone());
        buffer
    }

    /// The tensor on the device, uploading it the first time it is asked for.
    fn tensor(&self, name: &str) -> Uploaded {
        let mut tensors = self.tensors.lock().expect("tensors");
        if let Some(uploaded) = tensors.get(name) {
            return uploaded.clone();
        }
        let tensor = crate::weights::expect(name);
        let matrix = tensor.matrix();
        let (scale_bytes, integer_bytes) = match tensor.data {
            crate::weights::Data::Quantized {
                scales, integers, ..
            } => (scales, integers),
            _ => panic!("{name}: only four-bit tensors have a device path"),
        };
        // The scales stay `f16`, as the container keeps them. Widening them to
        // `f32` here costs four bytes per group on every read of every weight, and
        // `unpack2x16float` turns one word into the two scales it holds.
        let scales = upload_bytes(
            &self.device,
            &self.queue,
            "scales",
            scale_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let integers = upload_bytes(
            &self.device,
            &self.queue,
            name,
            integer_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let columns = matrix.columns;
        let group = matrix.group;
        let uploaded = Uploaded {
            integers,
            scales,
            rows: matrix.rows,
            columns,
            group,
        };
        tensors.insert(name.to_string(), uploaded.clone());
        uploaded
    }
}

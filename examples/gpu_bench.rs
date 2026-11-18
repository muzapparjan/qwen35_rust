//! The device backend against the CPU one.
//!
//! ```text
//! cargo run --release --features wgpu --example gpu_bench
//! ```
//!
//! Four sections:
//!
//! - one projection per submission, which is what a backend that submits per
//!   operator costs, and where it stops being worth it;
//! - one layer's MLP as a single pass;
//! - the whole model as a single pass, with the state seeded once;
//! - the three kernels that walk something serially inside a workgroup, each
//!   repeated for every layer of the model, so the hot one is visible.
//!
//! A projection is measured with the input already in place on both sides, so what
//! is compared is the kernel and not the setup. On the device that still includes
//! the upload of the results, because a caller has to read them.

use qwen35::backend;
use qwen35::gpu::Device;
use qwen35::ops;
use qwen35::weights;

struct Container;

impl backend::Weights for Container {
    fn matrix(&self, name: &str) -> weights::Matrix {
        weights::expect(name).matrix()
    }
}

/// One layer's MLP as a recorded pass, the unit a real forward would be made of.
fn mlp_pass(layer: &str) -> (backend::Spec, backend::Slot) {
    const HIDDEN: usize = 1024;
    const INTERMEDIATE: usize = 3584;
    let mut flow = backend::Flow::new();
    let input: Vec<f32> = (0..HIDDEN).map(|index| (index as f32 * 0.013).sin()).collect();
    let source = flow.bind(input);
    let normalized = flow.buffer(HIDDEN);
    let gate = flow.buffer(INTERMEDIATE);
    let up = flow.buffer(INTERMEDIATE);
    let activated = flow.buffer(INTERMEDIATE);
    let out = flow.buffer(HIDDEN);
    let norm = flow.bound(weights::expect(&format!("{layer}.post_attention_layernorm.weight")).values());
    flow.rms_norm(source, 1, HIDDEN, norm, 1e-6, normalized);
    let gate_weight = flow.tensor(&format!("{layer}.mlp.gate_proj.weight"));
    flow.linear(normalized, 1, gate_weight, gate);
    let up_weight = flow.tensor(&format!("{layer}.mlp.up_proj.weight"));
    flow.linear(normalized, 1, up_weight, up);
    flow.silu_mul(gate, up, activated);
    let down_weight = flow.tensor(&format!("{layer}.mlp.down_proj.weight"));
    flow.linear(activated, 1, down_weight, out);
    (flow.finish(), out)
}

fn index(slot: backend::Slot) -> usize {
    match slot {
        backend::Slot::Buffer(index) => index,
        backend::Slot::Bound(_) | backend::Slot::External(_) => {
            panic!("the slot is not an arena buffer")
        }
    }
}

fn main() {
    let device = match Device::new() {
        Ok(device) => device,
        Err(error) => {
            println!("no device: {error}");
            return;
        }
    };
    let description = device.description();
    println!("adapter  : {} ({})", description.name, description.backend);
    println!(
        "limits   : bind {:.0} MB, buffer {:.0} MB",
        description.binding_limit as f64 / 1e6,
        description.buffer_limit as f64 / 1e6
    );
    println!();

    // --- one operator at a time, which is what a per-operator backend costs
    let names = [
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.3.mlp.gate_proj.weight",
        "model.language_model.embed_tokens.weight",
    ];
    println!("{:<48} {:>9} {:>9} {:>8}", "one operator per submission", "cpu ms", "gpu ms", "ratio");
    for name in names {
        let tensor = weights::expect(name);
        let matrix = tensor.matrix();
        let input: Vec<f32> = (0..matrix.columns)
            .map(|index| (index as f32 * 0.017).sin())
            .collect();
        let mut cpu = vec![0.0f32; matrix.rows];
        let mut gpu = vec![0.0f32; matrix.rows];
        device.quantised_matvec(name, &input, &mut gpu);
        let cpu_ms = time(3, || ops::linear(&input, 1, &matrix, None, &mut cpu));
        let gpu_ms = time(3, || device.quantised_matvec(name, &input, &mut gpu));
        let short = name.split("layers.").last().unwrap_or(name);
        let short = short.replace("model.language_model.", "…");
        let short = short.replace(".weight", "");
        println!(
            "{:<48} {:>9.2} {:>9.2} {:>7.2}x",
            short,
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms
        );
    }

    // --- the whole pass, recorded as one pass
    println!();
    let (spec, out) = mlp_pass("model.language_model.layers.3");
    let out_index = index(out);
    let mut arena = backend::Arena::new(&spec);
    // Warm both paths: the first call faults in the weights and uploads them.
    backend::execute(&Container, &spec, &mut arena);
    device.execute(&spec);
    let portable_ms = time(5, || backend::execute(&Container, &spec, &mut arena));
    let device_ms = time(5, || {
        device.execute(&spec);
    });
    let ported = arena.buffer(out_index).to_vec();
    let on_device = device.execute(&spec);
    let device_values = on_device.buffer(out_index);
    let mut worst = 0.0f32;
    for (left, right) in ported.iter().zip(device_values) {
        worst = worst.max((left - right).abs() / left.abs().max(1.0));
    }
    println!("{:<48} {:>9} {:>9} {:>8}", "one layer's MLP as one pass", "cpu ms", "gpu ms", "ratio");
    println!(
        "{:<48} {:>9.2} {:>9.2} {:>7.2}x   worst {:.1e}",
        format!("{} operators, {} tensors", spec.ops.len(), spec.tensors.len()),
        portable_ms,
        device_ms,
        portable_ms / device_ms,
        worst
    );

    // --- the whole model, recorded as one pass
    //
    // This is the number the backend exists for: a forward pass is a few hundred
    // operators, and the cost of a submission is paid once rather than per
    // operator.
    println!();
    let model = qwen35::model::Model::new();
    let ids: Vec<u32> = qwen35::tokenizer::tokenizer()
        .encode("The capital of France is")
        .into_iter()
        .take(9)
        .collect();
    let positions: Vec<[u32; 3]> = (0..ids.len()).map(|token| [token as u32; 3]).collect();
    let recorded =
        model.record(&ids, ids.len(), &positions, &qwen35::model::Cache::new());
    let logits_index = index(recorded.logits);
    let mut arena = backend::Arena::with_state(&recorded.spec, recorded.state.clone());
    let mut state = qwen35::gpu::DeviceState::seeded(&recorded.spec, &recorded.state);
    backend::execute(&Container, &recorded.spec, &mut arena);
    device.execute_stateful_reading(&recorded.spec, &mut state, &[logits_index]);
    let host_ms = time(3, || backend::execute(&Container, &recorded.spec, &mut arena));
    let whole_ms = time(3, || {
        device.execute_stateful_reading(&recorded.spec, &mut state, &[logits_index]);
    });
    // The comparison starts from the recording's state on both sides: a recorded
    // pass writes the recurrent state in place, so a second application is not the
    // same pass as the first one.
    let mut fresh = backend::Arena::with_state(&recorded.spec, recorded.state.clone());
    backend::execute(&Container, &recorded.spec, &mut fresh);
    let mut fresh_state = qwen35::gpu::DeviceState::seeded(&recorded.spec, &recorded.state);
    let on_device =
        device.execute_stateful_reading(&recorded.spec, &mut fresh_state, &[logits_index]);
    let host_values = fresh.buffer(logits_index).to_vec();
    let device_values = on_device.buffer(logits_index);
    let mut worst = 0.0f32;
    for (left, right) in host_values.iter().zip(device_values) {
        worst = worst.max((left - right).abs() / left.abs().max(1.0));
    }
    println!(
        "{:<48} {:>9} {:>9} {:>8}",
        "the whole model as one pass", "cpu ms", "gpu ms", "ratio"
    );
    println!(
        "{:<48} {:>9.2} {:>9.2} {:>7.2}x   worst {:.1e}",
        format!(
            "{} operators, {} tensors, {} tokens",
            recorded.spec.ops.len(),
            recorded.spec.tensors.len(),
            ids.len()
        ),
        host_ms,
        whole_ms,
        host_ms / whole_ms,
        worst
    );

    // --- where the pass' time goes
    //
    // The kernels with a serial walk inside them — the recurrent state, the
    // convolution window, attention — are the ones a device can be slow at, because
    // the width they parallelise over is the head dimension rather than the token
    // count. Each row is that kernel repeated for as many layers as the model has,
    // with the shapes a five-token prompt gives.
    println!();
    println!(
        "{:<48} {:>9} {:>9} {:>8}",
        "one kernel over the whole model", "cpu ms", "gpu ms", "ratio"
    );
    let sections: [(&str, backend::Spec); 7] = [
        ("delta_rule x 18", delta_rule_pass(18)),
        ("convolve x 18", convolve_pass(18)),
        ("attention x 6", attention_pass(6)),
        ("attention x 6, 512 past", attention_long_pass(6, 512)),
        // The same number of dispatches with nothing in them: what a kernel row
        // costs when the arithmetic is free is the harness, not the kernel.
        ("24 empty dispatches", scale_pass(24)),
        ("128 empty dispatches", scale_pass(128)),
        ("512 empty dispatches", scale_pass(512)),
    ];
    for (label, spec) in sections {
        let mut arena = backend::Arena::new(&spec);
        let mut state = qwen35::gpu::DeviceState::new();
        backend::execute(&Container, &spec, &mut arena);
        // No slot is read: this row is the kernels and the dispatches, which is what
        // the model's pass pays for each of them.
        device.execute_stateful_reading(&spec, &mut state, &[]);
        let cpu_ms = time(3, || backend::execute(&Container, &spec, &mut arena));
        let gpu_ms = time(3, || {
            device.execute_stateful_reading(&spec, &mut state, &[]);
        });
        println!("{label:<48} {cpu_ms:>9.2} {gpu_ms:>9.2} {:>7.2}x", cpu_ms / gpu_ms);
    }

    // --- the weight stream on its own
    //
    // A pass reads every weight once per chunk of four tokens, so the prefill is
    // limited by how fast the device can stream them. This section measures exactly
    // that: the same projection repeated, with the bytes read known from the shape,
    // so the figure is a bandwidth and not a time to be compared against a machine
    // that happens to be busy.
    println!();
    println!(
        "{:<48} {:>9} {:>9} {:>10}",
        "weight stream", "cpu ms", "gpu ms", "gpu GB/s"
    );
    for (name, tokens, copies) in [
        ("model.language_model.layers.3.mlp.down_proj.weight", 5usize, 10usize),
        ("model.language_model.layers.3.mlp.down_proj.weight", 5, 50),
        ("model.language_model.layers.3.mlp.down_proj.weight", 5, 200),
        ("model.language_model.layers.3.mlp.down_proj.weight", 1, 50),
        ("model.language_model.embed_tokens.weight", 5, 20),
    ] {
        let matrix = weights::expect(name).matrix();
        let bytes = (matrix.rows * matrix.columns / 2) as f64;
        let (spec, out) = linear_pass(copies, name, tokens, matrix.columns, matrix.rows);
        let out_index = index(out);
        let mut arena = backend::Arena::new(&spec);
        backend::execute(&Container, &spec, &mut arena);
        device.execute(&spec);
        let cpu_ms = time(3, || backend::execute(&Container, &spec, &mut arena));
        let gpu_ms = time(3, || {
            device.execute(&spec);
        });
        let read = bytes * copies as f64 / 1e9;
        println!(
            "{:<48} {cpu_ms:>9.2} {gpu_ms:>9.2} {:>10.0}",
            format!("{} x{copies} x{tokens} tokens", name.split('.').nth(4).unwrap_or(name)),
            read / (gpu_ms / 1000.0)
        );
        let _ = out_index;
    }

    // --- what a submission costs before any arithmetic
    //
    // The same slot list the whole model's pass declares, with no operators in it:
    // buffer leases, the staging copies and the readback of every slot. A recorded
    // pass produces a few hundred intermediates, and this says what copying them all
    // out costs.
    println!();
    {
        let mut total = 0usize;
        let empty = backend::Spec {
            tensors: Vec::new(),
            vectors: Vec::new(),
            buffers: recorded.spec.buffers.clone(),
            externals: recorded.spec.externals.clone(),
            ops: Vec::new(),
        };
        for length in &empty.buffers {
            total += length;
        }
        let mut empty_state = qwen35::gpu::DeviceState::new();
        device.execute_stateful(&empty, &mut empty_state);
        let empty_ms = time(5, || {
            device.execute_stateful(&empty, &mut empty_state);
        });
        // The same pass with one slot read back, which is what a caller of the whole
        // model asks for.
        device.execute_stateful_reading(&empty, &mut empty_state, &[0]);
        let one_ms = time(5, || {
            device.execute_stateful_reading(&empty, &mut empty_state, &[0]);
        });
        // And with none, which is the floor under a pass: the leases and the
        // submission.
        let none_ms = time(5, || {
            device.execute_stateful_reading(&empty, &mut empty_state, &[]);
        });
        println!(
            "{:<48} {:>9.2} {:>7.1}",
            format!(
                "empty pass, {} slots: all {:.0} MB read back",
                empty.buffers.len(),
                total as f64 * 4.0 / 1e6
            ),
            empty_ms,
            ""
        );
        println!(
            "{:<48} {:>9.2} {:>7.1}",
            "empty pass, one slot read back",
            one_ms,
            ""
        );
        println!(
            "{:<48} {:>9.2} {:>7.1}",
            "empty pass, no slot read back",
            none_ms,
            ""
        );
        // The same bytes in four slots instead of hundreds: one copy command each
        // instead of one per intermediate.
        let same_bytes = total.div_ceil(4);
        let few = backend::Spec {
            tensors: Vec::new(),
            vectors: Vec::new(),
            buffers: vec![same_bytes; 4],
            externals: Vec::new(),
            ops: Vec::new(),
        };
        let mut few_state = qwen35::gpu::DeviceState::new();
        device.execute_stateful(&few, &mut few_state);
        let few_ms = time(5, || {
            device.execute_stateful(&few, &mut few_state);
        });
        println!(
            "{:<48} {:>9.2} {:>7.1}",
            format!("empty pass: 4 slots, {:.0} MB", same_bytes as f64 * 4.0 * 4.0 / 1e6),
            few_ms,
            ""
        );
    }

    // --- one decode step
    //
    // Generation reads every weight once per token whatever the prompt was, so the
    // step that matters is a single token with a cache behind it. The state is
    // carried in device buffers, so what is timed is the pass and not an upload.
    println!();
    {
        let model = qwen35::model::Model::new();
        let prompt: Vec<u32> = qwen35::tokenizer::tokenizer()
            .encode("The capital of France is")
            .into_iter()
            .take(5)
            .collect();
        let past = 64usize;
        let state = vec![Vec::new(); 48];
        let positions: Vec<[u32; 3]> = vec![[past as u32; 3]];
        let step = model.record_with_state(&[prompt[0]], 1, &positions, past, &state);
        let logits_index = index(step.logits);
        let mut arena = backend::Arena::new(&step.spec);
        let mut device_state = qwen35::gpu::DeviceState::new();
        backend::execute(&Container, &step.spec, &mut arena);
        device.execute_stateful_reading(&step.spec, &mut device_state, &[logits_index]);
        let cpu_ms = time(3, || backend::execute(&Container, &step.spec, &mut arena));
        let gpu_ms = time(3, || {
            device.execute_stateful_reading(&step.spec, &mut device_state, &[logits_index]);
        });
        let _ = logits_index;
        println!(
            "{:<48} {:>9.2} {:>9.2} {:>7.2}x",
            "one decode token",
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms
        );
    }
}

/// The same projection repeated, each into its own output buffer.
fn linear_pass(
    copies: usize,
    name: &str,
    tokens: usize,
    columns: usize,
    rows: usize,
) -> (backend::Spec, backend::Slot) {
    let mut flow = backend::Flow::new();
    let input: Vec<f32> = (0..tokens * columns)
        .map(|index| (index as f32 * 0.013).sin())
        .collect();
    let source = flow.bind(input);
    let weight = flow.tensor(name);
    let mut out = flow.buffer(tokens * rows);
    for _ in 0..copies {
        out = flow.buffer(tokens * rows);
        flow.linear(source, tokens, weight, out);
    }
    (flow.finish(), out)
}

/// The recurrent state of a linear-attention layer, repeated per layer.
fn delta_rule_pass(layers: usize) -> backend::Spec {
    const HEADS: usize = 16;
    const TOKENS: usize = 5;
    const KEY: usize = 128;
    const VALUE: usize = 128;
    let pair = TOKENS * HEADS;
    let mut flow = backend::Flow::new();
    let query = flow.bind(vec![0.11; pair * KEY]);
    let key = flow.bind(vec![0.02; pair * KEY]);
    let value = flow.bind(vec![0.05; pair * VALUE]);
    let beta = flow.bind(vec![0.5; pair]);
    let decay = flow.bind(vec![0.93; pair]);
    for layer in 0..layers {
        let state = flow.external(layer, HEADS * KEY * VALUE);
        let out = flow.buffer(pair * VALUE);
        flow.delta_rule(
            query, key, value, beta, decay, state, TOKENS, HEADS, KEY, VALUE, out,
        );
    }
    flow.finish()
}

/// The depthwise convolution and its carried window.
fn convolve_pass(layers: usize) -> backend::Spec {
    const CHANNELS: usize = 6144;
    const TAPS: usize = 4;
    const TOKENS: usize = 5;
    let mut flow = backend::Flow::new();
    let input = flow.bind(vec![0.03; TOKENS * CHANNELS]);
    let weights = flow.bound(vec![0.25; CHANNELS * TAPS]);
    for layer in 0..layers {
        let state = flow.external(layer, CHANNELS * TAPS);
        let out = flow.buffer(TOKENS * CHANNELS);
        flow.convolve(input, weights, state, TAPS, CHANNELS, TOKENS, out);
    }
    flow.finish()
}

/// `count` copies of the smallest kernel there is: one buffer multiplied by a
/// scalar, 64 values wide.
fn scale_pass(count: usize) -> backend::Spec {
    let mut flow = backend::Flow::new();
    let input = flow.bind(vec![1.5f32; 64]);
    for _ in 0..count {
        let out = flow.buffer(64);
        flow.scale(input, 2.0, out);
    }
    flow.finish()
}

/// Attention over a cache holding the prompt.
fn attention_pass(layers: usize) -> backend::Spec {
    attention_shaped(layers, 0)
}

/// Attention with a cache behind the prompt, which is the regime it exists for: the
/// score rows are longer than the prefill and the work per dispatch grows with them.
fn attention_long_pass(layers: usize, past: usize) -> backend::Spec {
    attention_shaped(layers, past)
}

fn attention_shaped(layers: usize, past: usize) -> backend::Spec {
    const TOKENS: usize = 5;
    const HEADS: usize = 16;
    const KV_HEADS: usize = 2;
    const WIDTH: usize = 256;
    let positions = past + TOKENS;
    let mut flow = backend::Flow::new();
    let query = flow.bind(vec![0.07; TOKENS * HEADS * WIDTH]);
    let key = flow.bind(vec![0.03; TOKENS * KV_HEADS * WIDTH]);
    let value = flow.bind(vec![0.05; TOKENS * KV_HEADS * WIDTH]);
    for layer in 0..layers {
        let keys = flow.external(2 * layer, positions * KV_HEADS * WIDTH);
        let values = flow.external(2 * layer + 1, positions * KV_HEADS * WIDTH);
        let out = flow.buffer(TOKENS * HEADS * WIDTH);
        flow.attention(
            query, key, value, keys, values, TOKENS, HEADS, KV_HEADS, WIDTH, past, out,
        );
    }
    flow.finish()
}

fn time(repeats: usize, mut body: impl FnMut()) -> f64 {
    let started = std::time::Instant::now();
    for _ in 0..repeats {
        body();
    }
    started.elapsed().as_secs_f64() * 1000.0 / repeats as f64
}

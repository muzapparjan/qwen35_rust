//! The GPU backend against the CPU one.
//!
//! The gates are split by what kind of result is being compared, because the two
//! kinds have different obligations:
//!
//! - expansion is elementwise, so anything but bit equality is a defect;
//! - a dot product is a reduction, and WGSL leaves the association order and the
//!   fusion of a multiply-add to the implementation, so a relative tolerance is
//!   the honest gate.
//!
//! A machine with no usable adapter skips these rather than failing: the backend
//! being unavailable is not the arithmetic being wrong.

#![cfg(feature = "wgpu")]

use qwen35::backend;
use qwen35::gpu::Device;
use qwen35::ops;
use qwen35::weights;

fn device() -> Option<Device> {
    match Device::new() {
        Ok(device) => {
            let description = device.description();
            eprintln!(
                "adapter: {} ({}) bind {:.0} MB buffer {:.0} MB",
                description.name,
                description.backend,
                description.binding_limit as f64 / 1e6,
                description.buffer_limit as f64 / 1e6
            );
            Some(device)
        }
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

/// Expansion has no reduction in it, so the two paths must agree exactly.
#[test]
fn the_device_expands_a_tensor_to_the_same_bits() {
    let Some(device) = device() else {
        return;
    };
    // One of the small projections: enough rows to be a real shape, small enough
    // that comparing every value is cheap.
    let name = "model.language_model.layers.3.self_attn.k_proj.weight";
    let expected = weights::expect(name).values();
    let mut actual = vec![0.0f32; expected.len()];
    device.dequantize(name, &mut actual);
    let differences = expected
        .iter()
        .zip(&actual)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    assert_eq!(
        differences,
        0,
        "{name}: {differences} of {} values differ",
        expected.len()
    );
}

/// A reduction differs in association order, so the gate is relative.
#[test]
fn the_device_multiplies_a_row_the_way_the_cpu_does() {
    let Some(device) = device() else {
        return;
    };
    let name = "model.language_model.layers.3.self_attn.k_proj.weight";
    let tensor = weights::expect(name);
    let matrix = tensor.matrix();
    let input: Vec<f32> = (0..matrix.columns)
        .map(|index| (index as f32 * 0.017).sin())
        .collect();

    let mut expected = vec![0.0f32; matrix.rows];
    ops::linear(&input, 1, &matrix, None, &mut expected);

    let mut actual = vec![0.0f32; matrix.rows];
    device.quantised_matvec(name, &input, &mut actual);

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for (row, (left, right)) in expected.iter().zip(&actual).enumerate() {
        let scale = left.abs().max(1.0);
        let relative = (left - right).abs() / scale;
        if relative > worst {
            worst = relative;
            worst_row = row;
        }
    }
    assert!(
        worst < 1e-5,
        "{name}: row {worst_row} differs by {worst} ({} vs {})",
        expected[worst_row],
        actual[worst_row]
    );
}

/// The container's weights, seen as a pass' weight source.
struct Container;

impl backend::Weights for Container {
    fn matrix(&self, name: &str) -> weights::Matrix {
        weights::expect(name).matrix()
    }
}

/// One layer's MLP as a recorded pass: the norm, both gate projections, the
/// activation, and the output projection.
///
/// The point of the gate is that the recording reaches the device whole: the
/// backend issues the pass as a few submissions of many dispatches rather than one
/// round trip per operator, and every stage has to match the portable executor.
#[test]
fn a_recorded_pass_reaches_the_device_whole() {
    let Some(device) = device() else {
        return;
    };
    const LAYER: &str = "model.language_model.layers.3";
    const HIDDEN: usize = 1024;
    const INTERMEDIATE: usize = 3584;

    let input: Vec<f32> = (0..HIDDEN).map(|index| (index as f32 * 0.013).sin()).collect();

    let mut flow = backend::Flow::new();
    let source = flow.bind(input.clone());
    let normalized = flow.buffer(HIDDEN);
    let gate = flow.buffer(INTERMEDIATE);
    let up = flow.buffer(INTERMEDIATE);
    let activated = flow.buffer(INTERMEDIATE);
    let out = flow.buffer(HIDDEN);

    let norm_weight = flow.bound(
        weights::expect(&format!("{LAYER}.post_attention_layernorm.weight")).values(),
    );
    flow.rms_norm(source, 1, HIDDEN, norm_weight, 1e-6, normalized);
    let gate_weight = flow.tensor(&format!("{LAYER}.mlp.gate_proj.weight"));
    flow.linear(normalized, 1, gate_weight, gate);
    let up_weight = flow.tensor(&format!("{LAYER}.mlp.up_proj.weight"));
    flow.linear(normalized, 1, up_weight, up);
    flow.silu_mul(gate, up, activated);
    let down_weight = flow.tensor(&format!("{LAYER}.mlp.down_proj.weight"));
    flow.linear(activated, 1, down_weight, out);
    let spec = flow.finish();

    assert_eq!(spec.ops.len(), 5, "the pass is five operators");
    assert_eq!(spec.tensors.len(), 3, "three tensors, one binding each");

    // --- the portable executor
    let mut arena = backend::Arena::new(&spec);
    let portable = std::time::Instant::now();
    backend::execute(&Container, &spec, &mut arena);
    let portable_seconds = portable.elapsed().as_secs_f64();

    // --- the device, which records the pass itself
    let device_arena = device.execute(&spec);

    // Every stage, so a failure says which kernel drifted instead of only that
    // the end of the pass did.
    let stages = [
        ("norm", normalized),
        ("gate", gate),
        ("up", up),
        ("activated", activated),
        ("projection", out),
    ];
    let mut worst = 0.0f32;
    for (name, slot) in stages {
        let buffer = buffer_of(&spec, slot);
        let left = arena.buffer(buffer);
        let right = device_arena.buffer(buffer);
        let mut worst_here = 0.0f32;
        for (a, b) in left.iter().zip(right) {
            worst_here = worst_here.max((a - b).abs() / a.abs().max(1.0));
        }
        assert!(
            worst_here < 1e-4,
            "{name}: the two executors differ by {worst_here}"
        );
        worst = worst.max(worst_here);
    }
    println!(
        "mlp pass: {} operators, {} tensors; portable {:.2} ms; worst {:.1e}",
        spec.ops.len(),
        spec.tensors.len(),
        1000.0 * portable_seconds,
        worst
    );
    println!(
        "mlp pass: {} operators, {} tensors; portable {:.2} ms; worst {:.1e}",
        spec.ops.len(),
        spec.tensors.len(),
        1000.0 * portable_seconds,
        worst
    );
}

/// The arena index a slot names.
fn buffer_of(spec: &backend::Spec, slot: backend::Slot) -> usize {
    match slot {
        backend::Slot::Buffer(index) => index,
        backend::Slot::Bound(_) | backend::Slot::External(_) => {
            let _ = spec;
            panic!("the slot is not an arena buffer")
        }
    }
}

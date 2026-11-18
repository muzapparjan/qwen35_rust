//! The kernels the model's pass does not reach.
//!
//! The whole-model gate (`tests/gpu_model.rs`) reaches fifteen of the twenty-one
//! operators. `layer_norm`, `softmax`, `argmax`, `sum` and `scale` are part of the
//! backend's contract but no recorded pass emits them yet, so nothing would notice
//! if they were wrong. This file runs each one through both executors on the same
//! spec. (`dequantize` and the quantised matvec are gated by `tests/gpu.rs`.)
//!
//! Same split as `tests/gpu.rs`: elementwise results must agree bit for bit, and a
//! reduction gets a relative tolerance.

#![cfg(feature = "wgpu")]

use qwen35::backend::{self, Arena, Flow, Slot};
use qwen35::gpu::Device;

fn device() -> Option<Device> {
    match Device::new() {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

/// Run a spec on both sides and return the CPU's answer and the device's.
fn both(flow: Flow, slot: Slot) -> (Vec<f32>, Vec<f32>) {
    let device = device().expect("a device");
    let spec = flow.finish();
    let index = match slot {
        Slot::Buffer(index) => index,
        _ => panic!("the compared slot is an arena buffer"),
    };
    let mut host = Arena::new(&spec);
    backend::execute(&Container, &spec, &mut host);
    let arena = device.execute(&spec);
    (host.buffer(index).to_vec(), arena.buffer(index).to_vec())
}

/// The CPU executor's weight source: the container, by name.
struct Container;

impl backend::Weights for Container {
    fn matrix(&self, name: &str) -> qwen35::weights::Matrix {
        qwen35::weights::expect(name).matrix()
    }
}

fn worst(expected: &[f32], actual: &[f32]) -> f32 {
    expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn differences(expected: &[f32], actual: &[f32]) -> usize {
    expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count()
}

#[test]
fn the_device_multiplies_a_vector() {
    let Some(_) = device() else {
        return;
    };
    let values: Vec<f32> = (0..64).map(|index| index as f32 * 0.25 - 8.0).collect();
    let mut flow = Flow::new();
    let input = flow.bind(values.clone());
    let out = flow.buffer(values.len());
    flow.scale(input, 0.75, out);

    let (expected, actual) = both(flow, out);
    assert_eq!(
        differences(&expected, &actual),
        0,
        "scale_all differs: worst {}",
        worst(&expected, &actual)
    );
}

#[test]
fn the_device_normalises_a_layer() {
    let Some(_) = device() else {
        return;
    };
    let rows = 4;
    let width = 32;
    let values: Vec<f32> = (0..rows * width)
        .map(|index| ((index as f32) * 0.11).sin() * 3.0)
        .collect();
    let weight: Vec<f32> = (0..width).map(|index| 1.0 + index as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..width).map(|index| index as f32 * 0.002).collect();

    let mut flow = Flow::new();
    let input = flow.bind(values);
    let scale = flow.bound(weight);
    let shift = flow.bound(bias);
    let out = flow.buffer(rows * width);
    flow.layer_norm(input, rows, width, scale, shift, 1e-6, out);

    let (expected, actual) = both(flow, out);
    let error = worst(&expected, &actual);
    assert!(error < 1e-5, "layer_norm differs by {error:.3e}");
}

#[test]
fn the_device_softmaxes_a_row() {
    let Some(_) = device() else {
        return;
    };
    let rows = 3;
    let width = 40;
    let values: Vec<f32> = (0..rows * width)
        .map(|index| ((index as f32) * 0.37).cos() * 5.0)
        .collect();

    let mut flow = Flow::new();
    let input = flow.bind(values);
    let out = flow.buffer(rows * width);
    flow.softmax(input, rows, width, out);

    let (expected, actual) = both(flow, out);
    let error = worst(&expected, &actual);
    assert!(error < 1e-6, "softmax differs by {error:.3e}");
    // Every row sums to one, on both sides: this is the property a wrong
    // normalisation would break even if the values looked plausible.
    for row in 0..rows {
        let host: f32 = expected[row * width..(row + 1) * width].iter().sum();
        let device: f32 = actual[row * width..(row + 1) * width].iter().sum();
        assert!((host - 1.0).abs() < 1e-5, "the host row sums to {host}");
        assert!((device - 1.0).abs() < 1e-5, "the device row sums to {device}");
    }
}

#[test]
fn the_device_takes_the_argmax_of_a_row() {
    let Some(_) = device() else {
        return;
    };
    let rows = 3;
    let width = 1000;
    let mut values = vec![0.0f32; rows * width];
    // A tie is not a tie here: each row's maximum is unique, so the comparison is
    // about which element won rather than about how a tie broke.
    for row in 0..rows {
        for column in 0..width {
            values[row * width + column] = ((row * 7 + column) as f32 * 0.13).sin();
        }
        values[row * width + 100 * (row + 1)] = 5.0;
    }

    let mut flow = Flow::new();
    let input = flow.bind(values);
    let out = flow.buffer(rows);
    flow.argmax(input, rows, width, out);

    let (expected, actual) = both(flow, out);
    assert_eq!(differences(&expected, &actual), 0, "argmax disagrees");
    for row in 0..rows {
        assert_eq!(actual[row] as usize, 100 * (row + 1), "row {row}");
    }
}

#[test]
fn the_device_sums_a_run() {
    let Some(_) = device() else {
        return;
    };
    let length = 4097;
    let values: Vec<f32> = (0..length)
        .map(|index| ((index as f32) * 0.017).sin())
        .collect();
    let expected_sum: f64 = values.iter().map(|value| *value as f64).sum();

    let mut flow = Flow::new();
    let input = flow.bind(values);
    let out = flow.buffer(1);
    flow.sum(input, length, out);

    let (expected, actual) = both(flow, out);
    let error = ((expected[0] - actual[0]) as f64).abs();
    assert!(
        error < 1e-3,
        "sum differs by {error:.3e}: {} vs {} (exact {expected_sum})",
        expected[0],
        actual[0]
    );
}

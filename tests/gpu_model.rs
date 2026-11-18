//! The whole recorded pass on the device, against the same pass on the host.
//!
//! [`tests/gpu.rs`] gates single kernels. That is not enough: a kernel that adds
//! the right numbers to the wrong place agrees with itself. Here the entire
//! forward pass — every layer, both attention kinds, the output head — runs as one
//! submission and the logits are compared against the host executor running the
//! same spec.
//!
//! The host executor is bit-identical to [`qwen35::model::Model::forward`] (see
//! `tests/recorded.rs`), so the logits compared here are the model's own.
//!
//! The gate is relative, because a reduction's association order is the
//! implementation's choice. It is tight enough to be a gate rather than a smoke
//! test: a defect that puts a tensor in the wrong place moves a logit by a factor,
//! not by an `f32`'s last bit.

#![cfg(feature = "wgpu")]

use qwen35::backend::{Arena, Slot, Spec, layout};
use qwen35::gpu::{Device, DeviceState};
use qwen35::model::record::execute_recorded;
use qwen35::model::{Cache, Model, NoTrace};

fn device() -> Option<Device> {
    match Device::new() {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

fn prompt() -> Vec<u32> {
    qwen35::tokenizer::tokenizer()
        .encode("The capital of France is")
        .into_iter()
        .take(9)
        .collect()
}

fn positions(count: usize, past: usize) -> Vec<[u32; 3]> {
    (0..count).map(|token| [(past + token) as u32; 3]).collect()
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best as u32
}

/// The logits a pass left in the slot it named.
fn logits_of(arena: &Arena, spec: &Spec, logits: Slot) -> Vec<f32> {
    match logits {
        Slot::Buffer(index) => {
            assert!(index < spec.buffers.len(), "the logits index");
            arena.buffer(index).to_vec()
        }
        _ => panic!("the logits are an arena buffer"),
    }
}

/// Every operator of a pass, against the host executor on the same spec.
#[test]
fn the_whole_pass_runs_on_the_device() {
    let Some(device) = device() else {
        return;
    };
    let model = Model::new();
    let ids = prompt();
    let recorded = model.record(&ids, ids.len(), &positions(ids.len(), 0), &Cache::new());
    eprintln!(
        "{} operators, {} tensors, {} state buffers",
        recorded.spec.ops.len(),
        recorded.spec.tensors.len(),
        recorded.spec.externals.len()
    );

    let (expected, _) = execute_recorded(&recorded);

    // The state starts empty, exactly as the recording declared it.
    let mut state = DeviceState::seeded(&recorded.spec, &recorded.state);
    let arena = device.execute_stateful(&recorded.spec, &mut state);
    let actual = logits_of(&arena, &recorded.spec, recorded.logits);
    assert_eq!(actual.len(), expected.len(), "logit count");

    let mut relative = 0.0f32;
    let mut at = 0usize;
    for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
        let distance = (left - right).abs() / left.abs().max(1.0);
        if distance > relative {
            relative = distance;
            at = index;
        }
    }
    eprintln!(
        "worst relative {relative:.3e} at {at} ({} vs {})",
        expected[at], actual[at]
    );
    assert!(
        relative < 1e-3,
        "logits differ by {relative:.3e} at {at}: {} expected, {} on the device",
        expected[at],
        actual[at]
    );
    // The turn the model would take has to be the same one: a tolerance that
    // admitted a different argmax would admit a different answer.
    assert_eq!(
        argmax(&expected),
        argmax(&actual),
        "the two paths chose different tokens"
    );
}

/// A decode loop, with the state staying in device buffers.
///
/// The first call seeds the caches; each later call declares one more position, so
/// the buffers are replaced while their prefix has to survive. The gate is the
/// tokens: a cache lost between steps changes what the model attends to, and the
/// turn diverges.
#[test]
fn the_device_carries_its_state_between_steps() {
    let Some(device) = device() else {
        return;
    };
    let model = Model::new();
    let ids = prompt();
    let prompt_length = 5;

    // The host path, one token at a time.
    let mut cache = Cache::new();
    let mut logits = model.forward(&ids[..prompt_length], &mut cache, &mut NoTrace);
    let mut host = Vec::new();
    for _ in 0..3 {
        let next = argmax(&logits);
        host.push(next);
        logits = model.forward(&[next], &mut cache, &mut NoTrace);
    }

    // The device path. One `DeviceState` across the steps: the buffers the first
    // call created are the ones the later calls write into.
    let recorded = model.record(
        &ids[..prompt_length],
        prompt_length,
        &positions(prompt_length, 0),
        &Cache::new(),
    );
    let mut state = DeviceState::seeded(&recorded.spec, &recorded.state);
    let arena = device.execute_stateful(&recorded.spec, &mut state);
    let mut logits = logits_of(&arena, &recorded.spec, recorded.logits);

    let mut device_tokens = Vec::new();
    let mut past = prompt_length;
    for _ in 0..3 {
        let next = argmax(&logits);
        device_tokens.push(next);
        // One more position: the declared lengths grow, so the buffers are
        // replaced and the prefix is copied forward. The state stays on the
        // device, so the recording does not gather it.
        let step = model.record_step(&[next], 1, &positions(1, past), past);
        let arena = device.execute_stateful(&step.spec, &mut state);
        logits = logits_of(&arena, &step.spec, step.logits);
        past += 1;
    }

    assert_eq!(host, device_tokens, "the two paths chose different tokens");
}

/// The arena holds the pass' high-water mark, not the sum of its intermediates.
///
/// A recording declares a slot for every value the pass ever makes, and a prefill
/// makes one per operator per token. Sharing storage between slots whose lifetimes
/// do not overlap is only sound if it changes nothing, so this runs the pass twice
/// on the device — once with every slot wanted, which keeps every slot's storage to
/// itself, and once with only the logits wanted — and requires the same numbers.
#[test]
fn the_arena_shares_storage_between_slots_that_are_not_live_at_once() {
    let Some(device) = device() else {
        return;
    };
    let model = Model::new();
    let ids = prompt();
    let recorded = model.record(
        &ids,
        ids.len(),
        &positions(ids.len(), 0),
        &Cache::new(),
    );
    let logits = match recorded.logits {
        Slot::Buffer(index) => index,
        _ => panic!("the logits are an arena buffer"),
    };

    let declared: usize = recorded.spec.buffers.iter().sum();
    let mut all = vec![true; recorded.spec.buffers.len()];
    let whole = layout(&recorded.spec, &all);
    let mut some = vec![false; recorded.spec.buffers.len()];
    some[logits] = true;
    let shared = layout(&recorded.spec, &some);
    let kept: usize = shared.lengths().iter().sum();
    eprintln!(
        "{} slots, {:.1} MB declared; every slot wanted: {} storages, {:.1} MB; only the logits: {} storages, {:.1} MB",
        recorded.spec.buffers.len(),
        declared as f64 * 4.0 / 1e6,
        whole.lengths().len(),
        whole.lengths().iter().sum::<usize>() as f64 * 4.0 / 1e6,
        shared.lengths().len(),
        kept as f64 * 4.0 / 1e6
    );
    assert_eq!(
        whole.lengths().len(),
        recorded.spec.buffers.len(),
        "a wanted slot must keep storage of its own"
    );
    assert!(
        kept * 4 < declared,
        "sharing kept {kept} of {declared} values"
    );

    let mut state = DeviceState::seeded(&recorded.spec, &recorded.state);
    let everything = device.execute_stateful(&recorded.spec, &mut state);
    all.iter_mut().for_each(|flag| *flag = false);
    let mut state = DeviceState::seeded(&recorded.spec, &recorded.state);
    let only = device.execute_stateful_reading(&recorded.spec, &mut state, &[logits]);

    // The arena a caller reads one slot from holds that slot and the storage of
    // whatever is live beside it, not the pass' whole declaration.
    let held = only.held_bytes();
    assert!(
        held < declared * 4,
        "the arena holds {held} bytes of a pass that declares {} bytes",
        declared * 4
    );

    let left = everything.buffer(logits);
    let right = only.buffer(logits);
    assert_eq!(left.len(), right.len(), "logit count");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {index} differs between the shared and the private layout"
        );
    }
}

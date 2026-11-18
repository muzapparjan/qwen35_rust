//! The recorded forward pass against the direct one.
//!
//! [`Model::forward`] runs one kernel per operator on host slices;
//! [`Model::forward_recorded`] records the same pass as a dataflow graph and runs
//! it through one executor. They have to agree **bit for bit**: the recording is
//! the same arithmetic in the same order, so any difference is a defect in the
//! recording rather than a rounding difference.
//!
//! This is the gate the device backend stands on. If the recorded pass is not the
//! same function as the direct one, comparing a device against it compares the
//! device against something that is not the model.

use qwen35::model::{Cache, Model, NoTrace};

fn ids() -> Vec<u32> {
    // A short prompt: enough to exercise both attention kinds and the cache.
    qwen35::tokenizer::tokenizer()
        .encode("The capital of France is")
        .into_iter()
        .take(9)
        .collect()
}

#[test]
fn the_recorded_pass_is_the_direct_one_bit_for_bit() {
    let model = Model::new();
    let ids = ids();

    let mut direct_cache = Cache::new();
    let direct = model.forward(&ids, &mut direct_cache, &mut NoTrace);

    let mut recorded_cache = Cache::new();
    let recorded = model.forward_recorded(&ids, &mut recorded_cache);

    assert_eq!(direct.len(), recorded.len(), "logit count");
    let mut differing = 0usize;
    let mut at = 0usize;
    let mut worst = 0.0f32;
    for (index, (left, right)) in direct.iter().zip(&recorded).enumerate() {
        if left.to_bits() != right.to_bits() {
            differing += 1;
            let distance = (left - right).abs();
            if distance > worst {
                worst = distance;
                at = index;
            }
        }
    }
    assert_eq!(
        differing, 0,
        "{differing} of {} logits differ; worst {worst} at {at} ({} vs {})",
        direct.len(),
        direct[at],
        recorded[at]
    );
}

/// The same, for a pass that starts from a cache that is already filled: the
/// recorded path has to read the caller's state, not a zeroed copy.
#[test]
fn the_recorded_pass_reads_the_cache_it_is_given() {
    let model = Model::new();
    let ids = ids();
    let prompt = &ids[..4];

    let mut direct_cache = Cache::new();
    let _ = model.forward(prompt, &mut direct_cache, &mut NoTrace);
    let direct = model.forward(&ids[4..], &mut direct_cache, &mut NoTrace);

    let mut recorded_cache = Cache::new();
    let _ = model.forward(prompt, &mut recorded_cache, &mut NoTrace);
    let recorded = model.forward_recorded(&ids[4..], &mut recorded_cache);

    let mut differing = 0usize;
    for (left, right) in direct.iter().zip(&recorded) {
        if left.to_bits() != right.to_bits() {
            differing += 1;
        }
    }
    assert_eq!(differing, 0, "{differing} logits differ after a cached prefix");
}

/// Several steps in a row, with the state carried by the caller.
///
/// This is the shape a generation loop has: if the recorded pass did not carry its
/// state, the second step would diverge even though the first agreed.
#[test]
fn the_recorded_steps_carry_their_state() {
    let model = Model::new();
    let ids = ids();
    let prompt = &ids[..5];

    // The direct path, one token at a time.
    let mut direct_cache = Cache::new();
    let mut logits = model.forward(prompt, &mut direct_cache, &mut NoTrace);
    let mut direct = Vec::new();
    for _ in 0..3 {
        let next = top(&logits);
        direct.push(next);
        logits = model.forward(&[next], &mut direct_cache, &mut NoTrace);
    }

    // The recorded path, carrying the state between steps.
    let mut recorded = Vec::new();
    let initial = Cache::new();
    let (mut logits, mut state) = model.forward_recorded_step(prompt, 0, &initial_state(&initial));
    let mut past = prompt.len();
    for _ in 0..3 {
        let next = top(&logits);
        recorded.push(next);
        let (next_logits, next_state) = model.forward_recorded_step(&[next], past, &state);
        logits = next_logits;
        state = next_state;
        past += 1;
    }

    assert_eq!(direct, recorded, "the two paths chose different tokens");
}

fn top(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// The state of an empty cache, in the order the recorder expects: one key cache
/// and one value cache per full-attention layer, then the convolution window and
/// the recurrent state per linear-attention layer.
fn initial_state(cache: &Cache) -> Vec<Vec<f32>> {
    let mut state = Vec::new();
    for _ in &cache.attention {
        state.push(Vec::new());
        state.push(Vec::new());
    }
    for linear in &cache.linear {
        state.push(linear.conv.clone());
        state.push(linear.state.clone());
    }
    state
}

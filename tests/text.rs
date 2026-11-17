//! The text model, layer by layer, against tensors dumped from the reference.
//!
//! ```text
//! python dev/dump_reference.py
//! cargo test --release --features reference-fixtures --test text -- --nocapture
//! ```
//!
//! The dumps are produced from the **dequantised** container, so a difference is
//! a difference in arithmetic rather than in quantisation.

#![cfg(feature = "reference-fixtures")]

mod common;

use std::collections::HashMap;
use std::path::Path;

use qwen35::model::{Cache, Model, NoTrace, Trace};

struct Recording {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Trace for Recording {
    fn record(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        // The dump writes `layers.3.self_attn.out` as `layers_3_self_attn_out`
        // because the name becomes a file name there; the comparison uses the
        // same spelling on this side.
        self.tensors
            .insert(name.replace('.', "_"), (shape.to_vec(), values.to_vec()));
    }
}

fn directory(case: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("dev")
        .join("fixtures")
        .join("text")
        .join(case)
}

fn read_ids(case: &str) -> Vec<u32> {
    let path = directory(case).join("inputs.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}\nrun `python dev/dump_reference.py` from the repository root first",
            path.display()
        )
    });
    text.lines().map(|line| line.trim().parse().expect("id")).collect()
}

struct Expected {
    name: String,
    values: Vec<f32>,
}

fn read_expected(case: &str) -> Vec<Expected> {
    let base = directory(case);
    let manifest = std::fs::read_to_string(base.join("manifest.txt")).expect("manifest");
    let mut expected = Vec::new();
    for line in manifest.lines() {
        let mut fields = line.split('\t');
        let name = fields.next().expect("name").to_string();
        let shape: Vec<usize> = fields
            .next()
            .expect("shape")
            .split(',')
            .map(|dimension| dimension.parse().expect("dimension"))
            .collect();
        let bytes = std::fs::read(base.join(format!("{name}.bin"))).expect("tensor file");
        assert_eq!(bytes.len(), 4 * shape.iter().product::<usize>(), "{name}");
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        expected.push(Expected { name, values });
    }
    expected
}

fn metrics(expected: &[f32], actual: &[f32]) -> (f64, f64) {
    assert_eq!(expected.len(), actual.len(), "length");
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    for (wanted, got) in expected.iter().zip(actual) {
        worst = worst.max(f64::from((wanted - got).abs()));
        peak = peak.max(f64::from(wanted.abs()));
    }
    (worst, worst / peak.max(f64::MIN_POSITIVE))
}

/// The convolution is not a module in the reference, so it is not dumped; the
/// checkpoints that are dumped start at the layer's normalised input.
#[test]
fn every_text_layer_matches_the_reference() {
    let cases = ["hello", "chinese", "mixed"];
    let mut failures = Vec::new();
    for case in cases {
        let ids = read_ids(case);
        let expected = read_expected(case);
        let mut recording = Recording {
            tensors: HashMap::new(),
        };
        let model = Model::new();
        let mut cache = Cache::new();
        let logits = model.forward(&ids, &mut cache, &mut recording);

        println!("== {case} ({} tokens) ==", ids.len());
        let mut worst_layer: (f64, String) = (0.0, String::new());
        for entry in &expected {
            let Some((shape, actual)) = recording.tensors.get(&entry.name) else {
                failures.push(format!("{case}: {:?} was not recorded", entry.name));
                continue;
            };
            let values = actual.clone();
            let _ = shape;
            if values.len() != entry.values.len() {
                failures.push(format!(
                    "{case}: {:?}: {} values against {}",
                    entry.name,
                    values.len(),
                    entry.values.len()
                ));
                continue;
            }
            let (worst, relative) = metrics(&entry.values, &values);
            let layer = entry
                .name
                .strip_prefix("layers_")
                .and_then(|rest| rest.split('_').next())
                .and_then(|index| index.parse::<usize>().ok());
            if let Some(layer) = layer {
                if relative > worst_layer.0 {
                    worst_layer = (relative, format!("layer {layer} {}", entry.name));
                }
            }
            if entry.name.starts_with("layers_") && entry.name.ends_with("_out") {
                let rest = &entry.name["layers_".len()..];
                if rest.split_once('_').map(|(_, tail)| tail) == Some("out") {
                    println!("  {:<44} max|d| {worst:.3e}  rel {relative:.3e}", entry.name);
                }
            }
            let gate = if entry.name.starts_with("logits") { 3e-3 } else { 1e-3 };
            if relative > gate || !(worst == worst) {
                failures.push(format!(
                    "{case}: {:?}: max|d| {worst:.3e}  rel {relative:.3e}  (gate {gate:.0e})",
                    entry.name
                ));
            }
        }
        let reference_logits = expected
            .iter()
            .find(|entry| entry.name == "logits_last")
            .expect("logits were dumped");
        let top_expected = top(&reference_logits.values);
        let top_actual = top(&logits);
        println!("  top-1: port {top_actual} reference {top_expected}");
        if top_actual != top_expected {
            failures.push(format!("{case}: top-1 {top_actual} against {top_expected}"));
        }
        println!("  worst layer: {} rel {:.3e}", worst_layer.1, worst_layer.0);
    }
    assert!(
        failures.is_empty(),
        "{} checkpoints differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn top(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// A model with no trace still produces the same logits as one with a trace.
#[test]
fn tracing_does_not_change_the_answer() {
    let ids = read_ids("hello");
    let model = Model::new();
    let mut cache = Cache::new();
    let plain = model.forward(&ids, &mut cache, &mut NoTrace);
    let mut recording = Recording {
        tensors: HashMap::new(),
    };
    let mut cache = Cache::new();
    let traced = model.forward(&ids, &mut cache, &mut recording);
    assert_eq!(plain, traced);
    assert!(recording.tensors.contains_key("final_norm_out"));
}

/// One call for the whole prompt and one call per token must agree.
///
/// They go through the same cached state, so the difference is exactly zero
/// here; the reference's own two paths differ by 6.3e-7 because its prefill uses
/// the chunked form of the delta rule.
#[test]
fn prefill_and_decode_agree() {
    let ids = read_ids("mixed");
    let model = Model::new();

    let mut cache = Cache::new();
    let whole = model.forward(&ids, &mut cache, &mut NoTrace);

    let mut cache = Cache::new();
    let split = ids.len() / 2;
    let _ = model.forward(&ids[..split], &mut cache, &mut NoTrace);
    let mut stepped = Vec::new();
    for id in &ids[split..] {
        stepped = model.forward(&[*id], &mut cache, &mut NoTrace);
    }

    let (worst, relative) = metrics(&whole, &stepped);
    println!("prefill against decode: max|d| {worst:.3e}  rel {relative:.3e}");
    assert_eq!(worst, 0.0, "the two paths are the same code");
    assert_eq!(top(&whole), top(&stepped));
}

/// Only the last position is ever projected into the vocabulary.
#[test]
fn only_the_last_position_is_projected() {
    let ids = read_ids("mixed");
    let model = Model::new();
    let mut recording = Recording {
        tensors: HashMap::new(),
    };
    let mut cache = Cache::new();
    let logits = model.forward(&ids, &mut cache, &mut recording);
    assert_eq!(logits.len(), qwen35::weights::manifest::VOCAB_SIZE);
    let (shape, values) = recording.tensors.get("logits_last").expect("recorded");
    assert_eq!(shape, &[qwen35::weights::manifest::VOCAB_SIZE]);
    assert_eq!(values.len(), logits.len());
}
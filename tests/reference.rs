//! Comparison against the reference implementation's own reader.
//!
//! The reference for M1 is `reference.quant.quantlib.read_tensor`, which is what
//! produced the checkpoint's quality numbers; reproducing it bit for bit is the
//! criterion, so this file compares bytes and reports `max|d|` on full dumps.
//!
//! The dumps are not part of the repository, so this file builds only with the
//! `reference-fixtures` feature:
//!
//! ```text
//! python dev/dump_dequant.py
//! cargo test --release --features reference-fixtures --test reference -- --nocapture
//! ```
//!
//! Without the dumps the tests fail rather than pass silently.

#![cfg(feature = "reference-fixtures")]

mod common;

use std::path::Path;

use qwen35::weights::{self, Data};

/// One line per tensor, as `dev/dump_dequant.py` writes it.
struct Expected {
    name: String,
    count: usize,
    digest: u32,
    absolute_max: f64,
}

fn read_expected() -> Vec<Expected> {
    let path = common::fixture("dequant.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}\nrun `python dev/dump_dequant.py` from the repository root first",
            path.display()
        )
    });
    let mut expected = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let mut fields = line.split('\t');
        let name = fields.next().unwrap_or_default().to_string();
        let count = fields
            .next()
            .unwrap_or_else(|| panic!("line {}: no count", number + 1))
            .parse()
            .unwrap_or_else(|_| panic!("line {}: count is not a number", number + 1));
        let digest = u32::from_str_radix(
            fields
                .next()
                .unwrap_or_else(|| panic!("line {}: no digest", number + 1)),
            16,
        )
        .unwrap_or_else(|_| panic!("line {}: digest is not hex", number + 1));
        let absolute_max = fields
            .next()
            .unwrap_or_else(|| panic!("line {}: no maximum", number + 1))
            .parse()
            .unwrap_or_else(|_| panic!("line {}: maximum is not a number", number + 1));
        expected.push(Expected {
            name,
            count,
            digest,
            absolute_max,
        });
    }
    expected
}

#[test]
fn every_tensor_dequantises_exactly_like_the_reference() {
    let expected = read_expected();
    assert!(
        expected.len() >= 488,
        "{} lines: the dump does not cover the container",
        expected.len()
    );

    let mut checked = 0usize;
    let mut values = 0usize;
    let mut failures = Vec::new();
    let started = std::time::Instant::now();
    for entry in &expected {
        let tensor = match weights::tensor(&entry.name) {
            Some(tensor) => tensor,
            None => {
                failures.push(format!("{}: missing from the container", entry.name));
                continue;
            }
        };
        if tensor.numel() != entry.count {
            failures.push(format!(
                "{}: {} values, the reference has {}",
                entry.name,
                tensor.numel(),
                entry.count
            ));
            continue;
        }
        let actual = tensor.values();
        let digest = common::crc32(&common::f32_le_bytes(&actual));
        if digest != entry.digest {
            let actual_max = actual
                .iter()
                .fold(0.0f64, |worst, value| worst.max(f64::from(value.abs())));
            failures.push(format!(
                "{}: crc32 {digest:08x} against {:08x}; |value| max {actual_max:.6e} against {:.6e}",
                entry.name, entry.digest, entry.absolute_max
            ));
            continue;
        }
        checked += 1;
        values += actual.len();
        // Nothing to assert about memory here; the table is freed per tensor.
        drop(actual);
    }

    println!(
        "tensors {checked}/{}  values {values}  elapsed {:.2} s",
        expected.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(
        failures.is_empty(),
        "{} tensors differ from the reference:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn dumped_tensors_are_element_for_element_identical() {
    let directory = common::fixture("dequant");
    let mut dumps = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("f32") {
                dumps.push(path);
            }
        }
    }
    assert!(
        !dumps.is_empty(),
        "{}: no dumps; run `python dev/dump_dequant.py` first",
        directory.display()
    );
    dumps.sort();

    for path in &dumps {
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("dump file name");
        let tensor = weights::tensor(name)
            .unwrap_or_else(|| panic!("{}: the container has no tensor {name:?}", path.display()));
        let bytes = std::fs::read(path).expect("dump is readable");
        assert_eq!(
            bytes.len(),
            4 * tensor.numel(),
            "{name}: the dump is {} bytes for {} values",
            bytes.len(),
            tensor.numel()
        );
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let actual = tensor.values();
        let mut worst = 0.0f64;
        let mut first_difference = None;
        for (index, (wanted, got)) in expected.iter().zip(&actual).enumerate() {
            let difference = f64::from((wanted - got).abs());
            if difference > worst {
                worst = difference;
            }
            if difference != 0.0 && first_difference.is_none() {
                first_difference = Some((index, *wanted, *got));
            }
        }
        let scale = expected
            .iter()
            .fold(0.0f64, |largest, value| largest.max(f64::from(value.abs())))
            .max(f64::MIN_POSITIVE);
        println!(
            "{name:<70} {} values  max|d| {worst:.3e}  rel {:.3e}",
            expected.len(),
            worst / scale
        );
        if let Some((index, wanted, got)) = first_difference {
            panic!("{name}: first difference at {index}: reference {wanted:?}, port {got:?}");
        }
    }
    println!("{} tensors compared element by element", dumps.len());
}

#[test]
fn the_dump_describes_the_embedded_file() {
    // A dump taken from another container would make every comparison above
    // meaningless, so the two have to be the same model by construction.
    let container = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/weights/qwen3.5-0.8b-q4.qwqt");
    let bytes = std::fs::read(&container).expect("the container is in the repository");
    assert_eq!(bytes.len(), weights::manifest::FILE_LEN);
    assert_eq!(&bytes[..4], b"QWQT");
    let expected = read_expected();
    assert_eq!(expected.len(), weights::manifest::TENSOR_COUNT);
    for (entry, spec) in expected.iter().zip(weights::manifest::TENSORS) {
        assert_eq!(entry.name, spec.name);
        assert_eq!(entry.count, spec.shape.iter().product::<usize>());
    }
    match weights::expect("model.language_model.embed_tokens.weight").data {
        Data::Quantized { .. } => {}
        other => panic!("the embedding table should be quantised, got {other:?}"),
    }
}

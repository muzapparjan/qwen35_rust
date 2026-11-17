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

/// One corpus line: the text, the pieces the reference's pre-tokeniser made, and
/// the ids the reference tokenizer produced.
struct TokenCase {
    text: String,
    pieces: Vec<String>,
    ids: Vec<u32>,
}

fn read_token_cases() -> Vec<TokenCase> {
    let path = common::fixture("tokenizer.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}\nrun `python dev/dump_tokenizer.py` from the repository root first",
            path.display()
        )
    });
    let mut cases = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let mut fields = line.split('\t');
        let decode = |field: &str| -> String {
            if field == "-" {
                return String::new();
            }
            let bytes: Vec<u8> = (0..field.len())
                .step_by(2)
                .map(|index| u8::from_str_radix(&field[index..index + 2], 16).expect("hex"))
                .collect();
            String::from_utf8(bytes).expect("the corpus is UTF-8")
        };
        let source = fields.next().expect("text");
        let pieces = fields
            .next()
            .unwrap_or_else(|| panic!("line {}: no pieces", number + 1));
        let ids = fields
            .next()
            .unwrap_or_else(|| panic!("line {}: no ids", number + 1));
        cases.push(TokenCase {
            text: decode(source),
            pieces: if pieces == "-" {
                Vec::new()
            } else {
                pieces.split(' ').map(decode).collect()
            },
            ids: if ids == "-" {
                Vec::new()
            } else {
                ids.split(' ').map(|id| id.parse().expect("id")).collect()
            },
        });
    }
    cases
}

/// The full pipeline: added tokens, NFC, pre-tokeniser, BPE.
///
/// This is M2's criterion. Pieces are compared as well as ids, because a wrong
/// split can still produce the right ids for short text and would leave the
/// cause invisible.
#[test]
fn the_tokenizer_matches_the_reference_on_a_corpus() {
    use qwen35::tokenizer::{normalize, pieces, tokenizer};

    let cases = read_token_cases();
    assert!(cases.len() >= 50, "{} cases", cases.len());
    let tokenizer = tokenizer();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for case in &cases {
        let ids = tokenizer.encode(&case.text);
        if ids != case.ids {
            failures.push(format!(
                "{:?}: ids {ids:?}, reference {:?}",
                case.text, case.ids
            ));
            continue;
        }
        // The dump records the pieces of the normalised text as a whole, before
        // added tokens are cut out, so this is the piece-level comparison.
        let normalized = normalize(&case.text);
        let actual: Vec<&str> = pieces(&normalized);
        let expected: Vec<&str> = case.pieces.iter().map(String::as_str).collect();
        if actual != expected {
            failures.push(format!(
                "{:?}: pieces {actual:?}, reference {expected:?}",
                case.text
            ));
            continue;
        }
        checked += 1;
    }
    println!("tokenizer corpus: {checked}/{} cases", cases.len());
    assert!(
        failures.is_empty(),
        "{} cases differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The chat template, checked against the release's own Jinja output.
///
/// The fixture carries the messages as JSON because that is what the dump script
/// has; the Rust side rebuilds the same messages from the case name, so the thing
/// compared is the template's output rather than the fixture's encoding.
#[test]
fn the_chat_template_matches_the_release() {
    use qwen35::chat::{self, Message, Options};

    let path = common::fixture("chat.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}\nrun `python dev/dump_chat.py` from the repository root first",
            path.display()
        )
    });

    let mut checked = 0usize;
    let mut failures = Vec::new();
    for line in text.lines() {
        let mut fields = line.split('\t');
        let name = fields.next().expect("case name");
        let _messages = fields.next().expect("messages");
        let thinking = fields.next().expect("thinking flag") == "1";
        let encoded = fields.next().expect("output");
        let expected: Vec<u8> = (0..encoded.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&encoded[index..index + 2], 16).expect("hex"))
            .collect();
        let expected = String::from_utf8(expected).expect("UTF-8");

        let messages: Vec<Message> = match name {
            "plain_user" | "plain_user_thinking" => vec![Message::user("你好")],
            "system_then_user" => vec![
                Message::system("You are helpful."),
                Message::user("Hi"),
            ],
            "multi_turn" => vec![
                Message::system("Be brief."),
                Message::user("What is 2+2?"),
                Message::assistant("4"),
                Message::user("And 3+3?"),
            ],
            "assistant_reasoning" => vec![
                Message::user("Add them."),
                Message::assistant_thinking("4", "two plus two"),
                Message::user("Thanks"),
            ],
            "whitespace_is_trimmed" => vec![
                Message::system("  padded  \n"),
                Message::user("\n  spaced  \n"),
            ],
            "content_parts" => vec![Message::user("part one, part two")],
            "empty_user" => vec![Message::user("")],
            other => panic!("{other}: no messages are defined for this case"),
        };

        let options = Options {
            add_generation_prompt: true,
            enable_thinking: thinking,
        };
        let actual = chat::render(&messages, &options).expect("renders");
        if actual != expected {
            failures.push(format!("{name}:\n  port     {actual:?}\n  reference {expected:?}"));
            continue;
        }
        checked += 1;
    }
    println!("chat template: {checked} cases");
    assert!(
        failures.is_empty(),
        "{} cases differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

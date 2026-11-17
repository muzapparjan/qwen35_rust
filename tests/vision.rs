//! The vision tower and the image preprocessing, against the reference's dumps.
//!
//! ```text
//! python dev/dump_vision.py
//! cargo test --release --features reference-fixtures --test vision -- --nocapture
//! ```
//!
//! The dumps come from the **dequantised** container, so a difference is a
//! difference in arithmetic rather than in quantisation. Two halves are checked
//! separately, because they fail for different reasons: the preprocessing (raw
//! bytes in, patch vectors out) and the tower (patch vectors in, merged tokens
//! out).

#![cfg(feature = "reference-fixtures")]

use std::path::{Path, PathBuf};

use qwen35::model::{Cache, Model, NoTrace};
use qwen35::multimodal;
use qwen35::vision;

fn directory(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("dev")
        .join("fixtures")
        .join("vision")
        .join(case)
}

fn read_settings(case: &str) -> Vec<(String, String)> {
    let path = directory(case).join("case.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}\nrun `python dev/dump_vision.py` from the repository root first",
            path.display()
        )
    });
    text.lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn setting(settings: &[(String, String)], key: &str) -> String {
    settings
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| panic!("{key} missing from case.txt"))
}

fn read_floats(case: &str, name: &str) -> Vec<f32> {
    let path = directory(case).join(format!("{name}.bin"));
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
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

const CASES: [&str; 2] = ["resized", "square"];

/// Raw bytes in, patch vectors out.
#[test]
fn the_preprocessing_matches_the_processor() {
    for case in CASES {
        let settings = read_settings(case);
        let height: usize = setting(&settings, "height").parse().expect("height");
        let width: usize = setting(&settings, "width").parse().expect("width");
        let image = std::fs::read(directory(case).join("image.bin")).expect("image");
        let expected = read_floats(case, "pixel_values");

        let (values, patch_h, patch_w, resized_height, resized_width) =
            vision::preprocess::pixel_values(&image, height, width).expect("preprocess");
        let (worst, relative) = metrics(&expected, &values);
        println!(
            "{case}: {height}x{width} -> {resized_height}x{resized_width} \
             ({patch_h}x{patch_w} patches)  max|d| {worst:.3e}  rel {relative:.3e}"
        );
        // One step of a byte is 1/127.5 in these units. The reference resamples
        // in a fixed-point buffer, so its own output carries that granularity;
        // requiring less would ascribe precision to the fixture that it does not
        // have. The measured difference is exactly one step on the scaling case
        // and 3.5e-8 on the one whose sides are already a multiple of 32.
        assert!(
            worst <= 1.0 / 127.5 + 1e-6,
            "{case}: preprocessing differs by {worst:.3e}"
        );
    }
}

/// Patch vectors in, merged tokens out.
///
/// The tower is fed the **dumped** patch rows, so this measures the arithmetic
/// alone; the preprocessing is gated by its own test. Feeding it our own rows
/// instead mixes the two, and this tower amplifies its input by more than two
/// orders of magnitude, so a one-byte difference in preprocessing would show up
/// here as a difference in the tower.
#[test]
fn the_tower_matches_the_reference() {
    for case in CASES {
        let settings = read_settings(case);
        let grid_thw = setting(&settings, "grid_thw");
        let sizes: Vec<usize> = grid_thw
            .split(',')
            .map(|value| value.parse().expect("grid"))
            .collect();
        let grid = vision::Grid {
            temporal: sizes[0],
            height: sizes[1],
            width: sizes[2],
        };
        let patches = read_floats(case, "pixel_values");
        let expected = read_floats(case, "merger.out");

        let encoded = vision::encode_patches(&patches, grid);
        assert_eq!(encoded.values.len(), expected.len(), "{case}: token count");
        let (worst, relative) = metrics(&expected, &encoded.values);
        println!(
            "{case}: {} tokens from {} patches  max|d| {worst:.3e}  rel {relative:.3e}",
            encoded.tokens,
            grid.patches()
        );
        assert!(relative < 1e-3, "{case}: tower differs by {relative:.3e}");
    }
}

/// The whole way through, which is what a caller gets.
#[test]
fn the_whole_path_stays_close() {
    for case in CASES {
        let settings = read_settings(case);
        let height: usize = setting(&settings, "height").parse().expect("height");
        let width: usize = setting(&settings, "width").parse().expect("width");
        let image = std::fs::read(directory(case).join("image.bin")).expect("image");
        let expected = read_floats(case, "merger.out");
        let encoded = vision::encode(&image, height, width).expect("encode");
        let (_, relative) = metrics(&expected, &encoded.values);
        println!("{case}: end to end rel {relative:.3e}");
        // The preprocessing's one-byte difference, amplified by the tower.
        assert!(relative < 1e-2, "{case}: end to end differs by {relative:.3e}");
    }
}

/// An image prompt through the text model: the scatter, the multimodal positions
/// and the tower all at once.
///
/// The preprocessing's one-byte difference is in the input here, so this is the
/// loosest of the vision gates; the top-1 comparison is the part that matters,
/// because it says the answer survived.
#[test]
fn an_image_prompt_reaches_the_reference_logits() {
    let model = Model::new();
    for case in CASES {
        let settings = read_settings(case);
        let height: usize = setting(&settings, "height").parse().expect("height");
        let width: usize = setting(&settings, "width").parse().expect("width");
        let image = std::fs::read(directory(case).join("image.bin")).expect("image");
        let ids: Vec<u32> = std::fs::read_to_string(directory(case).join("prompt_ids.txt"))
            .expect("prompt ids")
            .lines()
            .map(|line| line.trim().parse().expect("id"))
            .collect();
        let expected = read_floats(case, "prompt_logits");

        let encoded = vision::encode(&image, height, width).expect("encode");
        let input = multimodal::Image {
            values: &encoded.values,
            grid: encoded.grid,
        };
        let mut cache = Cache::new();
        let logits = model
            .forward_prompt(&ids, &[input], &mut cache, &mut NoTrace)
            .expect("forward");
        let (_, relative) = metrics(&expected, &logits);
        let wanted = top(&expected);
        let got = top(&logits);
        println!(
            "{case}: {} ids  logits rel {relative:.3e}  top-1 {wanted} -> {got}",
            ids.len()
        );
        assert_eq!(wanted, got, "{case}: the answer changed");
        assert!(relative < 5e-2, "{case}: logits differ by {relative:.3e}");
    }
}

/// The token after an image prompt, which is where the multimodal position offset
/// shows up: its position is the prompt's largest plus one, not its length.
#[test]
fn the_token_after_an_image_prompt_continues_from_the_right_place() {
    let model = Model::new();
    for case in CASES {
        let settings = read_settings(case);
        let height: usize = setting(&settings, "height").parse().expect("height");
        let width: usize = setting(&settings, "width").parse().expect("width");
        let image = std::fs::read(directory(case).join("image.bin")).expect("image");
        let ids: Vec<u32> = std::fs::read_to_string(directory(case).join("prompt_ids.txt"))
            .expect("prompt ids")
            .lines()
            .map(|line| line.trim().parse().expect("id"))
            .collect();
        let next: u32 = std::fs::read_to_string(directory(case).join("prompt_next_id.txt"))
            .expect("next id")
            .trim()
            .parse()
            .expect("id");
        let expected = read_floats(case, "prompt_next_logits");

        let encoded = vision::encode(&image, height, width).expect("encode");
        let input = multimodal::Image {
            values: &encoded.values,
            grid: encoded.grid,
        };
        let mut cache = Cache::new();
        let logits = model
            .forward_prompt(&ids, &[input], &mut cache, &mut NoTrace)
            .expect("forward");
        assert_eq!(top(&logits), next as usize, "{case}: the prompt's top-1 moved");
        let stepped = model.forward(&[next], &mut cache, &mut NoTrace);
        let (_, relative) = metrics(&expected, &stepped);
        println!(
            "{case}: step after the prompt rel {relative:.3e}  top-1 {} -> {}",
            top(&expected),
            top(&stepped)
        );
        assert_eq!(top(&expected), top(&stepped), "{case}: the step's answer changed");
        assert!(relative < 5e-2, "{case}: the step differs by {relative:.3e}");
    }
}

/// The prompt a caller renders has to be the one the processor would build.
///
/// The template writes one `<|image_pad|>`; the processor expands it to one token
/// per merged patch. This checks the rendered text, the expansion and the special
/// tokens against the ids the processor emitted for the same question.
#[test]
fn the_rendered_prompt_is_the_processor_one() {
    use qwen35::chat::{self, Message, Options};
    use qwen35::tokenizer::tokenizer;

    for case in CASES {
        let settings = read_settings(case);
        let sizes: Vec<usize> = setting(&settings, "grid_thw")
            .split(',')
            .map(|value| value.parse().expect("grid"))
            .collect();
        let grid = vision::Grid {
            temporal: sizes[0],
            height: sizes[1],
            width: sizes[2],
        };
        let content = "<|vision_start|><|image_pad|><|vision_end|>What is in this image?";
        let text = chat::render(&[Message::user(content)], &Options::default()).expect("render");
        let ids = tokenizer().encode(&text);
        let expanded = multimodal::expand_placeholders(&ids, &[grid]).expect("expand");
        let expected: Vec<u32> = std::fs::read_to_string(directory(case).join("prompt_ids.txt"))
            .expect("prompt ids")
            .lines()
            .map(|line| line.trim().parse().expect("id"))
            .collect();
        println!("{case}: {} ids, processor {}", expanded.len(), expected.len());
        assert_eq!(expanded, expected, "{case}: the prompt differs");
    }
}

fn top(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

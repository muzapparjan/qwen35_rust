//! Describe an image, greedily.
//!
//! ```text
//! cargo run --release --example describe_image -- photo.ppm
//! cargo run --release --example describe_image -- --tokens 32 photo.ppm "What is in this image?"
//! ```
//!
//! The image is a binary PPM (`P6`, 8 bits per channel), which needs no decoder:
//! three bytes per pixel after a four-field header. Everything after that is the
//! library: resize, patches, twelve blocks, the merger, the scatter into the text
//! sequence, the multimodal positions, then generation.
//!
//! The prompt is the text template with one `<|image_pad|>` in it, expanded to one
//! token per merged patch — the same ids the reference processor emits, which
//! `tests/vision.rs` checks.

use qwen35::chat::{self, Message, Options};
use qwen35::model::{Cache, Model, NoTrace};
use qwen35::multimodal::{self, Image};
use qwen35::tokenizer::tokenizer;
use qwen35::vision;
use qwen35::weights::manifest;

fn main() {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut tokens = 32usize;
    if let Some(position) = arguments.iter().position(|value| value == "--tokens") {
        arguments.remove(position);
        tokens = arguments
            .remove(position)
            .parse()
            .expect("--tokens needs a number");
    }
    let path = arguments
        .first()
        .cloned()
        .unwrap_or_else(|| "photo.ppm".to_string());
    let question = if arguments.len() > 1 {
        arguments[1..].join(" ")
    } else {
        "What is in this image?".to_string()
    };

    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let (pixels, height, width) = read_ppm(&bytes).expect("a binary PPM of 8-bit channels");
    println!("image: {width}x{height}");

    let started = std::time::Instant::now();
    let encoded =
        vision::encode(&pixels, height, width).expect("the image has a resizable shape");
    println!(
        "vision: {:.2} s  {} patches -> {} tokens  (resized {}x{})",
        started.elapsed().as_secs_f64(),
        encoded.grid.height * encoded.grid.width,
        encoded.tokens,
        encoded.resized.0,
        encoded.resized.1
    );

    let tokenizer = tokenizer();
    let content = format!(
        "<|vision_start|><|image_pad|><|vision_end|>{question}"
    );
    let rendered = chat::render(&[Message::user(&content)], &Options::default())
        .expect("the conversation renders");
    let ids = tokenizer.encode(&rendered);
    let ids = multimodal::expand_placeholders(&ids, &[encoded.grid]).expect("one image");
    println!("prompt tokens: {}", ids.len());

    let model = Model::new();
    let mut cache = Cache::new();
    let input = Image {
        values: &encoded.values,
        grid: encoded.grid,
    };
    let started = std::time::Instant::now();
    let mut logits = model
        .forward_prompt(&ids, &[input], &mut cache, &mut NoTrace)
        .expect("one image");
    println!("prefill: {:.2} s", started.elapsed().as_secs_f64());

    let mut produced = Vec::new();
    let mut streamed = tokenizer.decoder();
    for _ in 0..tokens {
        let next = top(&logits);
        if next == manifest::EOS_TOKEN_ID {
            break;
        }
        produced.push(next);
        logits = model.forward(&[next], &mut cache, &mut NoTrace);
        print!("{}", streamed.feed(next));
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    println!("new tokens: {}", produced.len());
    println!("ids: {produced:?}");
}

/// A binary PPM: `P6`, the two sides, the maximum value, then three bytes per
/// pixel. Comments (`#` to the end of the line) may appear between the fields.
fn read_ppm(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let mut fields = Vec::new();
    let mut index = 0;
    while fields.len() < 4 {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b'#' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        let start = index;
        while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if start == index {
            return Err("the header ends early".to_string());
        }
        fields.push(String::from_utf8_lossy(&bytes[start..index]).to_string());
    }
    if fields[0] != "P6" {
        return Err(format!("{} is not a binary PPM", fields[0]));
    }
    let width: usize = fields[1].parse().map_err(|_| "width".to_string())?;
    let height: usize = fields[2].parse().map_err(|_| "height".to_string())?;
    let maximum: usize = fields[3].parse().map_err(|_| "maximum".to_string())?;
    if maximum != 255 {
        return Err(format!("the maximum is {maximum}, not 255"));
    }
    // Exactly one whitespace byte separates the header from the samples.
    index += 1;
    let expected = width * height * 3;
    let pixels = bytes
        .get(index..index + expected)
        .ok_or_else(|| format!("{} bytes for {width}x{height}", bytes.len() - index))?;
    Ok((pixels.to_vec(), height, width))
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

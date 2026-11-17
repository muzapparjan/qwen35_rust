//! Generate a reply, greedily, and print it.
//!
//! ```text
//! cargo run --release --example say_text -- "用一句话介绍你自己。"
//! cargo run --release --example say_text -- --tokens 16 "Hello"
//! ```
//!
//! This is the CPU path end to end: chat template, tokenizer, 24 layers with a
//! cache, and the tied output head. It is not fast — the quantised matrix-vector
//! product is still the portable one — but it is the same arithmetic the
//! reference implementation performs.

use qwen35::chat::{self, Message, Options};
use qwen35::model::{Cache, Model, NoTrace};
use qwen35::tokenizer::tokenizer;
use qwen35::weights::manifest;

fn main() {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut tokens = 16usize;
    if let Some(position) = arguments.iter().position(|value| value == "--tokens") {
        arguments.remove(position);
        tokens = arguments
            .remove(position)
            .parse()
            .expect("--tokens needs a number");
    }
    let prompt = if arguments.is_empty() {
        "用一句话介绍你自己。".to_string()
    } else {
        arguments.join(" ")
    };

    let tokenizer = tokenizer();
    let rendered = chat::render(
        &[
            Message::system("You are a helpful assistant."),
            Message::user(&prompt),
        ],
        &Options::default(),
    )
    .expect("the conversation renders");
    let mut ids = tokenizer.encode(&rendered);
    println!("prompt tokens: {}", ids.len());

    let model = Model::new();
    let mut cache = Cache::new();
    let started = std::time::Instant::now();
    let mut logits = model.forward(&ids, &mut cache, &mut NoTrace);
    println!("prefill: {:.2} s", started.elapsed().as_secs_f64());

    let mut produced = Vec::new();
    for step in 0..tokens {
        let next = top(&logits);
        // The checkpoint's `generation_config.eos_token_id` is `<|endoftext|>`
        // (248044), not `<|im_end|>` (248046). Stopping on either is a choice;
        // this one matches what `reference/run_chat.py` does by default.
        if next == manifest::EOS_TOKEN_ID {
            break;
        }
        produced.push(next);
        ids.push(next);
        let started = std::time::Instant::now();
        logits = model.forward(&[next], &mut cache, &mut NoTrace);
        if step == 0 {
            println!("first token: {:.2} s", started.elapsed().as_secs_f64());
        }
        print!("{}", tokenizer.decode(&[next]));
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    println!("new tokens: {}", produced.len());
    println!("ids: {produced:?}");
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

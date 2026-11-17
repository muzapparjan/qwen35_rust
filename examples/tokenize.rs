//! Render a chat prompt, tokenise it, and decode it back.
//!
//! ```text
//! cargo run --example tokenize -- "你好"
//! cargo run --example tokenize -- --thinking "Write a haiku about rain."
//! ```

use qwen35::chat::{self, Message, Options};
use qwen35::tokenizer::tokenizer;

fn main() {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let thinking = arguments.first().map(String::as_str) == Some("--thinking");
    if thinking {
        arguments.remove(0);
    }
    let prompt = if arguments.is_empty() {
        "用一句话介绍你自己。".to_string()
    } else {
        arguments.join(" ")
    };

    let messages = [
        Message::system("You are a helpful assistant."),
        Message::user(&prompt),
    ];
    let options = Options {
        add_generation_prompt: true,
        enable_thinking: thinking,
    };
    let rendered = chat::render(&messages, &options).expect("the conversation renders");
    let tokenizer = tokenizer();
    let ids = tokenizer.encode(&rendered);

    println!("prompt   : {prompt:?}");
    println!("thinking : {thinking}");
    println!("rendered : {rendered:?}");
    println!("tokens   : {}", ids.len());
    println!("ids      : {ids:?}");
    println!("decoded  : {:?}", tokenizer.decode(&ids));
    assert_eq!(tokenizer.decode(&ids), rendered, "the ids do not round-trip");
}

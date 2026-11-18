//! Prefill and decode timing, in the same shape as the reference benchmark.
//!
//! ```text
//! cargo run --release --example bench -- --prompt hello --tokens 32 --repeats 3
//! cargo run --release --example bench -- --threads 4
//! ```
//!
//! Prefill is one call for the whole prompt; decode is one call per token after
//! that. Both are timed with a cache reset between runs, and the reported number
//! is the median so a scheduling hiccup does not dominate.

use qwen35::model::{Cache, Model, NoTrace};
use qwen35::parallel;
use qwen35::tokenizer::tokenizer;

fn main() {
    let mut prompt = "用一句话介绍你自己。".to_string();
    let mut tokens = 16usize;
    let mut repeats = 3usize;
    let mut threads = 0usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--prompt" => prompt = arguments.next().expect("--prompt needs text"),
            "--tokens" => tokens = arguments.next().expect("--tokens needs a number").parse().expect("number"),
            "--repeats" => repeats = arguments.next().expect("--repeats needs a number").parse().expect("number"),
            "--threads" => threads = arguments.next().expect("--threads needs a number").parse().expect("number"),
            other => prompt = other.to_string(),
        }
    }
    if threads > 0 {
        // A benchmark owns the machine; the library's default budget leaves cores
        // for a host and runs the workers below normal priority.
        parallel::set_budget(parallel::Budget {
            threads,
            reserve: 0,
            priority: parallel::Priority::Normal,
        });
    } else {
        parallel::set_budget(parallel::Budget {
            reserve: 0,
            priority: parallel::Priority::Normal,
            ..parallel::Budget::default()
        });
    }

    let ids = tokenizer().encode(&prompt);
    println!("prompt tokens : {}", ids.len());
    println!("workers       : {}", parallel::threads());

    let model = Model::new();
    // One warm-up run: the first call faults in the weights' pages.
    let _ = model.forward(&[ids[0]], &mut Cache::new(), &mut NoTrace);

    let mut prefills = Vec::new();
    let mut decodes = Vec::new();
    for _ in 0..repeats {
        let mut cache = Cache::new();
        let started = std::time::Instant::now();
        let mut logits = model.forward(&ids, &mut cache, &mut NoTrace);
        prefills.push(started.elapsed().as_secs_f64());

        let mut per_token = Vec::new();
        for _ in 0..tokens {
            let next = top(&logits);
            let started = std::time::Instant::now();
            logits = model.forward(&[next], &mut cache, &mut NoTrace);
            per_token.push(started.elapsed().as_secs_f64());
        }
        per_token.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        // The first token also writes the caches' first entries; drop it.
        let kept = &per_token[1.min(per_token.len() - 1)..];
        decodes.push(kept[kept.len() / 2]);
    }
    prefills.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    decodes.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

    let prefill = prefills[prefills.len() / 2];
    let decode = decodes[decodes.len() / 2];
    println!("prefill       : {:.3} s  {:.1} tok/s", prefill, ids.len() as f64 / prefill);
    println!("decode        : {:.3} s/token  {:.2} tok/s", decode, 1.0 / decode);
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

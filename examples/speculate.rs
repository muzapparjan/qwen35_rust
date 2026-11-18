//! Speculative decoding with the checkpoint's draft head, against plain greedy.
//!
//! ```text
//! cargo run --release --example speculate -- --tokens 48 --drafts 3
//! ```
//!
//! Two things are measured, and they are separate questions:
//!
//! - whether the draft head is wired the way the checkpoint was trained: it has
//!   to predict the token the target itself picks, which is what the acceptance
//!   counts report;
//! - whether the round is worth its cost, which is the wall-clock comparison.
//!
//! The tokens both paths produce must be identical: acceptance only ever keeps a
//! draft the target would have chosen, and the token that follows a rejected
//! tail is the target's own argmax, so speculation changes the cost and not the
//! output.

use qwen35::model::{Cache, Model, NoTrace};
use qwen35::parallel;
use qwen35::weights::manifest;

fn main() {
    let mut prompt = "用一句话介绍你自己。".to_string();
    let mut tokens = 48usize;
    let mut drafts = 3usize;
    let mut threads = 0usize;
    let mut polite = true;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--prompt" => prompt = arguments.next().expect("--prompt needs text"),
            "--tokens" => tokens = arguments.next().expect("--tokens needs a number").parse().expect("number"),
            "--drafts" => drafts = arguments.next().expect("--drafts needs a number").parse().expect("number"),
            "--threads" => threads = arguments.next().expect("--threads needs a number").parse().expect("number"),
            "--background" => polite = true,
            "--foreground" => polite = false,
            other => prompt = other.to_string(),
        }
    }
    // The library's default budget leaves two cores and runs below normal
    // priority, which is what an embedded host wants. A timing measurement has
    // to ask for the machine or it measures the host's load instead.
    parallel::set_budget(parallel::Budget {
        threads,
        reserve: 0,
        priority: if polite {
            parallel::Priority::Background
        } else {
            parallel::Priority::Normal
        },
    });

    let ids = qwen35::tokenizer::tokenizer().encode(&prompt);
    if ids.is_empty() {
        eprintln!("the prompt encodes to no tokens");
        std::process::exit(2);
    }
    if tokens == 0 {
        eprintln!("--tokens needs a positive number of decode steps");
        std::process::exit(2);
    }
    if drafts == 0 {
        eprintln!("--drafts needs at least one proposal");
        std::process::exit(2);
    }
    println!("prompt tokens : {}", ids.len());
    println!(
        "workers       : {} at {:?} priority",
        parallel::threads(),
        parallel::budget().priority
    );
    println!("draft steps   : {drafts}");

    let model = Model::new();
    let _ = model.forward(&[ids[0]], &mut Cache::new(), &mut NoTrace);

    // --- the plain path
    let mut cache = Cache::new();
    let mut logits = model.forward(&ids, &mut cache, &mut NoTrace);
    let mut plain = Vec::with_capacity(tokens);
    let started = std::time::Instant::now();
    for _ in 0..tokens {
        let next = top(&logits);
        plain.push(next);
        logits = model.forward(&[next], &mut cache, &mut NoTrace);
    }
    let plain_seconds = started.elapsed().as_secs_f64();

    // --- the speculative path
    let mut cache = Cache::new();
    let outputs = model.forward_states(&ids, &mut cache, &mut NoTrace);
    let width = manifest::HIDDEN_SIZE;
    let last = outputs.hidden.len() - width;
    let mut logits = outputs.logits;
    let mut hidden = outputs.hidden[last..].to_vec();
    model.prime_draft(&mut cache, &ids, &outputs.hidden);

    let mut speculative = Vec::with_capacity(tokens);
    let mut rounds = 0usize;
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    let mut rounds_with_a_match = 0usize;
    let mut replays = 0usize;
    let started = std::time::Instant::now();
    while speculative.len() < tokens {
        let round = model.speculative_round(&mut cache, &logits, &hidden, drafts);
        rounds += 1;
        proposed += round.drafted;
        accepted += round.accepted;
        if round.accepted > 0 {
            rounds_with_a_match += 1;
        }
        if round.replayed {
            replays += 1;
        }
        speculative.extend_from_slice(&round.tokens);
        logits = round.logits;
        hidden = round.hidden;
    }
    let speculative_seconds = started.elapsed().as_secs_f64();
    speculative.truncate(tokens);

    println!();
    println!("plain        : {plain_seconds:.3} s  {:.1} ms/token", 1000.0 * plain_seconds / tokens as f64);
    println!(
        "speculative  : {speculative_seconds:.3} s  {:.1} ms/token",
        1000.0 * speculative_seconds / speculative.len() as f64
    );
    println!("speedup      : {:.2}x", plain_seconds / speculative_seconds);
    println!();
    println!("rounds       : {rounds}");
    println!("per round    : {:.1} ms", 1000.0 * speculative_seconds / rounds as f64);
    println!("draft accept : {accepted}/{proposed} = {:.3}", accepted as f64 / proposed as f64);
    println!("rounds lvl 1 : {rounds_with_a_match}/{rounds} = {:.3}", rounds_with_a_match as f64 / rounds as f64);
    println!("tokens/round : {:.2}", speculative.len() as f64 / rounds as f64);
    println!("replayed     : {replays}/{rounds}");

    let same = plain.iter().zip(speculative.iter()).all(|(a, b)| a == b);
    println!();
    println!("tokens equal : {same}");
    if !same {
        let differ = plain.iter().zip(speculative.iter()).position(|(a, b)| a != b);
        println!("first difference at {differ:?}");
        std::process::exit(1);
    }

    // --- what a verification pass costs on its own: the round only pays off
    // when a batch of candidates is cheaper than the same tokens one at a time.
    let mut one = Cache::new();
    let _ = model.forward(&ids, &mut one, &mut NoTrace);
    let started = std::time::Instant::now();
    let _ = model.forward(&plain[..1], &mut one, &mut NoTrace);
    let single_ms = 1000.0 * started.elapsed().as_secs_f64();

    let mut many = Cache::new();
    let _ = model.forward_states(&ids, &mut many, &mut NoTrace);
    let span = (drafts + 1).min(plain.len());
    let candidates: Vec<u32> = plain[..span].to_vec();
    let started = std::time::Instant::now();
    let _ = model.forward_states(&candidates, &mut many, &mut NoTrace);
    let batch_ms = 1000.0 * started.elapsed().as_secs_f64();

    println!();
    println!("pass cost    : 1 token {single_ms:.1} ms, {} tokens {batch_ms:.1} ms", candidates.len());

    // --- the round's parts, timed on their own
    let steps = ids.len().saturating_sub(1);
    let mut primed = Cache::new();
    let outputs = model.forward_states(&ids, &mut primed, &mut NoTrace);
    let started = std::time::Instant::now();
    model.prime_draft(&mut primed, &ids, &outputs.hidden);
    let draft_step_ms = 1000.0 * started.elapsed().as_secs_f64() / steps as f64;

    let started = std::time::Instant::now();
    let mut projections = vec![0.0f32; drafts * manifest::VOCAB_SIZE];
    model.output_logits_batch(&outputs.hidden[..drafts * manifest::HIDDEN_SIZE], drafts, &mut projections);
    let head_batch_ms = 1000.0 * started.elapsed().as_secs_f64();

    let mut single = vec![0.0f32; manifest::VOCAB_SIZE];
    let started = std::time::Instant::now();
    model.output_logits(&outputs.hidden[..manifest::HIDDEN_SIZE], &mut single);
    let head_single_ms = 1000.0 * started.elapsed().as_secs_f64();

    println!();
    println!("draft step   : {draft_step_ms:.2} ms");
    println!("head, 1 row  : {head_single_ms:.2} ms");
    println!("head, {drafts} rows  : {head_batch_ms:.2} ms");

    // --- how the batch path scales: a round only pays when k + 1 tokens cost
    // less than k + 1 separate passes, and that is a question about this kernel.
    println!();
    println!("tokens  model   head   model/token");
    for rows in [1usize, 2, 3, 4, 6, 8] {
        let mut cache = Cache::new();
        let _ = model.forward_states(&ids, &mut cache, &mut NoTrace);
        let batch: Vec<u32> = plain.iter().copied().cycle().take(rows).collect();
        let started = std::time::Instant::now();
        let outputs = model.forward_states(&batch, &mut cache, &mut NoTrace);
        let model_ms = 1000.0 * started.elapsed().as_secs_f64();

        let mut projections = vec![0.0f32; rows * manifest::VOCAB_SIZE];
        let started = std::time::Instant::now();
        model.output_logits_batch(&outputs.hidden[..rows * manifest::HIDDEN_SIZE], rows, &mut projections);
        let head_ms = 1000.0 * started.elapsed().as_secs_f64();
        println!("{rows:6}  {model_ms:6.1}  {head_ms:5.1}   {:.1}", model_ms / rows as f64);
    }
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

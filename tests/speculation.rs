//! Speculation has to change the cost and nothing else.
//!
//! The head's weights are in the container but the reference implementation never
//! loads them, so there is no dump to compare against. What can be pinned is the
//! property the round exists for: every token it returns is the token greedy
//! decoding would have returned. A round keeps a draft only when the target's own
//! logits agree with it, and the token after a rejected tail is the target's
//! argmax, so the two paths have to agree token for token.

use qwen35::model::{Cache, Model, NoTrace};
use qwen35::weights::manifest;

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

#[test]
fn a_round_keeps_only_what_greedy_would_have_chosen() {
    let model = Model::new();
    let ids = qwen35::tokenizer::tokenizer().encode("The capital of France is");
    let tokens = 6usize;
    let steps = 2usize;

    // --- plain greedy
    let mut cache = Cache::new();
    let mut logits = model.forward(&ids, &mut cache, &mut NoTrace);
    let mut plain = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        let next = argmax(&logits);
        plain.push(next);
        logits = model.forward(&[next], &mut cache, &mut NoTrace);
    }

    // --- speculative
    let mut cache = Cache::new();
    let outputs = model.forward_states(&ids, &mut cache, &mut NoTrace);
    model.prime_draft(&mut cache, &ids, &outputs.hidden);
    let width = manifest::HIDDEN_SIZE;
    let mut logits = outputs.logits;
    let mut hidden = outputs.hidden[outputs.hidden.len() - width..].to_vec();

    let mut speculative = Vec::with_capacity(tokens);
    while speculative.len() < tokens {
        let round = model.speculative_round(&mut cache, &logits, &hidden, steps);
        assert!(!round.tokens.is_empty(), "a round returns at least one token");
        assert!(round.accepted <= round.drafted, "accepted beyond what was drafted");
        speculative.extend_from_slice(&round.tokens);
        logits = round.logits;
        hidden = round.hidden;
    }
    speculative.truncate(tokens);

    assert_eq!(speculative, plain, "speculation changed the output");
}

#[test]
fn a_rejected_tail_leaves_the_cache_as_if_it_never_ran() {
    // The linear-attention layers have no addressable history: every token
    // multiplies their recurrent state in place. A round that proposed a tail the
    // target rejected has to put that state back, and the only visible proof is
    // that a later token comes out the same as it would have without the round.
    let model = Model::new();
    let ids = qwen35::tokenizer::tokenizer().encode("Water boils at a temperature of");
    // Long enough that a rejection in the first round is followed by rounds that
    // read the cache the rollback left behind.
    let tokens = 14usize;

    let mut clean = Cache::new();
    let mut logits = model.forward(&ids, &mut clean, &mut NoTrace);
    let mut expected = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        let next = argmax(&logits);
        expected.push(next);
        logits = model.forward(&[next], &mut clean, &mut NoTrace);
    }

    // Draft four steps so the tail is likely to be rejected, then keep one more
    // token through the target alone: the cache the round left behind has to
    // produce the same continuation as a cache that never speculated.
    let mut cache = Cache::new();
    let outputs = model.forward_states(&ids, &mut cache, &mut NoTrace);
    model.prime_draft(&mut cache, &ids, &outputs.hidden);
    let width = manifest::HIDDEN_SIZE;
    let mut logits = outputs.logits;
    let mut hidden = outputs.hidden[outputs.hidden.len() - width..].to_vec();

    let mut speculative = Vec::new();
    let mut rejected = false;
    while speculative.len() < tokens {
        let round = model.speculative_round(&mut cache, &logits, &hidden, 4);
        rejected |= round.replayed;
        speculative.extend_from_slice(&round.tokens);
        logits = round.logits;
        hidden = round.hidden;
    }
    speculative.truncate(tokens);

    assert_eq!(speculative, expected, "a rollback left the cache in a different state");
    assert!(
        rejected,
        "this prompt was expected to reject a draft, which is the path under test"
    );
}

#[test]
fn a_round_leaves_the_draft_cache_holding_only_accepted_tokens() {
    // Only the next round reads the draft head's cache, so an entry for a rejected
    // proposal does not change this round's output. It becomes history a round
    // later, where the head attends to a token the target refused, so the count is
    // what has to be pinned: one entry per accepted token and none for the rest.
    let model = Model::new();
    let ids = qwen35::tokenizer::tokenizer().encode("Water boils at a temperature of");
    let tokens = 14usize;
    let steps = 4usize;

    let mut cache = Cache::new();
    let outputs = model.forward_states(&ids, &mut cache, &mut NoTrace);
    model.prime_draft(&mut cache, &ids, &outputs.hidden);
    let width = manifest::HIDDEN_SIZE;
    let mut logits = outputs.logits;
    let mut hidden = outputs.hidden[outputs.hidden.len() - width..].to_vec();

    let mut produced = 0usize;
    let mut replayed = 0usize;
    while produced < tokens {
        let before = cache.mtp.positions;
        let round = model.speculative_round(&mut cache, &logits, &hidden, steps);
        let kept = cache.mtp.positions - before;
        if round.replayed {
            replayed += 1;
            // The drafting loop wrote one entry per proposal; the accepted
            // proposals are its first `accepted + 1` ones.
            assert_eq!(
                kept,
                round.accepted + 1,
                "a rejected round kept {kept} draft entries of {} accepted tokens",
                round.accepted + 1
            );
        } else {
            assert_eq!(
                kept, steps,
                "a round that accepted every proposal kept {kept} draft entries"
            );
        }
        produced += round.tokens.len();
        logits = round.logits;
        hidden = round.hidden;
    }
    assert!(
        replayed > 0,
        "no round rejected a draft, so the rollback was not exercised"
    );
}

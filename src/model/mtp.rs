//! The multi-token prediction head, and the speculative round that uses it.
//!
//! The checkpoint ships one extra decoder layer under `mtp.*`. The reference
//! implementation never loads it — `modeling_qwen3_5.py` lists `r"^mtp.*"` in
//! `_keys_to_ignore_on_load_unexpected` — because it is a speculative-decoding
//! drafter rather than part of the text model. It is still part of the weights,
//! and the serving engines read it as:
//!
//! 1. normalise the embedding of the token it is predicting from, and normalise
//!    the target's hidden state for the position before that token;
//! 2. concatenate the two, embedding first, and project from `2 * HIDDEN_SIZE`
//!    back to `HIDDEN_SIZE` (`mtp.fc`);
//! 3. run one full-attention decoder layer (`mtp.layers.0`), which has the same
//!    operators and shapes as a full-attention text layer, and so runs the same
//!    code;
//! 4. normalise the result (`mtp.norm`) and read it through the text model's
//!    tied output head.
//!
//! The shapes alone do not prove that wiring — every weight would fit more than
//! one arrangement — so the round reports how often the head predicted the
//! token the target itself chose. A head connected the wrong way agrees about as
//! often as chance.

use crate::ops;
use crate::weights::manifest;

use super::{embedding_matrix, matrix_at, vector_at, Cache, Model, NoTrace};

/// What one speculative round produced.
pub struct Round {
    /// Tokens the target confirmed, in order. Always at least one: the round
    /// starts from the token the target's own logits select.
    pub tokens: Vec<u32>,
    /// Draft tokens proposed beyond that first one.
    pub drafted: usize,
    /// Draft tokens the target confirmed.
    pub accepted: usize,
    /// Whether the kept prefix had to be walked a second time, which a round
    /// pays when the target rejects part of the tail.
    pub replayed: bool,
    /// Logits at the last confirmed position; their argmax is the next token.
    pub logits: Vec<f32>,
    /// The hidden state of the last confirmed position, for the next round.
    pub hidden: Vec<f32>,
}

/// The largest index of `values`; ties go to the lower index.
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

impl Model {
    /// One draft step: the logits for the token after `token`, and the hidden
    /// state the next draft step consumes.
    ///
    /// `previous` is the target model's normalised hidden state for the position
    /// before `token`, and `position` is where `token` sits. The draft layer
    /// keeps its own attention cache, so one call adds one entry to it.
    fn draft(
        &self,
        previous: &[f32],
        token: u32,
        position: u32,
        cache: &mut Cache,
        logits: &mut [f32],
    ) -> Vec<f32> {
        let hidden = manifest::HIDDEN_SIZE;
        let eps = manifest::RMS_NORM_EPS;
        let embed = embedding_matrix();

        // --- the two inputs, each normalised on its own
        let mut embedding = vec![0.0f32; hidden];
        ops::row_values(&embed, token as usize, &mut embedding);
        let mut embedding_norm = vec![0.0f32; hidden];
        ops::rms_norm_zero_centered(
            &embedding,
            1,
            hidden,
            &vector_at("mtp", "pre_fc_norm_embedding.weight"),
            eps,
            &mut embedding_norm,
        );

        let mut previous_norm = vec![0.0f32; hidden];
        ops::rms_norm_zero_centered(
            previous,
            1,
            hidden,
            &vector_at("mtp", "pre_fc_norm_hidden.weight"),
            eps,
            &mut previous_norm,
        );

        // Embedding first. The projection is one matrix, so it cannot tell the
        // order; the checkpoint's own consumers all concatenate this way.
        let mut fused = Vec::with_capacity(2 * hidden);
        fused.extend_from_slice(&embedding_norm);
        fused.extend_from_slice(&previous_norm);

        let mut x = vec![0.0f32; hidden];
        ops::linear(&fused, 1, &matrix_at("mtp", "fc.weight"), None, &mut x);

        // --- one decoder layer, the same code a text layer runs
        let base = "mtp.layers.0";
        let mut normalized = vec![0.0f32; hidden];
        let mut mixed = vec![0.0f32; hidden];
        ops::rms_norm_zero_centered(
            &x,
            1,
            hidden,
            &vector_at(base, "input_layernorm.weight"),
            eps,
            &mut normalized,
        );
        let past = cache.mtp.positions;
        self.full_attention(
            base,
            base,
            &mut cache.mtp,
            &normalized,
            1,
            past,
            &[[position; 3]],
            &mut NoTrace,
            &mut mixed,
        );
        for index in 0..hidden {
            x[index] += mixed[index];
        }

        ops::rms_norm_zero_centered(
            &x,
            1,
            hidden,
            &vector_at(base, "post_attention_layernorm.weight"),
            eps,
            &mut normalized,
        );
        self.mlp(base, &normalized, 1, &mut mixed);
        for index in 0..hidden {
            x[index] += mixed[index];
        }

        // --- the head's own norm, then the text model's tied output projection
        let mut final_hidden = vec![0.0f32; hidden];
        ops::rms_norm_zero_centered(
            &x,
            1,
            hidden,
            &vector_at("mtp", "norm.weight"),
            eps,
            &mut final_hidden,
        );
        self.output_logits(&final_hidden, logits);
        final_hidden
    }

    /// Fill the draft head's cache over a prompt that was just forwarded.
    ///
    /// The draft layer attends over its own history, so a round that started
    /// from an empty cache would propose from nothing but its own last step. The
    /// prompt is consumed the way a round consumes generation: a token is paired
    /// with the target's hidden state for the position before it, so the first
    /// prompt token has no predecessor to pair with and is skipped, and the last
    /// hidden state predicts a token the loop discards.
    ///
    /// `hidden` is [`crate::model::Outputs::hidden`] for `ids`, one row per
    /// token.
    pub fn prime_draft(&self, cache: &mut Cache, ids: &[u32], hidden: &[f32]) {
        let width = manifest::HIDDEN_SIZE;
        assert_eq!(hidden.len(), ids.len() * width, "one hidden row per token");
        let first = cache.length as i64 - ids.len() as i64 + cache.position_delta;
        let first = first.max(0) as u32;
        let mut scratch = vec![0.0f32; manifest::VOCAB_SIZE];
        for index in 0..ids.len().saturating_sub(1) {
            self.draft(
                &hidden[index * width..(index + 1) * width],
                ids[index + 1],
                first + index as u32 + 1,
                cache,
                &mut scratch,
            );
        }
    }

    /// Draft, verify and keep: one speculative round.
    ///
    /// `logits` and `hidden` describe the position just before the next token,
    /// as [`Model::forward_states`] returns them. `steps` is how many draft
    /// tokens to propose beyond the one the target already determines.
    ///
    /// The tokens a round returns start with that determined token, so a round
    /// always yields at least one, and the logits and hidden it returns describe
    /// the position after the last of them.
    pub fn speculative_round(
        &self,
        cache: &mut Cache,
        logits: &[f32],
        hidden: &[f32],
        steps: usize,
    ) -> Round {
        assert!(steps >= 1, "a round drafts at least one token");
        let width = manifest::HIDDEN_SIZE;
        let first = argmax(logits);
        let start = (cache.length as i64 + cache.position_delta).max(0) as u32;

        // The state to come back to if the tail is rejected. Taken before the
        // drafting loop, because that loop advances the draft head's own cache and a
        // rollback has to undo those entries as well.
        let snapshot = cache.snapshot();

        // --- draft, each step reading the head's own previous output
        let mut drafts = Vec::with_capacity(steps + 1);
        drafts.push(first);
        let mut previous = hidden.to_vec();
        let mut proposal = vec![0.0f32; manifest::VOCAB_SIZE];
        for step in 0..steps {
            previous = self.draft(
                &previous,
                drafts[step],
                start + step as u32,
                cache,
                &mut proposal,
            );
            drafts.push(argmax(&proposal));
        }

        // --- verify every proposal in one pass
        let mut outputs = self.forward_states(&drafts, cache, &mut NoTrace);

        // Row `i` is the target's state after reading `drafts[..=i]`, so its
        // logits predict `drafts[i + 1]`. One batched pass covers the rows the
        // round compares, which is what makes the batch worth its cost: the
        // head alone is a quarter of the model's multiply-adds.
        let mut projections = vec![0.0f32; steps * manifest::VOCAB_SIZE];
        self.output_logits_batch(
            &outputs.hidden[..steps * width],
            steps,
            &mut projections,
        );
        let mut accepted = 0;
        while accepted < steps {
            let row = &projections[accepted * manifest::VOCAB_SIZE..(accepted + 1) * manifest::VOCAB_SIZE];
            if argmax(row) != drafts[accepted + 1] {
                break;
            }
            accepted += 1;
        }

        // --- a rejected tail moved the recurrent state past the part the target
        // kept, and that product cannot be undone, so the kept prefix is walked
        // again from the copy taken before the pass.
        //
        // The batch above covers the rows the comparisons read, which is one row
        // fewer than the batch has: the row after the last accepted draft was
        // never compared against anything. That row, and the row a replay
        // rewrites, are projected one at a time.
        let replayed = accepted < steps;
        let kept: Vec<f32> = if replayed {
            // Keep one entry per accepted token: the drafting loop wrote one per
            // proposal, and the accepted proposals are its first `accepted + 1`
            // ones. When every proposal is accepted the loop wrote `steps` entries
            // and the last verified token has none.
            cache.rollback(&snapshot, (accepted + 1).min(steps));
            outputs = self.forward_states(&drafts[..=accepted], cache, &mut NoTrace);
            let base = accepted * width;
            let mut kept = vec![0.0f32; manifest::VOCAB_SIZE];
            self.output_logits(&outputs.hidden[base..base + width], &mut kept);
            kept
        } else {
            let base = accepted * width;
            let mut kept = vec![0.0f32; manifest::VOCAB_SIZE];
            self.output_logits(&outputs.hidden[base..base + width], &mut kept);
            kept
        };

        let base = accepted * width;
        Round {
            tokens: drafts[..=accepted].to_vec(),
            drafted: steps,
            accepted,
            replayed,
            logits: kept,
            hidden: outputs.hidden[base..base + width].to_vec(),
        }
    }
}

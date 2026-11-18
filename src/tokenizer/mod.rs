//! The tokenizer: a byte-level BPE over tables embedded at compile time.
//!
//! The tables come from the release's `tokenizer.json` and are pre-shaped by
//! `dev/gen_tokenizer.py`, so no JSON is parsed and no file is read at run time.
//! The blob is 5.8 MB: 256 byte-to-id entries, 247,587 merges, the token bytes in
//! id order (so decoding is a concatenation) and the 33 added tokens.
//!
//! Two facts the release hides, both checked by the generator rather than
//! assumed:
//!
//! - Tokens are stored in frequency order, so the id of a single byte's symbol is
//!   not the byte (` Ġ` is 220 while its byte is 32). The 256-entry table removes
//!   every string lookup from the encoder.
//! - Merge products are *almost* `256 + rank`: the release inserts one extra
//!   token at id 104,328. Each merge therefore carries its product id, and the
//!   generator checks that those ids increase with rank, which is what lets the
//!   encoder use the id as the merge priority.
//!
//! Still to come for this milestone: NFC normalisation and the pre-tokeniser
//! regex. Until those exist, [`Tokenizer::encode_piece`] is the entry point and
//! callers must supply the pieces.

pub mod manifest;
mod split;
mod unicode;
pub mod unicode_manifest;

pub use split::pieces;
pub use unicode::normalize;

use crate::blob::{u32_at, Aligned};
use std::sync::OnceLock;

static BLOB: Aligned<{ manifest::FILE_LEN }> = Aligned(*include_bytes!("data/tokenizer.bin"));

/// The blob, byte for byte.
pub fn blob() -> &'static [u8] {
    &BLOB.0
}

const _: () = assert!(manifest::BASE_TOKENS == 256);
const _: () = assert!(manifest::VOCAB_COUNT >= manifest::BASE_TOKENS);
const _: () = assert!(manifest::MERGE_COUNT > 0);
const _: () = assert!(manifest::ADDED_COUNT > 0);
const _: () = assert!(manifest::TOKENIZER_SIZE >= manifest::VOCAB_COUNT);
const _: () = assert!(cfg!(target_endian = "little"), "the blob is little endian");

/// One merge: the pair and the id it produces.
#[derive(Clone, Copy, Debug)]
struct Merge {
    left: u32,
    right: u32,
    merged: u32,
}

/// One token that is matched in raw text before tokenisation.
#[derive(Clone, Copy, Debug)]
struct Added {
    id: u32,
    content: &'static str,
    flags: u32,
}

impl Added {
    fn is_special(&self) -> bool {
        self.flags & manifest::FLAG_SPECIAL != 0
    }
}

/// A stretch of input: raw text, or an added token that is not tokenised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment<'a> {
    Text(&'a str),
    Added(u32),
}

/// The tokenizer's tables, parsed once from the embedded blob.
pub struct Tokenizer {
    byte_to_id: [u32; 256],
    merges: Vec<Merge>,
    offsets: Vec<u32>,
    pool: &'static [u8],
    added: Vec<Added>,
}

static INSTANCE: OnceLock<Tokenizer> = OnceLock::new();

/// The shared tokenizer, parsed on first use.
pub fn tokenizer() -> &'static Tokenizer {
    INSTANCE.get_or_init(Tokenizer::parse)
}

impl Tokenizer {
    fn parse() -> Tokenizer {
        let bytes = blob();
        assert_eq!(&bytes[..4], b"QWBT", "the tokenizer blob has no magic");
        assert_eq!(u32_at(bytes, 4), manifest::VERSION, "tokenizer blob version");

        let mut byte_to_id = [0u32; 256];
        for (byte, slot) in byte_to_id.iter_mut().enumerate() {
            *slot = u32_at(bytes, manifest::BYTE_TO_ID + 4 * byte);
        }
        let mut sorted = byte_to_id;
        sorted.sort_unstable();
        assert_eq!(sorted, std::array::from_fn(|index| index as u32), "the byte table is not a permutation");

        let mut merges = Vec::with_capacity(manifest::MERGE_COUNT);
        for index in 0..manifest::MERGE_COUNT {
            let base = manifest::MERGES + 12 * index;
            merges.push(Merge {
                left: u32_at(bytes, base),
                right: u32_at(bytes, base + 4),
                merged: u32_at(bytes, base + 8),
            });
        }
        assert!(
            merges.windows(2).all(|pair| (pair[0].left, pair[0].right) < (pair[1].left, pair[1].right)),
            "merges are not sorted by pair"
        );

        let mut offsets = Vec::with_capacity(manifest::VOCAB_COUNT + 1);
        for index in 0..=manifest::VOCAB_COUNT {
            offsets.push(u32_at(bytes, manifest::VOCAB_OFFSETS + 4 * index));
        }
        assert!(
            offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "the vocabulary offsets are not monotonic"
        );
        assert!(
            manifest::VOCAB_POOL + offsets[manifest::VOCAB_COUNT] as usize <= bytes.len(),
            "the vocabulary pool runs past the blob"
        );
        let pool = &bytes[manifest::VOCAB_POOL..manifest::VOCAB_POOL
            + offsets[manifest::VOCAB_COUNT] as usize];

        let mut added = Vec::with_capacity(manifest::ADDED_COUNT);
        for index in 0..manifest::ADDED_COUNT {
            let base = manifest::ADDED + 16 * index;
            let id = u32_at(bytes, base);
            let start = u32_at(bytes, base + 4) as usize;
            let length = u32_at(bytes, base + 8) as usize;
            let flags = u32_at(bytes, base + 12);
            let content = std::str::from_utf8(
                &bytes[manifest::ADDED_POOL + start..manifest::ADDED_POOL + start + length],
            )
            .expect("an added token is not valid UTF-8");
            added.push(Added { id, content, flags });
        }
        assert!(
            added.iter().all(|token| token.flags & manifest::FLAG_NORMALIZED == 0),
            "a normalised added token needs the normalisation pass this build does not have yet"
        );
        assert!(
            added.windows(2).all(|pair| pair[0].id < pair[1].id),
            "added tokens are not sorted by id"
        );

        Tokenizer {
            byte_to_id,
            merges,
            offsets,
            pool,
            added,
        }
    }

    /// The id a pair merges to, if it merges at all.
    fn merge_of(&self, left: u32, right: u32) -> Option<u32> {
        let mut low = 0usize;
        let mut high = self.merges.len();
        while low < high {
            let middle = (low + high) / 2;
            let merge = self.merges[middle];
            match (merge.left, merge.right).cmp(&(left, right)) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return Some(merge.merged),
            }
        }
        None
    }

    /// The raw bytes a token stands for, if the tokenizer knows the id.
    pub fn token_bytes(&self, id: u32) -> Option<&'static [u8]> {
        if (id as usize) < manifest::VOCAB_COUNT {
            let start = self.offsets[id as usize] as usize;
            let end = self.offsets[id as usize + 1] as usize;
            return Some(&self.pool[start..end]);
        }
        self.added
            .binary_search_by_key(&id, |token| token.id)
            .ok()
            .map(|index| self.added[index].content.as_bytes())
    }

    /// The id of an added token, if `text` starts with one.
    fn added_at(&self, text: &str) -> Option<u32> {
        self.added
            .iter()
            .filter(|token| text.starts_with(token.content))
            .max_by_key(|token| token.content.len())
            .map(|token| token.id)
    }

    /// Split `text` into added tokens and the text between them.
    ///
    /// Added tokens are matched in the raw text, before normalisation, which is
    /// what the release's `normalized: false` means for all 33 of them. Four of
    /// the 33 are marked special; the rest (`<tool_call>`, `<think>`, …) are
    /// matched the same way, so the flag only matters to callers that want to
    /// re-render text.
    pub fn split_added<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut start = 0usize;
        let mut index = 0usize;
        while index < text.len() {
            // Only try at character boundaries: an added token is ASCII.
            if let Some(id) = self.added_at(&text[index..]) {
                if start < index {
                    segments.push(Segment::Text(&text[start..index]));
                }
                segments.push(Segment::Added(id));
                index += self.added_content(id).len();
                start = index;
                continue;
            }
            index += text[index..].chars().next().map(char::len_utf8).unwrap_or(1);
        }
        if start < text.len() {
            segments.push(Segment::Text(&text[start..]));
        }
        segments
    }

    fn added_content(&self, id: u32) -> &'static str {
        self.added
            .binary_search_by_key(&id, |token| token.id)
            .map(|index| self.added[index].content)
            .expect("the id came from the added table")
    }

    /// Whether `id` is one of the added tokens.
    pub fn is_added(&self, id: u32) -> bool {
        self.added.binary_search_by_key(&id, |token| token.id).is_ok()
    }

    /// Whether `id` is an added token the release marks special.
    pub fn is_special(&self, id: u32) -> bool {
        self.added
            .binary_search_by_key(&id, |token| token.id)
            .map(|index| self.added[index].is_special())
            .unwrap_or(false)
    }

    /// Tokenise one pre-tokenised piece.
    ///
    /// A piece is already split the way the release's pre-tokeniser splits it;
    /// the bytes are mapped to their symbols' ids, then adjacent pairs are merged
    /// lowest-id-first, leftmost-first, until nothing merges. Merging one pair
    /// per pass and rescanning is what the release does (its priority queue pops
    /// one pair at a time), and a piece is short enough that the rescan is free.
    pub fn encode_piece(&self, piece: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> = piece
            .as_bytes()
            .iter()
            .map(|byte| self.byte_to_id[*byte as usize])
            .collect();
        while symbols.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..symbols.len() - 1 {
                if let Some(merged) = self.merge_of(symbols[index], symbols[index + 1]) {
                    match best {
                        Some((current, _)) if current <= merged => {}
                        _ => best = Some((merged, index)),
                    }
                }
            }
            let Some((merged, index)) = best else {
                break;
            };
            symbols[index] = merged;
            symbols.remove(index + 1);
        }
        symbols
    }

    /// Tokenise a run of pieces in order.
    pub fn encode_pieces<'a>(&self, pieces: impl IntoIterator<Item = &'a str>) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in pieces {
            ids.extend(self.encode_piece(piece));
        }
        ids
    }

    /// Tokenise `text` the way the release does.
    ///
    /// Added tokens are cut out of the raw text first, so their bytes are never
    /// normalised or split; each remaining stretch is normalised to NFC, split by
    /// the pre-tokeniser and encoded piece by piece.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for segment in self.split_added(text) {
            match segment {
                Segment::Added(id) => ids.push(id),
                Segment::Text(rest) => {
                    let normalized = unicode::normalize(rest);
                    for piece in split::pieces(&normalized) {
                        ids.extend(self.encode_piece(piece));
                    }
                }
            }
        }
        ids
    }

    /// The text a sequence of ids stands for.
    ///
    /// Ids the table does not know (the embedding table is padded to 248,320)
    /// contribute nothing.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(token) = self.token_bytes(*id) {
                bytes.extend_from_slice(token);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The added tokens, in id order.
    pub fn added_tokens(&self) -> impl Iterator<Item = (u32, &'static str, bool)> + '_ {
        self.added
            .iter()
            .map(|token| (token.id, token.content, token.is_special()))
    }

    /// A decoder for ids that arrive one at a time.
    ///
    /// [`Tokenizer::decode`] turns a whole sequence into text. A caller that
    /// writes each token as it is produced needs this instead: a character's bytes
    /// can be spread over several ids — nine of ten in `🎉🎉🎉 emoji` end in the
    /// middle of one — and decoding such an id alone yields `U+FFFD`.
    pub fn decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            pending: Vec::new(),
            out: String::new(),
        }
    }
}

/// Text from ids that arrive one at a time.
///
/// The unfinished tail of a character is held until the id that completes it, so
/// what comes out is only whole characters. Concatenating what a sequence of
/// [`IncrementalDecoder::feed`] calls returned is the same string as
/// [`Tokenizer::decode`] of the whole sequence, including the replacement
/// characters a genuinely broken byte sequence produces.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    /// Bytes of a character that has not ended yet; never more than three.
    pending: Vec<u8>,
    /// What the last call completed, so a caller can write it without allocating.
    out: String,
}

impl IncrementalDecoder<'static> {
    /// A decoder over [`tokenizer`], the shared table.
    pub fn new() -> IncrementalDecoder<'static> {
        tokenizer().decoder()
    }
}

impl Default for IncrementalDecoder<'static> {
    fn default() -> IncrementalDecoder<'static> {
        IncrementalDecoder::new()
    }
}

impl<'a> IncrementalDecoder<'a> {
    /// Add one id, and take the text that is now complete.
    pub fn feed(&mut self, id: u32) -> &str {
        self.out.clear();
        if let Some(bytes) = self.tokenizer.token_bytes(id) {
            self.pending.extend_from_slice(bytes);
        }
        self.drain();
        &self.out
    }

    /// Add several ids, and take the text that is now complete.
    pub fn feed_all(&mut self, ids: &[u32]) -> &str {
        self.out.clear();
        for id in ids {
            if let Some(bytes) = self.tokenizer.token_bytes(*id) {
                self.pending.extend_from_slice(bytes);
            }
        }
        self.drain();
        &self.out
    }

    /// The tail a sequence ended in the middle of, as replacement characters.
    ///
    /// Only a model that stopped mid-character leaves anything here; a finished
    /// text has nothing pending.
    pub fn finish(&mut self) -> &str {
        self.out.clear();
        if !self.pending.is_empty() {
            self.out.push_str(&String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
        &self.out
    }

    /// How many bytes of an unfinished character are held.
    pub fn pending_bytes(&self) -> usize {
        self.pending.len()
    }

    fn drain(&mut self) {
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    self.out.push_str(text);
                    self.pending.clear();
                    return;
                }
                Err(error) => match error.error_len() {
                    // A byte that cannot start or continue any character: the same
                    // replacement `decode` would produce for it.
                    Some(length) => {
                        let at = error.valid_up_to();
                        if let Ok(text) = std::str::from_utf8(&self.pending[..at]) {
                            self.out.push_str(text);
                        }
                        self.out.push('\u{FFFD}');
                        self.pending.drain(..at + length);
                    }
                    // A character that has not ended yet: keep it and wait.
                    None => {
                        let at = error.valid_up_to();
                        if let Ok(text) = std::str::from_utf8(&self.pending[..at]) {
                            self.out.push_str(text);
                        }
                        self.pending.drain(..at);
                        return;
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blob_is_a_tokenizer_table() {
        assert_eq!(&blob()[..4], b"QWBT");
        assert_eq!(blob().len(), manifest::FILE_LEN);
        assert_eq!(u32_at(blob(), 4), manifest::VERSION);
    }

    /// Ids that cut through characters: emoji, rare Hanzi, runes, a plane-1
    /// character, and text that is not cut at all.
    const SPLIT_CASES: &[&str] = &[
        "中文流式输出：第一个字符通常是多字节的。",
        "🎉🎉🎉 emoji",
        "龘靐齉爩 rare",
        "ᚠᚢᚦᚨᚱᚲ runes",
        "𠜎𠜱𠝹𠱓 extension B",
        "नमस्ते दुनिया",
        "Здравствуйте, мир",
        "a",
        "",
    ];

    #[test]
    fn feeding_ids_one_at_a_time_is_the_whole_decode() {
        let tokenizer = tokenizer();
        for text in SPLIT_CASES {
            let ids = tokenizer.encode(text);
            let mut decoder = tokenizer.decoder();
            let mut streamed = String::new();
            for id in &ids {
                streamed.push_str(decoder.feed(*id));
                assert!(
                    !streamed.contains('\u{FFFD}'),
                    "{text:?}: a character was written before it ended"
                );
            }
            streamed.push_str(decoder.finish());
            assert_eq!(streamed, tokenizer.decode(&ids), "{text:?}");
            assert_eq!(streamed, *text, "{text:?}");
            assert_eq!(decoder.pending_bytes(), 0, "{text:?} ended mid-character");
        }
    }

    #[test]
    fn the_whole_sequence_at_once_is_the_same_text() {
        let tokenizer = tokenizer();
        let ids: Vec<u32> = SPLIT_CASES
            .iter()
            .flat_map(|text| tokenizer.encode(text))
            .collect();
        let mut decoder = tokenizer.decoder();
        let at_once = decoder.feed_all(&ids).to_string();
        assert_eq!(at_once, tokenizer.decode(&ids));
        assert_eq!(decoder.pending_bytes(), 0, "a finished text holds a tail");
        assert_eq!(decoder.finish(), "", "nothing is held after a finished text");
        assert_eq!(decoder.feed_all(&[]), "", "an empty sequence is no text");
    }

    #[test]
    fn a_tail_that_never_completes_is_reported_as_a_replacement_character() {
        let tokenizer = tokenizer();
        // The first id of `🎉`, on its own: it starts a four-byte character that
        // nothing completes.
        let ids = tokenizer.encode("🎉");
        let mut decoder = tokenizer.decoder();
        let first = decoder.feed(ids[0]);
        assert_eq!(first, "", "a partial character is text only once it ends");
        assert!(decoder.pending_bytes() > 0);
        assert_eq!(decoder.finish(), "\u{FFFD}");
        assert_eq!(decoder.pending_bytes(), 0);
    }

    #[test]
    fn the_tables_cover_the_release() {
        assert_eq!(manifest::BASE_TOKENS, 256);
        assert_eq!(manifest::VOCAB_COUNT, 248_044);
        assert_eq!(manifest::MERGE_COUNT, 247_587);
        assert_eq!(manifest::ADDED_COUNT, 33);
        assert_eq!(manifest::TOKENIZER_SIZE, 248_077);
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.merges.len(), manifest::MERGE_COUNT);
        assert_eq!(tokenizer.added.len(), manifest::ADDED_COUNT);
    }

    #[test]
    fn every_byte_has_its_own_id() {
        let tokenizer = tokenizer();
        let mut seen: Vec<u32> = tokenizer.byte_to_id.to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..256).collect::<Vec<u32>>());
        // A space is not token 32: the release stores tokens in frequency order.
        assert_eq!(tokenizer.byte_to_id[b' ' as usize], 220);
        assert_eq!(tokenizer.byte_to_id[b'a' as usize], 64);
        assert_eq!(tokenizer.byte_to_id[b'A' as usize], 32);
    }

    #[test]
    fn single_pieces_match_the_reference_tokenizer() {
        let tokenizer = tokenizer();
        // Each string is one pre-tokenised piece, so these ids are also what
        // `AutoTokenizer.encode` returns for the whole string.
        let cases: [(&str, &[u32]); 8] = [
            ("Hello", &[9419]),
            (" world", &[1814]),
            ("world", &[14194]),
            ("!!!", &[11726]),
            ("  ", &[256]),
            ("你好", &[109266]),
            (" the", &[279]),
            ("don", &[14572]),
        ];
        for (text, expected) in cases {
            assert_eq!(tokenizer.encode_piece(text), expected, "{text:?}");
        }
    }

    #[test]
    fn pieces_concatenate_to_the_reference_ids() {
        let tokenizer = tokenizer();
        // `Hello, world!` splits as ["Hello", ",", " world", "!"].
        let ids = tokenizer.encode_pieces(["Hello", ",", " world", "!"]);
        assert_eq!(ids, [9419, 11, 1814, 0]);
        let ids = tokenizer.encode_pieces(["def", " f", "(x", "):\n", "   ", " return", " x", " +", " ", "1", "\n"]);
        assert_eq!(ids, [727, 281, 2007, 1590, 198, 262, 460, 830, 478, 220, 16, 198]);
    }

    #[test]
    fn decoding_returns_the_text_the_ids_came_from() {
        let tokenizer = tokenizer();
        for text in ["Hello, world!", "你好", "don't 3.14", "def f(x):\n    return x + 1\n", "🧠"] {
            let ids = tokenizer.encode_piece(text);
            assert_eq!(tokenizer.decode(&ids), text, "{text:?}");
        }
    }

    #[test]
    fn added_tokens_are_split_out_of_raw_text() {
        let tokenizer = tokenizer();
        let segments = tokenizer.split_added("hi<|im_end|>there");
        assert_eq!(
            segments,
            [
                Segment::Text("hi"),
                Segment::Added(manifest::IM_END),
                Segment::Text("there")
            ]
        );
        let ids = tokenizer.encode("hi<|im_end|>there");
        assert_eq!(ids[0], 5834, "hi");
        assert_eq!(ids[1], manifest::IM_END);
        assert_eq!(tokenizer.decode(&ids), "hi<|im_end|>there");
        assert!(tokenizer.is_special(manifest::IM_END));
        assert!(!tokenizer.is_special(manifest::THINK_START));
        assert!(tokenizer.is_added(manifest::THINK_START));
    }

    #[test]
    fn a_piece_split_by_an_added_token_keeps_the_bytes_around_it() {
        let tokenizer = tokenizer();
        let segments = tokenizer.split_added("a<tool_call>b");
        assert_eq!(
            segments,
            [
                Segment::Text("a"),
                Segment::Added(manifest::TOOL_CALL_START),
                Segment::Text("b")
            ]
        );
        // Adjacent added tokens produce no empty text segments.
        let segments = tokenizer.split_added("<think></think>");
        assert_eq!(
            segments,
            [
                Segment::Added(manifest::THINK_START),
                Segment::Added(manifest::THINK_END)
            ]
        );
    }
}

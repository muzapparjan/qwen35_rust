//! Unicode tables the tokenizer needs: NFC, and the letter/mark/number classes
//! its pre-tokeniser matches.
//!
//! The tables are measured against the tokenizer this crate is compared with,
//! not copied from a Unicode version: character classes were probed by running
//! the reference's own regex over every code point, and combining classes were
//! measured by ordering marks against each other with its normaliser. That
//! matters because the two disagreed in both directions — Python's
//! `unicodedata` is Unicode 14.0.0, and the reference knows marks it does not
//! while treating some of its marks as starters.
//!
//! Only *atomic* marks carry a combining class. Decomposition happens before
//! ordering, so a mark with a canonical decomposition is gone by the time its
//! own class could be looked at.

use crate::blob::{u32_at, Aligned};

use super::unicode_manifest as tables;

static BLOB: Aligned<{ tables::FILE_LEN }> = Aligned(*include_bytes!("data/unicode.bin"));

/// Letter, mark or number, as the pre-tokeniser's `\p{L}`, `\p{M}` and `\p{N}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Other,
    Letter,
    Mark,
    Number,
}

const HANGUL_FIRST: u32 = 0xAC00;
const HANGUL_LAST: u32 = 0xD7A3;
const HANGUL_VOWEL_COUNT: u32 = 21;
const HANGUL_TRAIL_COUNT: u32 = 28;

fn blob() -> &'static [u8] {
    &BLOB.0
}

const _: () = assert!(tables::CLASS_COUNT > 0);
const _: () = assert!(tables::COMPOSE_COUNT > 0);
const _: () = assert!(cfg!(target_endian = "little"), "the tables are little endian");

/// The class of `character`.
pub fn class_of(character: char) -> Class {
    let codepoint = character as u32;
    let bytes = blob();
    let (mut low, mut high) = (0usize, tables::CLASS_COUNT);
    while low < high {
        let middle = (low + high) / 2;
        let base = tables::CLASSES + 12 * middle;
        let start = u32_at(bytes, base);
        let end = u32_at(bytes, base + 4);
        if codepoint < start {
            high = middle;
        } else if codepoint > end {
            low = middle + 1;
        } else {
            return match u32_at(bytes, base + 8) {
                1 => Class::Letter,
                2 => Class::Mark,
                3 => Class::Number,
                _ => Class::Other,
            };
        }
    }
    Class::Other
}

/// Whether `character` is Unicode white space, which is what `\s` matches.
///
/// Written out rather than derived: `char::is_whitespace` in Rust follows
/// `White_Space`, and the reference agrees with it, but the list is short enough
/// that stating it removes a dependency on a standard library version.
pub fn is_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000D}'
            | '\u{0020}'
            | '\u{0085}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

/// The canonical combining class of `character`, or 0.
pub fn combining(character: char) -> u8 {
    let bytes = blob();
    let (mut low, mut high) = (0usize, tables::CCC_COUNT);
    while low < high {
        let middle = (low + high) / 2;
        let base = tables::CCC + 8 * middle;
        let codepoint = u32_at(bytes, base);
        match (character as u32).cmp(&codepoint) {
            std::cmp::Ordering::Less => high = middle,
            std::cmp::Ordering::Greater => low = middle + 1,
            std::cmp::Ordering::Equal => return u32_at(bytes, base + 4) as u8,
        }
    }
    0
}

/// The stored canonical decomposition of `codepoint`, already fully expanded.
fn decomposition(codepoint: u32) -> Option<&'static [u32]> {
    let bytes = blob();
    let (mut low, mut high) = (0usize, tables::DECOMP_COUNT);
    while low < high {
        let middle = (low + high) / 2;
        let base = tables::DECOMPS + 12 * middle;
        let key = u32_at(bytes, base);
        match codepoint.cmp(&key) {
            std::cmp::Ordering::Less => high = middle,
            std::cmp::Ordering::Greater => low = middle + 1,
            std::cmp::Ordering::Equal => {
                let offset = u32_at(bytes, base + 4) as usize;
                let length = u32_at(bytes, base + 8) as usize;
                let start = tables::POOL + 4 * offset;
                // SAFETY: the pool begins at a multiple of four bytes within a
                // 64-byte-aligned buffer, and every entry is a whole number of
                // `u32`, so the pointer is at least four-byte aligned; the
                // generator checked `length` against the pool's own length when it
                // wrote the file.
                return Some(unsafe {
                    std::slice::from_raw_parts(bytes[start..].as_ptr().cast::<u32>(), length)
                });
            }
        }
    }
    None
}

fn hangul_decomposition(codepoint: u32) -> Option<[u32; 3]> {
    if !(HANGUL_FIRST..=HANGUL_LAST).contains(&codepoint) {
        return None;
    }
    let index = codepoint - HANGUL_FIRST;
    let lead = 0x1100 + index / (HANGUL_VOWEL_COUNT * HANGUL_TRAIL_COUNT);
    let vowel = 0x1161 + (index % (HANGUL_VOWEL_COUNT * HANGUL_TRAIL_COUNT)) / HANGUL_TRAIL_COUNT;
    let trail = 0x11A7 + index % HANGUL_TRAIL_COUNT;
    Some([lead, vowel, trail])
}

fn hangul_composition(first: u32, second: u32) -> Option<u32> {
    if (0x1100..=0x1112).contains(&first) && (0x1161..=0x1175).contains(&second) {
        return Some(HANGUL_FIRST + ((first - 0x1100) * HANGUL_VOWEL_COUNT + (second - 0x1161)) * HANGUL_TRAIL_COUNT);
    }
    if (HANGUL_FIRST..=HANGUL_LAST).contains(&first)
        && (0x11A8..=0x11C2).contains(&second)
        && (first - HANGUL_FIRST) % HANGUL_TRAIL_COUNT == 0
    {
        return Some(first + (second - 0x11A7));
    }
    None
}

/// The composite of `first` and `second`, if the pair composes.
fn composition(first: u32, second: u32) -> Option<u32> {
    if let Some(value) = hangul_composition(first, second) {
        return Some(value);
    }
    let bytes = blob();
    let (mut low, mut high) = (0usize, tables::COMPOSE_COUNT);
    while low < high {
        let middle = (low + high) / 2;
        let base = tables::COMPOSE + 12 * middle;
        let left = u32_at(bytes, base);
        let right = u32_at(bytes, base + 4);
        match (first, second).cmp(&(left, right)) {
            std::cmp::Ordering::Less => high = middle,
            std::cmp::Ordering::Greater => low = middle + 1,
            std::cmp::Ordering::Equal => return Some(u32_at(bytes, base + 8)),
        }
    }
    None
}

fn decompose(codepoint: u32, out: &mut Vec<u32>) {
    if let Some(jamo) = hangul_decomposition(codepoint) {
        let trail_is_present = codepoint - HANGUL_FIRST;
        let count = if trail_is_present % HANGUL_TRAIL_COUNT == 0 { 2 } else { 3 };
        out.extend_from_slice(&jamo[..count]);
        return;
    }
    match decomposition(codepoint) {
        // The stored form is the fully expanded decomposition, in canonical
        // order, so one lookup is enough.
        Some(parts) => out.extend_from_slice(parts),
        None => out.push(codepoint),
    }
}

/// Put every run of non-starters into canonical order.
///
/// The sort is stable, which is what keeps two marks of the same class in the
/// order they were written.
fn canonical_order(values: &mut [u32]) {
    let mut index = 0;
    while index < values.len() {
        if combining_from_u32(values[index]) == 0 {
            index += 1;
            continue;
        }
        let mut end = index;
        while end < values.len() && combining_from_u32(values[end]) != 0 {
            end += 1;
        }
        values[index..end].sort_by_key(|value| combining_from_u32(*value));
        index = end;
    }
}

fn combining_from_u32(value: u32) -> u8 {
    char::from_u32(value).map(combining).unwrap_or(0)
}

/// NFC: decompose, order, compose.
pub fn normalize(text: &str) -> String {
    // ASCII has no decompositions, no marks and no compositions.
    if text.is_ascii() {
        return text.to_string();
    }
    let mut values = Vec::with_capacity(text.len());
    for character in text.chars() {
        decompose(character as u32, &mut values);
    }
    canonical_order(&mut values);
    if values.is_empty() {
        return String::new();
    }

    let mut result = Vec::with_capacity(values.len());
    result.push(values[0]);
    let mut starter = if combining_from_u32(values[0]) == 0 { Some(0usize) } else { None };
    let mut last_class = combining_from_u32(values[0]);
    for value in &values[1..] {
        let class = combining_from_u32(*value);
        if let Some(position) = starter {
            if last_class < class || last_class == 0 {
                if let Some(composite) = composition(result[position], *value) {
                    result[position] = composite;
                    continue;
                }
            }
        }
        if class == 0 {
            starter = Some(result.len());
        }
        last_class = class;
        result.push(*value);
    }

    result
        .iter()
        .map(|value| char::from_u32(*value).unwrap_or('\u{FFFD}'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_file_is_a_unicode_table() {
        assert_eq!(&blob()[..4], b"QWUN");
        assert_eq!(blob().len(), tables::FILE_LEN);
    }

    #[test]
    fn letters_marks_and_numbers_are_classified() {
        assert_eq!(class_of('a'), Class::Letter);
        assert_eq!(class_of('A'), Class::Letter);
        assert_eq!(class_of('你'), Class::Letter);
        assert_eq!(class_of('\u{0301}'), Class::Mark);
        assert_eq!(class_of('5'), Class::Number);
        assert_eq!(class_of('Ⅻ'), Class::Number);
        assert_eq!(class_of(' '), Class::Other);
        assert_eq!(class_of('!'), Class::Other);
        assert_eq!(class_of('\n'), Class::Other);
    }

    #[test]
    fn white_space_matches_the_reference_set() {
        for character in ['\t', '\n', '\u{000B}', '\u{000C}', '\r', ' ', '\u{0085}', '\u{00A0}', '\u{3000}'] {
            assert!(is_whitespace(character), "{character:?}");
        }
        // Python's `str.isspace` reports these; the regex `\s` does not.
        for character in ['\u{001C}', '\u{001D}', '\u{001E}', '\u{001F}'] {
            assert!(!is_whitespace(character), "{character:?}");
        }
        assert!(!is_whitespace('a'));
        assert!(!is_whitespace('\u{200B}'));
    }

    #[test]
    fn combining_classes_come_from_the_reference() {
        assert_eq!(combining('\u{0300}'), 230);
        assert_eq!(combining('\u{0308}'), 230);
        assert_eq!(combining('\u{0334}'), 1);
        assert_eq!(combining('\u{0345}'), 240);
        assert_eq!(combining('a'), 0);
        // A mark whose class decomposes away does not need an entry.
        assert_eq!(combining('\u{0344}'), 0);
    }

    #[test]
    fn composition_matches_the_orthography() {
        assert_eq!(normalize("e\u{0301}"), "é");
        assert_eq!(normalize("E\u{0301}"), "É");
        // Lowercase u with diaeresis and macron is U+01D6; the uppercase one is U+01D5.
        assert_eq!(normalize("u\u{0308}\u{0304}"), "ǖ");
        assert_eq!(normalize("U\u{0308}\u{0304}"), "Ǖ");
        assert_eq!(normalize("A\u{030A}"), "Å");
        // Already composed input is unchanged.
        assert_eq!(normalize("é"), "é");
        assert_eq!(normalize("Ǖ"), "Ǖ");
    }

    #[test]
    fn marks_are_put_in_canonical_order() {
        // U+0334 (class 1) sorts before U+0301 (class 230), whichever order they
        // were written in, and does not block the accent from composing.
        assert_eq!(normalize("a\u{0301}\u{0334}"), "\u{00E1}\u{0334}");
        assert_eq!(normalize("a\u{0334}\u{0301}"), "\u{00E1}\u{0334}");
        // Two marks of the same class keep the order they were written in, so the
        // one written first is the one that composes.
        assert_eq!(normalize("a\u{0301}\u{0300}"), "\u{00E1}\u{0300}");
        assert_eq!(normalize("a\u{0300}\u{0301}"), "\u{00E0}\u{0301}");
    }

    #[test]
    fn hangul_is_decomposed_and_recomposed() {
        assert_eq!(normalize("\u{AC00}"), "\u{AC00}");
        assert_eq!(normalize("\u{1100}\u{1161}"), "\u{AC00}");
        assert_eq!(normalize("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}");
        assert_eq!(normalize("\u{AC01}"), "\u{AC01}");
    }

    #[test]
    fn normalising_twice_changes_nothing() {
        let cases = [
            "Hello, world!",
            "e\u{0301} vs \u{00E9}",
            "Ǡ Ǖ Ǜ ẛ ṩ",
            "한국어 텍스트",
            "\u{1100}\u{1161}\u{11A8}",
            "a\u{0301}\u{0334}\u{0300}",
            "🧠 and 你",
        ];
        for case in cases {
            let once = normalize(case);
            let twice = normalize(&once);
            assert_eq!(once, twice, "{case:?}");
        }
    }

    #[test]
    fn ascii_passes_through() {
        for text in ["", "a", "Hello, world!", "!!!", "\n\t\r", "3.14"] {
            assert_eq!(normalize(text), text);
        }
    }
}

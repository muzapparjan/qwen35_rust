//! The pre-tokeniser: the release's split regex, as a matcher.
//!
//! The pattern is
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//! |\p{N}
//! | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```
//!
//! and it is evaluated leftmost-first: at each position the alternatives are
//! tried in order and the first that matches wins. Written as a matcher rather
//! than a regex engine because this is the only pattern the crate needs, and the
//! Unicode classes come from tables that were measured against the reference.
//!
//! Two branches are worth stating plainly, because their backtracking is
//! visible in the output:
//!
//! - `\s*[\r\n]+` matches a whitespace run only when the run ends with a line
//!   break; a run that ends in spaces falls through to the later branches.
//! - `\s+(?!\S)` gives back one whitespace character, so `"   x"` splits into
//!   `"  "` and `" x"` rather than `"   "` and `"x"`. That is what the reference
//!   produces, and it is why indentation does not become one piece with the
//!   following word.

use super::unicode::{class_of, is_whitespace, Class};

/// Whether `character` is a letter or a mark.
fn is_word(character: char) -> bool {
    matches!(class_of(character), Class::Letter | Class::Mark)
}

fn is_letter(character: char) -> bool {
    class_of(character) == Class::Letter
}

fn is_number(character: char) -> bool {
    class_of(character) == Class::Number
}

fn is_line_break(character: char) -> bool {
    character == '\r' || character == '\n'
}

/// A character that `[^...]` admits, given what the class list excludes.
fn is_symbol(character: char) -> bool {
    !is_whitespace(character) && !is_word(character) && !is_number(character)
}

/// A character the optional prefix of the second branch admits.
fn is_prefix(character: char) -> bool {
    !is_line_break(character) && !is_letter(character) && !is_number(character)
}

/// The length in bytes of the contraction at the start of `text`, if any.
///
/// The alternatives are ASCII and none is a prefix of another, so the match is
/// unambiguous.
fn contraction(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'\'' {
        return None;
    }
    let following = bytes[1].to_ascii_lowercase();
    match following {
        b's' | b't' | b'm' | b'd' => Some(2),
        b'r' | b'v' => (bytes.len() >= 3 && bytes[2].to_ascii_lowercase() == b'e').then_some(3),
        b'l' => (bytes.len() >= 3 && bytes[2].to_ascii_lowercase() == b'l').then_some(3),
        _ => None,
    }
}

/// The byte length the pattern matches at the start of `text`, if it matches.
fn match_at(text: &str) -> Option<usize> {
    let first = text.chars().next()?;
    if let Some(length) = contraction(text) {
        return Some(length);
    }
    if let Some(length) = word_piece(text, first) {
        return Some(length);
    }
    if is_number(first) {
        return Some(first.len_utf8());
    }
    if let Some(length) = symbol_piece(text, first) {
        return Some(length);
    }
    whitespace_piece(text)
}

/// `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`
fn word_piece(text: &str, first: char) -> Option<usize> {
    let rest = &text[first.len_utf8()..];
    if is_prefix(first) {
        // The prefix is taken only when a word character follows it.
        if let Some(body) = run_of(rest, is_word) {
            return Some(first.len_utf8() + body);
        }
    }
    // Otherwise the first character has to start the run itself.
    if is_word(first) {
        return Some(first.len_utf8() + run_of(rest, is_word).unwrap_or(0));
    }
    None
}

/// ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
fn symbol_piece(text: &str, first: char) -> Option<usize> {
    let (prefix, body) = if first == ' ' { (1, &text[1..]) } else { (0, text) };
    let start = body.chars().next()?;
    if !is_symbol(start) {
        return None;
    }
    let symbols = run_of(body, is_symbol).unwrap_or(0);
    let breaks = run_of(&body[symbols..], is_line_break).unwrap_or(0);
    Some(prefix + symbols + breaks)
}

/// `\s*[\r\n]+`, then `\s+(?!\S)`, then `\s+`.
fn whitespace_piece(text: &str) -> Option<usize> {
    let run = run_of(text, is_whitespace)?;
    // The first branch takes the run when it ends with a line break, which is
    // the longest whitespace prefix ending in one.
    let mut end = run;
    while end > 0 {
        let last = text[..end].chars().next_back().expect("inside the string");
        if is_line_break(last) {
            return Some(end);
        }
        end -= last.len_utf8();
    }
    // `\s+(?!\S)`: the run has to reach the end of the text. Greedy, so it takes
    // the whole run, then gives one character back when there is one to give.
    if run == text.len() {
        return Some(run);
    }
    let last = text[..run].chars().next_back().expect("inside the string");
    let shorter = run - last.len_utf8();
    if shorter > 0 {
        return Some(shorter);
    }
    // `\s+`
    Some(run)
}

/// The byte length of the run of characters starting at the front of `text` that
/// satisfy `keep`, or `None` when there is none.
fn run_of(text: &str, keep: fn(char) -> bool) -> Option<usize> {
    let mut length = 0;
    for character in text.chars() {
        if !keep(character) {
            break;
        }
        length += character.len_utf8();
    }
    (length > 0).then_some(length)
}

/// Split `text` the way the release's pre-tokeniser does.
///
/// Matches and the text between them are all pieces, in order; empty pieces are
/// not produced.
pub fn pieces(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < text.len() {
        match match_at(&text[index..]) {
            Some(length) if length > 0 => {
                if start < index {
                    result.push(&text[start..index]);
                }
                result.push(&text[index..index + length]);
                index += length;
                start = index;
            }
            _ => {
                let character = text[index..].chars().next().expect("inside the string");
                index += character.len_utf8();
            }
        }
    }
    if start < text.len() {
        result.push(&text[start..]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_keep_their_leading_space() {
        assert_eq!(pieces("hello world"), ["hello", " world"]);
        assert_eq!(pieces("hello"), ["hello"]);
        assert_eq!(pieces(" the"), [" the"]);
    }

    #[test]
    fn contractions_are_single_pieces() {
        assert_eq!(pieces("don't"), ["don", "'t"]);
        assert_eq!(pieces("we're"), ["we", "'re"]);
        assert_eq!(pieces("I'll"), ["I", "'ll"]);
        assert_eq!(pieces("it's"), ["it", "'s"]);
        assert_eq!(pieces("They'RE"), ["They", "'RE"]);
    }

    #[test]
    fn digits_are_individual_pieces() {
        assert_eq!(pieces("3.14"), ["3", ".", "1", "4"]);
        assert_eq!(pieces("42"), ["4", "2"]);
    }

    #[test]
    fn punctuation_runs_stay_together_and_swallow_line_breaks() {
        assert_eq!(pieces("!!!"), ["!!!"]);
        assert_eq!(pieces("), "), ["),", " "]);
        assert_eq!(pieces("):\n"), ["):\n"]);
        assert_eq!(pieces("a!!!\n\nb"), ["a", "!!!\n\n", "b"]);
    }

    #[test]
    fn a_space_never_starts_a_punctuation_piece() {
        // The optional leading space is only taken when punctuation follows it.
        assert_eq!(pieces(" !"), [" !"]);
        assert_eq!(pieces("a !"), ["a", " !"]);
    }

    #[test]
    fn indentation_gives_one_character_back() {
        // `\s+(?!\S)` leaves the last space for the word branch, which takes it
        // together with the word.
        assert_eq!(pieces("    return"), ["   ", " return"]);
        assert_eq!(pieces("  x"), [" ", " x"]);
        assert_eq!(pieces("   "), ["   "]);
        assert_eq!(pieces("  \n"), ["  \n"]);
        assert_eq!(pieces("\n\n"), ["\n\n"]);
    }

    #[test]
    fn a_whitespace_run_ending_in_spaces_stays_whole() {
        // Two newlines then two spaces at the end: the `\s*[\r\n]+` branch takes
        // the newlines, and the spaces are a piece of their own.
        assert_eq!(pieces("\n\n  "), ["\n\n", "  "]);
    }

    #[test]
    fn cjk_is_split_by_the_letter_class() {
        assert_eq!(pieces("你好，世界！"), ["你好", "，世界", "！"]);
    }

    #[test]
    fn a_python_snippet_splits_like_the_reference() {
        assert_eq!(
            pieces("def f(x):\n    return x + 1\n"),
            ["def", " f", "(x", "):\n", "   ", " return", " x", " +", " ", "1", "\n"]
        );
    }

    #[test]
    fn every_piece_concatenates_back_to_the_input() {
        for text in [
            "Hello, world!",
            "  leading and trailing  ",
            "def f(x):\n    return x + 1\n",
            "a\r\n\r\nb",
            "你好，世界！",
            "🧠🧠",
            "'quoted' and it's",
            "",
        ] {
            let joined: String = pieces(text).concat();
            assert_eq!(joined, text, "{text:?}");
        }
    }
}

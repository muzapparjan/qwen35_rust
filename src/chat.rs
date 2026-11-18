//! The chat template, as far as text-only generation needs it.
//!
//! The release ships a Jinja template. Rendering it needs a template engine and
//! a JSON parser, so this module implements the text-only subset directly and is
//! checked against the template's own output case by case:
//!
//! - content is trimmed; a system message is emitted once, before the turn loop,
//!   and only in the first position;
//! - an assistant message that comes *after* the last user query is wrapped in a
//!   `think` block, and its reasoning text goes inside that block;
//! - the generation prompt is `<|im_start|>assistant\n`, followed by an empty
//!   `think` block when thinking is off.
//!
//! What it does not accept: tool calls and tool messages. The template renders
//! them with a JSON encoding of each call, and a caller that needs that should
//! render its own text — [`render`] would silently drop the calls otherwise.

use std::fmt;

/// Who wrote a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One turn of a conversation.
#[derive(Clone, Copy, Debug)]
pub struct Message<'a> {
    pub role: Role,
    pub content: &'a str,
    /// Reasoning text for an assistant turn; only rendered for a turn after the
    /// last user query.
    pub reasoning: Option<&'a str>,
}

impl<'a> Message<'a> {
    pub fn system(content: &'a str) -> Message<'a> {
        Message {
            role: Role::System,
            content,
            reasoning: None,
        }
    }

    pub fn user(content: &'a str) -> Message<'a> {
        Message {
            role: Role::User,
            content,
            reasoning: None,
        }
    }

    pub fn assistant(content: &'a str) -> Message<'a> {
        Message {
            role: Role::Assistant,
            content,
            reasoning: None,
        }
    }

    /// An assistant turn with reasoning, as a model that thinks would produce.
    pub fn assistant_thinking(content: &'a str, reasoning: &'a str) -> Message<'a> {
        Message {
            role: Role::Assistant,
            content,
            reasoning: Some(reasoning),
        }
    }
}

/// How to finish the prompt.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Append `<|im_start|>assistant\n`, which is what a generation call wants.
    pub add_generation_prompt: bool,
    /// Open a `think` block after the assistant header instead of closing an
    /// empty one.
    pub enable_thinking: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            add_generation_prompt: true,
            enable_thinking: false,
        }
    }
}

/// Why a conversation cannot be rendered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    NoMessages,
    /// The template requires a user turn to answer.
    NoUserQuery,
    /// A system message may only be the first message.
    SystemNotFirst(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoMessages => write!(formatter, "no messages"),
            Error::NoUserQuery => write!(formatter, "no user query"),
            Error::SystemNotFirst(index) => {
                write!(formatter, "system message at {index} is not first")
            }
        }
    }
}

impl std::error::Error for Error {}

/// The index of the last user turn that asks something.
///
/// The template walks the messages backwards and takes the first user turn whose
/// content is not a wrapped tool response.
fn last_query(messages: &[Message]) -> Option<usize> {
    messages.iter().enumerate().rev().find_map(|(index, message)| {
        if message.role != Role::User {
            return None;
        }
        let content = message.content.trim();
        let is_tool_response =
            content.starts_with("<tool_response>") && content.ends_with("</tool_response>");
        (!is_tool_response).then_some(index)
    })
}

/// Render a conversation the way the release's template does.
pub fn render(messages: &[Message], options: &Options) -> Result<String, Error> {
    if messages.is_empty() {
        return Err(Error::NoMessages);
    }
    let last_query = last_query(messages).ok_or(Error::NoUserQuery)?;
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System && index != 0 {
            return Err(Error::SystemNotFirst(index));
        }
    }

    let mut output = String::new();
    let mut start = 0;
    if messages[0].role == Role::System {
        output.push_str("<|im_start|>system\n");
        output.push_str(messages[0].content.trim());
        output.push_str("<|im_end|>\n");
        start = 1;
    }
    for (index, message) in messages.iter().enumerate().skip(start) {
        match message.role {
            Role::System => unreachable!("checked above"),
            Role::User => {
                output.push_str("<|im_start|>user\n");
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                output.push_str("<|im_start|>assistant\n");
                // The block is part of the turn only where a turn carries reasoning.
                // A message that carries none renders as its answer alone; an empty
                // reasoning string is written as the empty block it is.
                if index > last_query
                    && let Some(reasoning) = message.reasoning
                {
                    output.push_str("<think>\n");
                    output.push_str(reasoning.trim());
                    output.push_str("\n</think>\n\n");
                }
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
            }
        }
    }
    if options.add_generation_prompt {
        output.push_str("<|im_start|>assistant\n");
        if options.enable_thinking {
            output.push_str("<think>\n");
        } else {
            output.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_default(messages: &[Message]) -> String {
        render(messages, &Options::default()).expect("renders")
    }

    #[test]
    fn a_single_user_turn() {
        assert_eq!(
            render_default(&[Message::user("你好")]),
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn thinking_changes_the_generation_prompt() {
        let options = Options {
            add_generation_prompt: true,
            enable_thinking: true,
        };
        assert_eq!(
            render(&[Message::user("你好")], &options).unwrap(),
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn a_system_message_comes_first_and_is_trimmed() {
        assert_eq!(
            render_default(&[Message::system("  padded  \n"), Message::user(" Hi ")]),
            "<|im_start|>system\npadded<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn a_multi_turn_conversation() {
        let rendered = render_default(&[
            Message::system("Be brief."),
            Message::user("What is 2+2?"),
            Message::assistant("4"),
            Message::user("And 3+3?"),
        ]);
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\nWhat is 2+2?<|im_end|>\n\
             <|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\nAnd 3+3?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn reasoning_is_dropped_before_the_last_query() {
        // The assistant turn is an answer to the only user turn, so it is not
        // after the last query and its reasoning is not rendered.
        let rendered = render_default(&[
            Message::user("Add them."),
            Message::assistant_thinking("4", "two plus two"),
            Message::user("Thanks"),
        ]);
        assert_eq!(
            rendered,
            "<|im_start|>user\nAdd them.<|im_end|>\n\
             <|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\nThanks<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn reasoning_after_the_last_query_is_rendered() {
        let rendered = render_default(&[
            Message::user("Explain."),
            Message::assistant_thinking("It is four.", "  2+2 = 4  "),
        ]);
        assert_eq!(
            rendered,
            "<|im_start|>user\nExplain.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n2+2 = 4\n</think>\n\nIt is four.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn an_answer_without_reasoning_has_no_thinking_block() {
        let rendered = render_default(&[
            Message::user("Explain."),
            Message::assistant("It is four."),
        ]);
        assert_eq!(
            rendered,
            "<|im_start|>user\nExplain.<|im_end|>\n\
             <|im_start|>assistant\nIt is four.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn a_system_message_may_not_come_later() {
        assert_eq!(
            render(
                &[Message::user("hi"), Message::system("late")],
                &Options::default()
            ),
            Err(Error::SystemNotFirst(1))
        );
    }

    #[test]
    fn a_conversation_needs_a_user_turn() {
        assert_eq!(
            render(&[Message::assistant("hi")], &Options::default()),
            Err(Error::NoUserQuery)
        );
        assert_eq!(render(&[], &Options::default()), Err(Error::NoMessages));
    }

    #[test]
    fn a_tool_response_alone_is_not_a_query() {
        let wrapped = "<tool_response>\n42\n</tool_response>";
        assert_eq!(
            render(&[Message::user(wrapped)], &Options::default()),
            Err(Error::NoUserQuery)
        );
    }

    #[test]
    fn without_the_generation_prompt_the_prompt_ends_at_the_last_turn() {
        let options = Options {
            add_generation_prompt: false,
            enable_thinking: false,
        };
        assert_eq!(
            render(&[Message::user("Hi")], &options).unwrap(),
            "<|im_start|>user\nHi<|im_end|>\n"
        );
    }
}

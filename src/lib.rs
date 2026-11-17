//! Qwen3.5-0.8B inference in Rust.
//!
//! The default build is CPU only and has no dependencies: the numeric kernels,
//! the tokenizer and the quantised weight reader are all `std`. The `wgpu`
//! feature adds a second backend for the same arithmetic; `force-scalar` turns
//! the vector kernels off so the portable path can be measured end to end.
//!
//! The weights are a QWQT container embedded with `include_bytes!`, so the
//! library performs no file IO at run time.

mod blob;
pub mod chat;
pub mod cpu;
pub mod model;
pub mod multimodal;
pub mod ops;
pub mod parallel;
pub mod tokenizer;
pub mod vision;
pub mod weights;

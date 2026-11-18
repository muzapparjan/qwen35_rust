//! Stops the build with a readable message when the weight container is absent.
//!
//! The container is a release asset rather than a tracked file, so a fresh
//! clone has to run `scripts/fetch-weights.ps1` before the crate can compile.
//! Without this, `include_bytes!` in `src/weights/mod.rs` is the first thing to
//! fail, and its message names a path without saying what to do about it.

use std::path::Path;

const CONTAINER: &str = "src/weights/qwen3.5-0.8b-q4.qwqt";

fn main() {
    if !Path::new(CONTAINER).exists() {
        panic!(
            "the weight container is missing: {CONTAINER}\n\
             Run scripts/fetch-weights.ps1 to download and verify it."
        );
    }
    println!("cargo:rerun-if-changed={CONTAINER}");
}

//! Helpers shared by the integration tests.
//!
//! `crc32` is zlib's (IEEE polynomial) so the digests can be produced by the
//! dump scripts with one `zlib.crc32` call and compared here without a
//! dependency.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

static TABLE: [u32; 256] = build_table();

/// CRC-32 of `bytes`, the value `zlib.crc32` returns.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut value = 0xFFFF_FFFFu32;
    for byte in bytes {
        let slot = ((value ^ u32::from(*byte)) & 0xff) as usize;
        value = TABLE[slot] ^ (value >> 8);
    }
    value ^ 0xFFFF_FFFF
}

/// The bytes `crc32` is taken over: little-endian `f32`, row-major.
pub fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 * values.len());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// A path inside `dev/fixtures`.
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("dev")
        .join("fixtures")
        .join(name)
}

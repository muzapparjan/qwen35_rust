//! Embedded byte buffers.
//!
//! `include_bytes!` gives no alignment guarantee, and a misaligned `u32` load is
//! undefined behaviour, so each blob is wrapped in an aligned array. The
//! container aligns its own sections to 64 bytes, which one aligned base is
//! enough to cover.

/// A byte buffer aligned for `u32` and `u64` loads.
#[repr(C, align(64))]
pub struct Aligned<const N: usize>(pub [u8; N]);

/// The `u32` at `offset`, little-endian.
pub fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

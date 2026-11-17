//! The model weights, embedded at compile time.
//!
//! The container is a QWQT file: an 8-byte prefix, a UTF-8 JSON header and a
//! payload that every tensor slices into. `manifest.rs` is generated from that
//! header, so nothing here parses JSON or does file IO.
//!
//! The file is embedded whole rather than unpacked into Rust constants: the
//! payload is 494,764,800 bytes, and rewriting it as generated source would cost
//! a much slower compile for no gain. Every offset in the manifest addresses the
//! embedded file directly.

pub mod manifest;

use manifest::{TensorSpec, FILE_LEN, TENSORS};

/// A byte buffer aligned so it can be reinterpreted as `f32` or `u16` without a
/// copy.
///
/// `include_bytes!` gives no alignment guarantee; a misaligned `f32` slice would
/// be undefined behaviour. The container aligns every tensor buffer to 64 bytes,
/// so one aligned base is enough for all of them.
#[repr(C, align(64))]
pub struct Aligned<const N: usize>(pub [u8; N]);

// `Aligned<N>` fixes the length, so a container swapped for a different one
// stops the build instead of letting the manifest address bytes that are not
// there.
static BLOB: Aligned<FILE_LEN> = Aligned(*include_bytes!("qwen3.5-0.8b-q4.qwqt"));

/// The embedded container, byte for byte, header included.
pub fn blob() -> &'static [u8] {
    &BLOB.0
}

// Integer relations between the architecture constants. A change that breaks one
// of these stops the build rather than a test run. They live here rather than in
// `manifest.rs` because that file is generated.
const _: () = assert!(manifest::HEAD_DIM * manifest::NUM_ATTENTION_HEADS == manifest::LINEAR_KEY_DIM);
const _: () = assert!(manifest::LINEAR_KEY_DIM == manifest::LINEAR_VALUE_DIM);
const _: () = assert!(manifest::LINEAR_KEY_HEADS == manifest::LINEAR_VALUE_HEADS);
const _: () = assert!(manifest::LINEAR_CONV_DIM == 2 * manifest::LINEAR_KEY_DIM + manifest::LINEAR_VALUE_DIM);
const _: () = assert!(manifest::ROTARY_DIM % 2 == 0);
const _: () = assert!(manifest::ROTARY_DIM == 2 * (manifest::MROPE_SECTION[0] + manifest::MROPE_SECTION[1] + manifest::MROPE_SECTION[2]));
const _: () = assert!(manifest::ROTARY_DIM <= manifest::HEAD_DIM);
const _: () = assert!(manifest::VISION_HIDDEN == manifest::VISION_HEAD_DIM * manifest::VISION_HEADS);
const _: () = assert!(manifest::VISION_MERGER_HIDDEN == manifest::VISION_HIDDEN * manifest::VISION_MERGE * manifest::VISION_MERGE);
const _: () = assert!(manifest::VISION_OUT_HIDDEN == manifest::HIDDEN_SIZE);
const _: () = assert!(manifest::VISION_GRID_SIDE * manifest::VISION_GRID_SIDE == manifest::VISION_POSITIONS);
const _: () = assert!(manifest::TOKENIZER_VOCAB_SIZE <= manifest::VOCAB_SIZE);
const _: () = assert!(manifest::LINEAR_LAYERS.len() + manifest::FULL_LAYERS.len() == manifest::NUM_HIDDEN_LAYERS);
const _: () = assert!(TENSORS.len() == manifest::TENSOR_COUNT);
const _: () = assert!(FILE_LEN == manifest::PAYLOAD_OFFSET + 494_764_800);
const _: () = assert!(cfg!(target_endian = "little"), "the container is little endian");

/// Where a tensor's values live.
///
/// Dense tensors stay as the container's bytes instead of being reinterpreted:
/// the payload starts at byte 80,646, which is not a multiple of four, so an
/// `f32` slice over it would be misaligned. The dense tensors are 2.4M values in
/// total, so converting them once costs nothing next to the 871M quantised ones.
#[derive(Clone, Copy, Debug)]
pub enum Data {
    /// Little-endian `f32`, four bytes per value.
    F32(&'static [u8]),
    /// Little-endian `bf16` or `f16`, two bytes per value.
    Half(&'static [u8]),
    /// Grouped symmetric integers, with one `f16` scale per group.
    ///
    /// A group never straddles a row: groups run along the last axis of the
    /// row-major flattening, which the container guarantees by construction.
    Quantized {
        /// `groups` little-endian `f16` scales.
        scales: &'static [u8],
        /// Packed values, least significant field first, offset by `2^(bits-1)`.
        integers: &'static [u8],
        bits: u8,
    },
}

/// One tensor of the model, with its PyTorch shape.
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub spec: &'static TensorSpec,
    pub data: Data,
}

impl Tensor {
    /// Number of values, the product of the shape.
    pub fn numel(&self) -> usize {
        self.spec.shape.iter().product()
    }

    /// The tensor's whole buffer, scales and padding included.
    pub fn raw(&self) -> &'static [u8] {
        &blob()[self.spec.offset..self.spec.offset + self.spec.length]
    }

    /// The dense values as `f32`, converting from `bf16` or `f16` when needed.
    ///
    /// Panics for a quantised tensor: those are read as group indices by the
    /// kernels and are never expanded into a whole `f32` tensor.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.data {
            Data::F32(bytes) => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            Data::Half(bytes) => match self.spec.kind {
                "bf16" => bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
                    })
                    .collect(),
                _ => bytes
                    .chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect(),
            },
            Data::Quantized { .. } => {
                panic!("{}: a quantised tensor has no dense values", self.spec.name)
            }
        }
    }
}

/// The manifest entry for `name`, if the checkpoint has such a tensor.
pub fn spec(name: &str) -> Option<&'static TensorSpec> {
    TENSORS
        .binary_search_by(|entry| entry.name.cmp(name))
        .ok()
        .map(|index| &TENSORS[index])
}

/// The tensor called `name`.
pub fn tensor(name: &str) -> Option<Tensor> {
    let spec = spec(name)?;
    Some(Tensor {
        spec,
        data: data_of(spec),
    })
}

/// The tensor called `name`; panics when the container lacks it.
///
/// Every call site passes a constant name, so a miss means the manifest and the
/// code disagree.
pub fn expect(name: &str) -> Tensor {
    match tensor(name) {
        Some(tensor) => tensor,
        None => panic!("the container has no tensor called {name:?}"),
    }
}

fn data_of(spec: &'static TensorSpec) -> Data {
    let numel = spec.shape.iter().product::<usize>();
    let buffer = &blob()[spec.offset..spec.offset + spec.length];
    match spec.kind {
        "f32" => {
            assert_eq!(buffer.len(), 4 * numel, "{}: f32 buffer size", spec.name);
            Data::F32(buffer)
        }
        "bf16" | "f16" => {
            assert_eq!(buffer.len(), 2 * numel, "{}: half buffer size", spec.name);
            Data::Half(buffer)
        }
        kind => {
            let bits = match kind {
                "q2" => 2,
                "q3" => 3,
                "q4" => 4,
                "q8" => 8,
                other => panic!("{}: unknown kind {other:?}", spec.name),
            };
            let scales_len = spec.scales_len;
            assert_eq!(spec.groups, numel / spec.group, "{}: group count", spec.name);
            let scales = &buffer[..scales_len];
            let integers = &buffer[scales_len..];
            Data::Quantized {
                scales,
                integers,
                bits,
            }
        }
    }
}

/// IEEE 754 binary16 to binary32.
///
/// The container's own scale values are f16; no dense tensor in this checkpoint
/// uses the format, but a container is allowed to.
fn f16_to_f32(half: u16) -> f32 {
    let sign = (u32::from(half) & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1f;
    let mantissa = u32::from(half & 0x03ff);
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal: renormalise by shifting until the implicit bit appears.
            let mut exponent: i32 = 127 - 15 + 1;
            let mut mantissa = mantissa;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | ((exponent as u32) << 23) | ((mantissa & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        // Infinity and NaN keep their payload; quiet NaN stays quiet.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifest::ALIGNMENT;

    #[test]
    fn blob_starts_with_the_container_magic() {
        assert_eq!(&blob()[..4], b"QWQT");
        assert_eq!(blob().len(), FILE_LEN);
    }

    #[test]
    fn blob_is_aligned_for_the_kernels() {
        assert_eq!(blob().as_ptr() as usize % ALIGNMENT, 0);
    }

    #[test]
    fn every_tensor_is_inside_the_blob_and_aligned() {
        let mut covered = 0usize;
        for entry in TENSORS {
            assert!(
                (entry.offset - manifest::PAYLOAD_OFFSET) % ALIGNMENT == 0,
                "{}: not on a payload boundary",
                entry.name
            );
            assert!(
                entry.offset + entry.length <= FILE_LEN,
                "{}: runs past the end",
                entry.name
            );
            assert!(entry.offset >= manifest::PAYLOAD_OFFSET, "{}: inside the header", entry.name);
            covered += entry.length;
        }
        assert!(covered > 0);
        assert!(
            covered <= FILE_LEN - manifest::PAYLOAD_OFFSET,
            "tensors overlap the payload padding"
        );
    }

    #[test]
    fn every_name_is_found_and_names_are_sorted() {
        for pair in TENSORS.windows(2) {
            assert!(pair[0].name < pair[1].name, "{} is not before {}", pair[0].name, pair[1].name);
        }
        for entry in TENSORS {
            let found = spec(entry.name).expect("sorted lookup");
            assert_eq!(found.offset, entry.offset);
        }
        assert!(spec("no.such.tensor").is_none());
    }

    #[test]
    fn tensor_counts_and_parameters_match_the_container() {
        assert_eq!(manifest::TENSOR_COUNT, 488);
        assert_eq!(manifest::MODEL_TENSOR_COUNT, 473);
        assert_eq!(manifest::PARAM_COUNT, 873_438_784);
        assert_eq!(manifest::MODEL_PARAM_COUNT, 852_985_920);
        assert_eq!(manifest::PARAM_COUNT - manifest::MODEL_PARAM_COUNT, 20_452_864);
    }

    #[test]
    fn shapes_agree_with_element_counts() {
        for entry in TENSORS {
            let numel: usize = entry.shape.iter().product();
            match entry.kind {
                "f32" => assert_eq!(entry.length, 4 * numel, "{}", entry.name),
                "bf16" | "f16" => assert_eq!(entry.length, 2 * numel, "{}", entry.name),
                kind => {
                    let bits = kind[1..].parse::<usize>().expect("quantised kind");
                    assert_eq!(entry.groups, numel / entry.group, "{}", entry.name);
                    assert_eq!(entry.length, entry.scales_len + numel * bits / 8, "{}", entry.name);
                }
            }
        }
    }

    #[test]
    fn f32_tensors_read_back_the_stored_numbers() {
        // `A_log` is a real parameter of the model, not padding: it holds the log
        // of a decay rate, measured at -6.5938 .. 0.2578 in this container.
        let tensor = expect("model.language_model.layers.0.linear_attn.A_log");
        assert_eq!(tensor.numel(), manifest::LINEAR_VALUE_HEADS);
        assert!(matches!(tensor.data, Data::F32(_)), "A_log should be f32");
        let values = tensor.to_f32();
        assert_eq!(values.len(), manifest::LINEAR_VALUE_HEADS);
        for value in values {
            assert!(value.is_finite(), "A_log holds {value}");
            assert!((-7.0..0.5).contains(&value), "A_log holds {value}");
        }
    }

    #[test]
    fn bf16_tensors_decode_to_the_stored_numbers() {
        // Every norm weight in the container is bf16; this one is a 1024-wide
        // vector of trained values, so a wrong shift would show up immediately.
        let tensor = expect("model.language_model.layers.0.input_layernorm.weight");
        assert_eq!(tensor.spec.kind, "bf16");
        let values = tensor.to_f32();
        assert_eq!(values.len(), manifest::HIDDEN_SIZE);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
        // bf16 keeps the top 16 bits, so the value round-trips through f32.
        for value in &values {
            assert_eq!(f32::from_bits(value.to_bits() & 0xffff_0000), *value);
        }
    }

    #[test]
    fn f16_decoding_matches_known_values() {
        let cases: [(u16, f32); 8] = [
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x7c00, f32::INFINITY),
            (0x7e00, f32::NAN),
        ];
        for (bits, expected) in cases {
            let value = f16_to_f32(bits);
            if expected.is_nan() {
                assert!(value.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(value.to_bits(), expected.to_bits(), "{bits:#06x}");
            }
        }
    }

    #[test]
    fn quantised_tensors_expose_scales_and_integers() {
        let tensor = expect("model.language_model.embed_tokens.weight");
        assert_eq!(tensor.spec.kind, "q4");
        match tensor.data {
            Data::Quantized { scales, integers, bits } => {
                assert_eq!(bits, 4);
                assert_eq!(scales.len(), tensor.spec.scales_len);
                assert_eq!(scales.len(), 2 * tensor.spec.groups);
                assert_eq!(integers.len(), tensor.numel() / 2);
            }
            other => panic!("embed_tokens should be quantised, got {other:?}"),
        }
    }
}

//! Numeric kernels.
//!
//! Sequences and weight matrices are row-major, and linear-layer weights follow
//! PyTorch: `[out_features, in_features]`.
//!
//! The quantised kernels read the packed integers in place. Expanding a weight
//! into `f32` first is what makes a quantised 0.8B model cost several gigabytes
//! of temporaries, so `dequantize` exists for tests and for the handful of dense
//! helper tensors, while the matrix kernels take the packed form directly.
//!
//! Everything here is decided in `f32` unless a comment says otherwise, and the
//! order of operations is part of each function's contract: the port is compared
//! against a reference that sums in a particular order.

use crate::cpu;

/// Values packed into one storage unit (a byte, or a three-byte group for
/// three-bit values).
fn packing_unit(bits: u8) -> usize {
    match bits {
        2 => 4,
        3 => 8,
        4 => 2,
        8 => 1,
        other => panic!("no packing for {other}-bit values"),
    }
}

/// Bytes one storage unit occupies.
fn unit_bytes(bits: u8) -> usize {
    match bits {
        2 | 4 | 8 => 1,
        3 => 3,
        other => panic!("no packing for {other}-bit values"),
    }
}

/// The largest magnitude a value can carry, before the sign.
fn magnitude(bits: u8) -> i32 {
    (1 << (bits - 1)) - 1
}

/// Decode one IEEE 754 binary16 value from little-endian bytes.
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = (u32::from(half) & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1f;
    let mantissa = u32::from(half & 0x03ff);
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal: shift until the implicit bit appears.
            let mut exponent: i32 = 127 - 15 + 1;
            let mut mantissa = mantissa;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | ((exponent as u32) << 23) | ((mantissa & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        // Infinity and NaN keep their payload; a quiet NaN stays quiet.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

/// The scale of group `index`, from little-endian `f16` bytes.
fn scale_of(scales: &[u8], index: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([scales[2 * index], scales[2 * index + 1]]))
}

fn check(bits: u8, group: usize, scales: &[u8], integers: &[u8], out: &[f32]) {
    assert!(group > 0, "group must be positive");
    assert_eq!(
        out.len() % group,
        0,
        "{} values do not fill groups of {group}",
        out.len()
    );
    assert!(
        group % packing_unit(bits) == 0,
        "a group of {group} {bits}-bit values straddles a storage unit"
    );
    let groups = out.len() / group;
    assert!(
        scales.len() >= 2 * groups,
        "{} scale bytes cover {groups} groups",
        scales.len()
    );
    assert_eq!(
        integers.len(),
        groups * group / packing_unit(bits) * unit_bytes(bits),
        "{} bytes hold {} {bits}-bit values",
        integers.len(),
        groups * group
    );
}

/// Expand grouped symmetric quantisation into `out`.
///
/// `q = stored - 2^(bits-1)` and `value = q * scale`, once per element. The
/// scale is `f16` on disk and multiplies a whole group; groups run along the
/// last axis of the row-major flattening and never straddle a row.
///
/// Every value is one multiply of an exactly-representable integer by an `f32`,
/// so the scalar and vector paths produce identical bits: there is no reduction
/// order to disagree about.
pub fn dequantize(bits: u8, group: usize, scales: &[u8], integers: &[u8], out: &mut [f32]) {
    check(bits, group, scales, integers, out);
    if out.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if cpu::features().avx2_fma && bits != 2 && bits != 3 {
        // SAFETY: the feature test above established AVX2 and FMA support.
        unsafe { dequantize_avx2(bits, group, scales, integers, out) };
        return;
    }
    dequantize_scalar(bits, group, scales, integers, out);
}

/// Portable expansion: one storage unit at a time.
pub fn dequantize_scalar(bits: u8, group: usize, scales: &[u8], integers: &[u8], out: &mut [f32]) {
    check(bits, group, scales, integers, out);
    let groups = out.len() / group;
    let bytes_per_group = group * bits as usize / 8;
    for index in 0..groups {
        let scale = scale_of(scales, index);
        let source = &integers[index * bytes_per_group..(index + 1) * bytes_per_group];
        let target = &mut out[index * group..(index + 1) * group];
        unpack_group(bits, source, target, scale);
    }
}

fn unpack_group(bits: u8, source: &[u8], target: &mut [f32], scale: f32) {
    let offset = 1 << (bits - 1);
    match bits {
        8 => {
            for (value, byte) in target.iter_mut().zip(source) {
                *value = f32::from(i16::from(*byte) - offset) * scale;
            }
        }
        4 => {
            for (pair, byte) in target.chunks_exact_mut(2).zip(source) {
                pair[0] = f32::from(i16::from(byte & 0x0f) - offset) * scale;
                pair[1] = f32::from(i16::from(byte >> 4) - offset) * scale;
            }
        }
        2 => {
            for (quad, byte) in target.chunks_exact_mut(4).zip(source) {
                quad[0] = f32::from(i16::from(byte & 0x03) - offset) * scale;
                quad[1] = f32::from(i16::from((byte >> 2) & 0x03) - offset) * scale;
                quad[2] = f32::from(i16::from((byte >> 4) & 0x03) - offset) * scale;
                quad[3] = f32::from(i16::from(byte >> 6) - offset) * scale;
            }
        }
        3 => {
            // Eight values in three little-endian bytes: value i is bits 3i..3i+2.
            for (octet, word) in target.chunks_exact_mut(8).zip(source.chunks_exact(3)) {
                let packed = u32::from(word[0]) | (u32::from(word[1]) << 8) | (u32::from(word[2]) << 16);
                for (index, value) in octet.iter_mut().enumerate() {
                    let digit = ((packed >> (3 * index)) & 0x07) as i16;
                    *value = f32::from(digit - offset) * scale;
                }
            }
        }
        other => panic!("no unpacking for {other}-bit values"),
    }
}

/// # Safety
/// The caller must have established AVX2 and FMA support.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequantize_avx2(bits: u8, group: usize, scales: &[u8], integers: &[u8], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let groups = out.len() / group;
    let bytes_per_group = group * bits as usize / 8;
    for index in 0..groups {
        let scale = _mm256_set1_ps(scale_of(scales, index));
        let source = &integers[index * bytes_per_group..(index + 1) * bytes_per_group];
        let target = &mut out[index * group..(index + 1) * group];
        // SAFETY: both slices are indexed only inside their own lengths; the
        // vector loops stop at a whole unit and the scalar tail covers the rest.
        unsafe {
            match bits {
                8 => {
                    let offset = _mm256_set1_epi32(magnitude(8) + 1);
                    let mut byte = 0;
                    while byte + 8 <= source.len() {
                        let raw = _mm_loadl_epi64(source.as_ptr().add(byte).cast::<__m128i>());
                        let wide = _mm256_sub_epi32(_mm256_cvtepu8_epi32(raw), offset);
                        let values = _mm256_mul_ps(_mm256_cvtepi32_ps(wide), scale);
                        _mm256_storeu_ps(target.as_mut_ptr().add(byte), values);
                        byte += 8;
                    }
                    for tail in byte..source.len() {
                        target[tail] = f32::from(i16::from(source[tail]) - 128) * scale_of(scales, index);
                    }
                }
                4 => {
                    let mask = _mm_set1_epi8(0x0f);
                    let offset = _mm256_set1_epi32(8);
                    let mut byte = 0;
                    while byte + 8 <= source.len() {
                        let raw = _mm_loadl_epi64(source.as_ptr().add(byte).cast::<__m128i>());
                        let low = _mm_and_si128(raw, mask);
                        // Shifting 16-bit lanes moves a byte's high nibble into the
                        // next lane's low byte; masking afterwards leaves exactly
                        // the high nibble of each byte.
                        let high = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
                        let interleaved = _mm_unpacklo_epi8(low, high);
                        let first = _mm256_cvtepu8_epi32(interleaved);
                        let second = _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(interleaved, interleaved));
                        let first = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(first, offset)), scale);
                        let second = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(second, offset)), scale);
                        let destination = target.as_mut_ptr().add(2 * byte);
                        _mm256_storeu_ps(destination, first);
                        _mm256_storeu_ps(destination.add(8), second);
                        byte += 8;
                    }
                    if byte < source.len() {
                        let scale = scale_of(scales, index);
                        unpack_group(4, &source[byte..], &mut target[2 * byte..], scale);
                    }
                }
                other => panic!("no vector path for {other}-bit values"),
            }
        }
    }
}

/// `sum_i a[i] * b[i]`, four accumulator chains.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sums = [0.0f32; 4];
    let mut left = a.chunks_exact(4);
    let mut right = b.chunks_exact(4);
    for (x, y) in left.by_ref().zip(right.by_ref()) {
        sums[0] += x[0] * y[0];
        sums[1] += x[1] * y[1];
        sums[2] += x[2] * y[2];
        sums[3] += x[3] * y[3];
    }
    let mut total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for (x, y) in left.remainder().iter().zip(right.remainder()) {
        total += x * y;
    }
    total
}

/// The value of row `index` of `weight`, expanded to `f32`.
///
/// Used for the embedding lookup: 18 rows of 1024 values, not a whole table.
pub fn row_values(weight: &crate::weights::Matrix, index: usize, out: &mut [f32]) {
    assert!(index < weight.rows, "row {index} of {}", weight.rows);
    assert_eq!(out.len(), weight.columns);
    match weight.data {
        crate::weights::Data::Quantized {
            scales,
            integers,
            bits,
        } => {
            let columns = weight.columns;
            let groups = columns / weight.group;
            let scale_bytes = 2 * groups;
            let value_bytes = columns * bits as usize / 8;
            dequantize(
                bits,
                weight.group,
                &scales[index * scale_bytes..(index + 1) * scale_bytes],
                &integers[index * value_bytes..(index + 1) * value_bytes],
                out,
            );
        }
        crate::weights::Data::F32(bytes) => {
            let start = index * 4 * weight.columns;
            for (slot, chunk) in out.iter_mut().zip(bytes[start..].chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        crate::weights::Data::Half(bytes) => {
            let start = index * 2 * weight.columns;
            let row = &bytes[start..start + 2 * weight.columns];
            for (slot, chunk) in out.iter_mut().zip(row.chunks_exact(2)) {
                *slot = match weight.kind {
                    "bf16" => f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16),
                    _ => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
                };
            }
        }
    }
}

/// `out[t, o] = sum_i x[t, i] * weight[o, i]`, reading the weight in place.
///
/// A quantised weight is never expanded into a whole `f32` matrix: the inner
/// loop decodes one group at a time, so the memory a call needs is the input,
/// the output and one group's worth of values.
pub fn linear(
    x: &[f32],
    rows: usize,
    weight: &crate::weights::Matrix,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    assert_eq!(x.len(), rows * weight.columns, "input shape");
    assert_eq!(out.len(), rows * weight.rows, "output shape");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), weight.rows, "bias shape");
    }
    match weight.data {
        crate::weights::Data::Quantized {
            scales,
            integers,
            bits,
        } => {
            assert_eq!(bits, 4, "only four-bit weights are used by this model");
            let columns = weight.columns;
            let group = weight.group;
            let groups = columns / group;
            for row in 0..rows {
                let source = &x[row * columns..(row + 1) * columns];
                let target = &mut out[row * weight.rows..(row + 1) * weight.rows];
                for (output, slot) in target.iter_mut().enumerate() {
                    let scale_base = output * groups;
                    let integer_base = output * columns / 2;
                    let mut total = 0.0f32;
                    for group_index in 0..groups {
                        let scale = f16_to_f32(u16::from_le_bytes([
                            scales[2 * (scale_base + group_index)],
                            scales[2 * (scale_base + group_index) + 1],
                        ]));
                        let bytes = &integers
                            [integer_base + group_index * group / 2..integer_base + (group_index + 1) * group / 2];
                        let source_base = group_index * group;
                        // Two chains, one per nibble lane: the low nibbles and the
                        // high nibbles are independent sums until the end.
                        let mut low_sum = 0.0f32;
                        let mut high_sum = 0.0f32;
                        for (pair, byte) in bytes.iter().enumerate() {
                            let low = f32::from(i16::from(byte & 0x0f) - 8);
                            let high = f32::from(i16::from(byte >> 4) - 8);
                            let index = source_base + 2 * pair;
                            low_sum += low * source[index];
                            high_sum += high * source[index + 1];
                        }
                        total += (low_sum + high_sum) * scale;
                    }
                    *slot = match bias {
                        Some(bias) => total + bias[output],
                        None => total,
                    };
                }
            }
        }
        _ => {
            // Dense weights are the small ones (norms, conv, the two 16-wide
            // projections), so expanding them per call costs nothing.
            let mut dense = vec![0.0f32; weight.rows * weight.columns];
            for row in 0..weight.rows {
                row_values(
                    weight,
                    row,
                    &mut dense[row * weight.columns..(row + 1) * weight.columns],
                );
            }
            for row in 0..rows {
                let source = &x[row * weight.columns..(row + 1) * weight.columns];
                let target = &mut out[row * weight.rows..(row + 1) * weight.rows];
                for (output, slot) in target.iter_mut().enumerate() {
                    let weights = &dense[output * weight.columns..(output + 1) * weight.columns];
                    *slot = match bias {
                        Some(bias) => dot(source, weights) + bias[output],
                        None => dot(source, weights),
                    };
                }
            }
        }
    }
}

/// `silu(x) = x * sigmoid(x)`.
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// `sigmoid(x)`.
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// `softplus(x)`, with torch's default threshold: `x` itself above 20.
pub fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// Zero-centred RMS normalisation over the last axis: `x / rms(x) * (1 + w)`.
pub fn rms_norm_zero_centered(
    x: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    assert_eq!(x.len(), rows * width);
    assert_eq!(weight.len(), width);
    assert_eq!(out.len(), rows * width);
    for row in 0..rows {
        let source = &x[row * width..(row + 1) * width];
        let target = &mut out[row * width..(row + 1) * width];
        let mean_square = source.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        for index in 0..width {
            target[index] = source[index] * inverse * (1.0 + weight[index]);
        }
    }
}

/// Gated RMS normalisation: `w * x / rms(x) * silu(gate)`, per row of `width`.
pub fn rms_norm_gated(
    x: &[f32],
    gate: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    assert_eq!(x.len(), rows * width);
    assert_eq!(gate.len(), rows * width);
    assert_eq!(weight.len(), width);
    assert_eq!(out.len(), rows * width);
    for row in 0..rows {
        let source = &x[row * width..(row + 1) * width];
        let gate = &gate[row * width..(row + 1) * width];
        let target = &mut out[row * width..(row + 1) * width];
        let mean_square = source.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        for index in 0..width {
            target[index] = weight[index] * source[index] * inverse * silu(gate[index]);
        }
    }
}

/// Divide a row by the square root of its sum of squares plus `eps`.
pub fn l2_normalize(row: &mut [f32], eps: f32) {
    let sum: f32 = row.iter().map(|value| value * value).sum();
    let inverse = 1.0 / (sum + eps).sqrt();
    for value in row.iter_mut() {
        *value *= inverse;
    }
}

/// In-place softmax over a row, in `f32` like the reference's attention.
pub fn softmax(row: &mut [f32]) {
    let mut largest = f32::NEG_INFINITY;
    for value in row.iter() {
        if *value > largest {
            largest = *value;
        }
    }
    let mut total = 0.0f32;
    for value in row.iter_mut() {
        let exponent = (*value - largest).exp();
        *value = exponent;
        total += exponent;
    }
    let inverse = 1.0 / total;
    for value in row.iter_mut() {
        *value *= inverse;
    }
}

/// Rotate the first `rotary` values of `vector` with `rotate_half` pairing.
///
/// The pairs are `(i, i + rotary/2)`, and `cos`/`sin` hold `rotary/2` values
/// because the rotation repeats them over both halves.
pub fn apply_rope(vector: &mut [f32], rotary: usize, cos: &[f32], sin: &[f32]) {
    let half = rotary / 2;
    assert!(vector.len() >= rotary);
    assert_eq!(cos.len(), half);
    assert_eq!(sin.len(), half);
    for index in 0..half {
        let a = vector[index];
        let b = vector[index + half];
        vector[index] = a * cos[index] - b * sin[index];
        vector[index + half] = b * cos[index] + a * sin[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn scales(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&f16_from_f32(*value).to_le_bytes());
        }
        bytes
    }

    /// Round an `f32` to the nearest `f16`, keeping it an `f32`, so the test data
    /// is exactly what the container would hold.
    fn f16_from_f32(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = (bits >> 23) & 0xff;
        let mantissa = bits & 0x007f_ffff;
        if exponent == 0xff {
            return sign | 0x7c00;
        }
        let biased = exponent as i32 - 127 + 15;
        if biased >= 0x1f {
            return sign | 0x7c00;
        }
        if biased <= 0 {
            // Below the f16 subnormal range: the tests do not rely on this.
            return sign;
        }
        // Round half to even on the 13 dropped bits.
        let round = (mantissa >> 13) & 1;
        let dropped = mantissa & 0x1fff;
        let mut half = mantissa >> 13;
        if dropped > 0x1000 || (dropped == 0x1000 && round == 1) {
            half += 1;
            if half == 0x400 {
                return sign | (((biased + 1) as u16) << 10);
            }
        }
        sign | ((biased as u16) << 10) | half as u16
    }

    #[test]
    fn four_bit_values_are_stored_offset_by_eight() {
        // 0x98 = high nibble 9, low nibble 8 -> q = +1 and 0.
        // 0x07 = high nibble 0, low nibble 7 -> q = -8 and -1.
        let mut out = [0.0f32; 4];
        dequantize_scalar(4, 4, &scales(&[0.5]), &[0x98, 0x07], &mut out);
        assert_eq!(out, [0.0, 0.5, -0.5, -4.0]);
    }

    #[test]
    fn two_and_three_bit_layouts_match_the_documented_bits() {
        let mut out = [0.0f32; 4];
        // 0b11_10_01_00 = fields 0, 1, 2, 3 -> q = -2, -1, 0, 1.
        dequantize_scalar(2, 4, &scales(&[1.0]), &[0b1110_0100], &mut out);
        assert_eq!(out, [-2.0, -1.0, 0.0, 1.0]);

        let mut out = [0.0f32; 8];
        // Values 7,6,5,4,3,2,1,0 in a 24-bit little-endian word -> q = 3..-4.
        let word: u32 = (7) | (6 << 3) | (5 << 6) | (4 << 9) | (3 << 12) | (2 << 15) | (1 << 18);
        let bytes = [word as u8, (word >> 8) as u8, (word >> 16) as u8];
        dequantize_scalar(3, 8, &scales(&[2.0]), &bytes, &mut out);
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -8.0]);
    }

    #[test]
    fn eight_bit_values_are_offset_by_one_hundred_twenty_eight() {
        let mut out = [0.0f32; 4];
        dequantize_scalar(8, 4, &scales(&[0.25]), &[0, 128, 129, 255], &mut out);
        assert_eq!(out, [-32.0, 0.0, 0.25, 31.75]);
    }

    #[test]
    fn the_vector_path_agrees_with_the_portable_one() {
        // A group of 32 with every nibble value represented, twice over.
        let mut integers = Vec::new();
        for index in 0..16u8 {
            integers.push(index | ((15 - index) << 4));
        }
        integers.extend_from_slice(&integers.clone());
        let scale = scales(&[0.0625, 1.5]);
        let mut slow = [0.0f32; 64];
        let mut fast = [0.0f32; 64];
        dequantize_scalar(4, 32, &scale, &integers, &mut slow);
        dequantize(4, 32, &scale, &integers, &mut fast);
        for (a, b) in slow.iter().zip(&fast) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
        }
    }
}

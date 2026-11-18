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

/// The kernels ask [`crate::cpu`] for the instruction sets they are allowed to
/// use; a target with none of them compiles the question out.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
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
///
/// Only the vector expansion uses it: the scalar path subtracts the offset
/// directly.
#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
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
    #[cfg(target_arch = "aarch64")]
    if cpu::features().neon && bits != 2 && bits != 3 {
        // SAFETY: the feature test above established NEON support, and `check`
        // has already validated the arguments.
        unsafe { neon::dequantize(bits, group, scales, integers, out) };
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

/// `sum_i a[i] * b[i]`, four accumulator chains, AVX2 where available.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::cpu::features().avx2_fma && a.len() >= 32 {
        // SAFETY: the feature test above established AVX2 and FMA support.
        return unsafe { dot_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::features().neon && a.len() >= 32 {
        // SAFETY: the feature test above, and `assert_eq!` has established that
        // both slices are the same length.
        return unsafe { neon::dot(a, b) };
    }
    dot_scalar(a, b)
}

/// Portable dot product: four accumulator chains.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
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

/// # Safety
/// The caller must have established AVX2 and FMA support.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    let vectors = a.len() / LANES;
    let vectorised = vectors * LANES;
    // SAFETY: the vector loop stays inside `vectorised`, a whole number of
    // eight-element vectors within both slices.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut index = 0;
        while index + 2 * LANES <= vectorised {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(index)),
                _mm256_loadu_ps(b.as_ptr().add(index)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(index + LANES)),
                _mm256_loadu_ps(b.as_ptr().add(index + LANES)),
                acc1,
            );
            index += 2 * LANES;
        }
        if index < vectorised {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(index)),
                _mm256_loadu_ps(b.as_ptr().add(index)),
                acc0,
            );
        }
        let mut total = horizontal_sum(_mm256_add_ps(acc0, acc1));
        for tail in vectorised..a.len() {
            total += *a.get_unchecked(tail) * *b.get_unchecked(tail);
        }
        total
    }
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
            if rows > 1 {
                quantised_linear_panel(
                    x, rows, weight, scales, integers, columns, group, groups, bias, out,
                );
                return;
            }
            // One token: nothing to reuse a slab for, so the values are unpacked
            // straight into the products.
            let source = &x[..columns];
            let features = crate::cpu::features();
            // Each kernel unpacks a whole number of values per round, so a group
            // that is not a multiple of that round falls back to the scalar
            // kernel rather than leaving a tail unread.
            let vectorised = (features.avx2_fma && group % 32 == 0 && columns % 2 == 0)
                || (features.neon && group % 16 == 0 && columns % 2 == 0);
            let dot_rows = |target: &mut [f32], first: usize| {
                for (offset, slot) in target.iter_mut().enumerate() {
                    let output = first + offset;
                    let value = quantised_dot(
                        integers,
                        scales,
                        source,
                        output,
                        columns,
                        group,
                        groups,
                        vectorised,
                    );
                    *slot = match bias {
                        Some(bias) => value + bias[output],
                        None => value,
                    };
                }
            };
            let macs = weight.rows * columns;
            let parts = crate::parallel::threads().min(weight.rows);
            if parts > 1 && macs >= PARALLEL_MACS {
                let per_part = weight.rows.div_ceil(parts);
                crate::parallel::pool().chunks_mut(per_part, out, &|index, chunk| {
                    dot_rows(chunk, index * per_part)
                });
            } else {
                dot_rows(out, 0);
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

/// Below this many multiply-adds a layer runs on one thread: the rendezvous costs
/// more than the split saves.
const PARALLEL_MACS: usize = 1 << 20;

/// How many output rows share one pass over the input.
const PANEL_ROWS: usize = 32;

/// How many columns one pass covers. A slab of the input for 32 rows and the
/// widest layer is a few tens of kilobytes, which stays in cache while the block
/// walks its rows.
const PANEL_COLUMNS: usize = 128;

/// `out[t, o] = sum_i x[t, i] * weight[o, i]` for a quantised weight.
///
/// The weight's bytes are read once per call and expanded one slab of columns at
/// a time; that slab of the input is then reused by every output row of a block.
/// Reading the whole input once per output row instead is what makes the naive
/// loop spend most of its time refetching it.
///
/// Blocks are accumulated output-major, so a worker owns a contiguous slice, and
/// transposed on the way out. The slab boundaries depend on the column count
/// alone, so every batch size produces the same bits: splitting a prompt into
/// two calls changes nothing. A single token is a different kernel (see
/// [`quantised_dot`]) and agrees to within the association error.
#[allow(clippy::too_many_arguments)]
fn quantised_linear_panel(
    x: &[f32],
    rows: usize,
    weight: &crate::weights::Matrix,
    scales: &[u8],
    integers: &[u8],
    columns: usize,
    group: usize,
    groups: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let outputs = weight.rows;
    let mut collected = vec![0.0f32; outputs * rows];
    let blocks = outputs.div_ceil(PANEL_ROWS);
    let parts = crate::parallel::threads().min(blocks);
    let work = rows as u64 * outputs as u64 * columns as u64;
    if parts > 1 && work >= PARALLEL_MACS as u64 {
        let per_part = blocks.div_ceil(parts);
        crate::parallel::pool().chunks_mut(
            per_part * PANEL_ROWS * rows,
            &mut collected,
            &|index, chunk| {
                let first = index * per_part * PANEL_ROWS;
                let count = chunk.len() / rows;
                panel_rows(
                    x, rows, scales, integers, columns, group, groups, first, count, chunk,
                );
            },
        );
    } else {
        panel_rows(
            x,
            rows,
            scales,
            integers,
            columns,
            group,
            groups,
            0,
            outputs,
            &mut collected,
        );
    }
    for output in 0..outputs {
        let bias = bias.map(|bias| bias[output]).unwrap_or(0.0);
        for token in 0..rows {
            out[token * outputs + output] = collected[output * rows + token] + bias;
        }
    }
}

/// Accumulate `count` output rows starting at `first` into `results`, which holds
/// `count * rows` values in output-major order.
#[allow(clippy::too_many_arguments)]
fn panel_rows(
    x: &[f32],
    rows: usize,
    scales: &[u8],
    integers: &[u8],
    columns: usize,
    group: usize,
    groups: usize,
    first: usize,
    count: usize,
    results: &mut [f32],
) {
    assert_eq!(results.len(), count * rows, "panel shape");
    assert_eq!(columns % group, 0, "a row of {columns} does not fill groups of {group}");
    for slot in results.iter_mut() {
        *slot = 0.0;
    }
    // The unit of work is a few rows' slabs against every token: the slabs are a
    // few hundred bytes and stay in L1, and one call covers the whole token axis
    // so the call overhead is amortised over a thousand multiply-adds.
    //
    // Several rows at a time is what makes the input pay: a token's vector is
    // loaded once and feeds every row of the group, so the load traffic per
    // product falls with the group size. Four rows needs four weight registers,
    // two input registers and eight accumulators, which is inside the sixteen
    // the register file has.
    let slab = PANEL_COLUMNS.min(columns);
    let mut values = vec![0.0f32; 4 * slab];
    let mut slab_start = 0;
    while slab_start < columns {
        let width = (columns - slab_start).min(slab);
        let mut row = 0;
        while row < count {
            let remaining = count - row;
            let rows_here = if remaining >= 4 {
                4
            } else if remaining >= 2 {
                2
            } else {
                1
            };
            for offset in 0..rows_here {
                let output = first + row + offset;
                let integer_base = output * columns / 2 + slab_start / 2;
                let scale_base = 2 * (output * groups + slab_start / group);
                dequantize(
                    4,
                    group,
                    &scales[scale_base..scale_base + 2 * width / group],
                    &integers[integer_base..integer_base + width / 2],
                    &mut values[offset * slab..offset * slab + width],
                );
            }
            let target = &mut results[row * rows..(row + rows_here) * rows];
            match rows_here {
                4 => accumulate_slab_quad(
                    &values[..width],
                    &values[slab..slab + width],
                    &values[2 * slab..2 * slab + width],
                    &values[3 * slab..3 * slab + width],
                    x,
                    columns,
                    slab_start,
                    target,
                ),
                2 => accumulate_slab_pair(
                    &values[..width],
                    &values[slab..slab + width],
                    x,
                    columns,
                    slab_start,
                    target,
                ),
                _ => accumulate_slab(&values[..width], x, columns, slab_start, target),
            }
            row += rows_here;
        }
        slab_start += width;
    }
}

/// Add `row · x[token, slab]` into `results` for every token.
///
/// One call covers the whole token axis of a slab. Calling a kernel per token
/// instead costs a prologue, a feature test and a `vzeroupper` for sixteen
/// multiply-adds, and that overhead was three times the work for a 128-value
/// slab.
fn accumulate_slab(row: &[f32], x: &[f32], stride: usize, offset: usize, results: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::cpu::features().avx2_fma && row.len() >= 32 {
        // SAFETY: the feature test above established AVX2 and FMA support.
        unsafe { accumulate_slab_avx2(row, x, stride, offset, results) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::features().neon && row.len() >= 32 {
        // SAFETY: the feature test above established NEON support; the caller
        // has already checked that `x` holds one row per result.
        unsafe { neon::accumulate_slab(row, x, stride, offset, results) };
        return;
    }
    for (token, slot) in results.iter_mut().enumerate() {
        let base = token * stride + offset;
        *slot += dot(row, &x[base..base + row.len()]);
    }
}

/// # Safety
/// The caller must have established AVX2 and FMA support. `x` must hold
/// `results.len()` rows of `stride` values, each at least `offset + row.len()`
/// long.
///
/// Each token gets two accumulator chains, reduced in the order [`dot_avx2`] uses,
/// so a token produces the same bits on its own as in a batch. Eight chains across
/// four tokens is what keeps the unit busy: one chain per token would leave it
/// idle three cycles out of four.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_slab_avx2(
    row: &[f32],
    x: &[f32],
    stride: usize,
    offset: usize,
    results: &mut [f32],
) {
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    let columns = row.len();
    let vectorised = columns / LANES * LANES;
    let tokens = results.len();
    let base = row.as_ptr();
    let mut token = 0;
    while token + 4 <= tokens {
        // SAFETY: the four token rows start inside `x`; the loads below stay
        // inside `columns` of each.
        let (first, second, third, fourth) = unsafe {
            let first = x.as_ptr().add(token * stride + offset);
            (first, first.add(stride), first.add(2 * stride), first.add(3 * stride))
        };
        // Eight accumulators, written out one per register: an array here stayed
        // in memory and cost four times the throughput.
        let mut low0 = _mm256_setzero_ps();
        let mut low1 = _mm256_setzero_ps();
        let mut low2 = _mm256_setzero_ps();
        let mut low3 = _mm256_setzero_ps();
        let mut high0 = _mm256_setzero_ps();
        let mut high1 = _mm256_setzero_ps();
        let mut high2 = _mm256_setzero_ps();
        let mut high3 = _mm256_setzero_ps();
        let mut index = 0;
        // SAFETY: every load stays inside `vectorised <= columns` of the row and
        // of each token's row.
        unsafe {
            while index + 2 * LANES <= vectorised {
                let weight0 = _mm256_loadu_ps(base.add(index));
                let weight1 = _mm256_loadu_ps(base.add(index + LANES));
                low0 = _mm256_fmadd_ps(weight0, _mm256_loadu_ps(first.add(index)), low0);
                high0 = _mm256_fmadd_ps(
                    weight1,
                    _mm256_loadu_ps(first.add(index + LANES)),
                    high0,
                );
                low1 = _mm256_fmadd_ps(weight0, _mm256_loadu_ps(second.add(index)), low1);
                high1 = _mm256_fmadd_ps(
                    weight1,
                    _mm256_loadu_ps(second.add(index + LANES)),
                    high1,
                );
                low2 = _mm256_fmadd_ps(weight0, _mm256_loadu_ps(third.add(index)), low2);
                high2 = _mm256_fmadd_ps(
                    weight1,
                    _mm256_loadu_ps(third.add(index + LANES)),
                    high2,
                );
                low3 = _mm256_fmadd_ps(weight0, _mm256_loadu_ps(fourth.add(index)), low3);
                high3 = _mm256_fmadd_ps(
                    weight1,
                    _mm256_loadu_ps(fourth.add(index + LANES)),
                    high3,
                );
                index += 2 * LANES;
            }
            if index < vectorised {
                let weight = _mm256_loadu_ps(base.add(index));
                low0 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(first.add(index)), low0);
                low1 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(second.add(index)), low1);
                low2 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(third.add(index)), low2);
                low3 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(fourth.add(index)), low3);
                index += LANES;
            }
            results[token] += horizontal_sum(_mm256_add_ps(low0, high0));
            results[token + 1] += horizontal_sum(_mm256_add_ps(low1, high1));
            results[token + 2] += horizontal_sum(_mm256_add_ps(low2, high2));
            results[token + 3] += horizontal_sum(_mm256_add_ps(low3, high3));
            for tail in index..columns {
                let weight = *base.add(tail);
                results[token] += weight * *first.add(tail);
                results[token + 1] += weight * *second.add(tail);
                results[token + 2] += weight * *third.add(tail);
                results[token + 3] += weight * *fourth.add(tail);
            }
        }
        token += 4;
    }
    // The last tokens have no quartet to fill; `dot` sums in the same order, so
    // they still get the same bits.
    while token < tokens {
        let base = token * stride + offset;
        results[token] += dot(row, &x[base..base + columns]);
        token += 1;
    }
}

/// Two rows' slabs against every token.
///
/// Every input vector is loaded once and used by both rows, which halves the
/// loads per product. The results hold the first row's tokens and then the
/// second's.
fn accumulate_slab_pair(
    row0: &[f32],
    row1: &[f32],
    x: &[f32],
    stride: usize,
    offset: usize,
    results: &mut [f32],
) {
    assert_eq!(row0.len(), row1.len(), "pair of slabs");
    let tokens = results.len() / 2;
    assert_eq!(results.len(), 2 * tokens, "pair of results");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::cpu::features().avx2_fma && row0.len() >= 32 {
        // SAFETY: the feature test above established AVX2 and FMA support.
        unsafe { accumulate_slab_pair_avx2(row0, row1, x, stride, offset, results) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::features().neon && row0.len() >= 32 {
        // SAFETY: the feature test above established NEON support; `results`
        // holds two runs by the assertion above.
        unsafe { neon::accumulate_slab_pair(row0, row1, x, stride, offset, results) };
        return;
    }
    for token in 0..tokens {
        let base = token * stride + offset;
        results[token] += dot(row0, &x[base..base + row0.len()]);
        results[tokens + token] += dot(row1, &x[base..base + row1.len()]);
    }
}

/// # Safety
/// The caller must have established AVX2 and FMA support, and `results` must hold
/// two runs of one value per token.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_slab_pair_avx2(
    row0: &[f32],
    row1: &[f32],
    x: &[f32],
    stride: usize,
    offset: usize,
    results: &mut [f32],
) {
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    let columns = row0.len();
    let vectorised = columns / LANES * LANES;
    let tokens = results.len() / 2;
    let base0 = row0.as_ptr();
    let base1 = row1.as_ptr();
    let mut token = 0;
    while token + 4 <= tokens {
        // SAFETY: the four token rows start inside `x`; the loads below stay
        // inside `columns` of each.
        let (first, second, third, fourth) = unsafe {
            let first = x.as_ptr().add(token * stride + offset);
            (first, first.add(stride), first.add(2 * stride), first.add(3 * stride))
        };
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();
        let mut index = 0;
        // SAFETY: every load stays inside `vectorised <= columns` of both rows and
        // of each token's row.
        unsafe {
            while index < vectorised {
                let weight0 = _mm256_loadu_ps(base0.add(index));
                let weight1 = _mm256_loadu_ps(base1.add(index));
                let input0 = _mm256_loadu_ps(first.add(index));
                let input1 = _mm256_loadu_ps(second.add(index));
                let input2 = _mm256_loadu_ps(third.add(index));
                let input3 = _mm256_loadu_ps(fourth.add(index));
                a0 = _mm256_fmadd_ps(weight0, input0, a0);
                b0 = _mm256_fmadd_ps(weight1, input0, b0);
                a1 = _mm256_fmadd_ps(weight0, input1, a1);
                b1 = _mm256_fmadd_ps(weight1, input1, b1);
                a2 = _mm256_fmadd_ps(weight0, input2, a2);
                b2 = _mm256_fmadd_ps(weight1, input2, b2);
                a3 = _mm256_fmadd_ps(weight0, input3, a3);
                b3 = _mm256_fmadd_ps(weight1, input3, b3);
                index += LANES;
            }
            results[token] += horizontal_sum(a0);
            results[token + 1] += horizontal_sum(a1);
            results[token + 2] += horizontal_sum(a2);
            results[token + 3] += horizontal_sum(a3);
            results[tokens + token] += horizontal_sum(b0);
            results[tokens + token + 1] += horizontal_sum(b1);
            results[tokens + token + 2] += horizontal_sum(b2);
            results[tokens + token + 3] += horizontal_sum(b3);
            for tail in index..columns {
                let weight0 = *base0.add(tail);
                let weight1 = *base1.add(tail);
                results[token] += weight0 * *first.add(tail);
                results[token + 1] += weight0 * *second.add(tail);
                results[token + 2] += weight0 * *third.add(tail);
                results[token + 3] += weight0 * *fourth.add(tail);
                results[tokens + token] += weight1 * *first.add(tail);
                results[tokens + token + 1] += weight1 * *second.add(tail);
                results[tokens + token + 2] += weight1 * *third.add(tail);
                results[tokens + token + 3] += weight1 * *fourth.add(tail);
            }
        }
        token += 4;
    }
    while token < tokens {
        let base = token * stride + offset;
        results[token] += dot(row0, &x[base..base + columns]);
        results[tokens + token] += dot(row1, &x[base..base + columns]);
        token += 1;
    }
}

/// Four rows' slabs against every token, two tokens per pass.
///
/// The results hold the four rows one after another, each with one value per
/// token.
#[allow(clippy::too_many_arguments)]
fn accumulate_slab_quad(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    row3: &[f32],
    x: &[f32],
    stride: usize,
    offset: usize,
    results: &mut [f32],
) {
    assert_eq!(row0.len(), row1.len(), "quad of slabs");
    assert_eq!(row0.len(), row2.len(), "quad of slabs");
    assert_eq!(row0.len(), row3.len(), "quad of slabs");
    let tokens = results.len() / 4;
    assert_eq!(results.len(), 4 * tokens, "quad of results");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::cpu::features().avx2_fma && row0.len() >= 32 {
        // SAFETY: the feature test above established AVX2 and FMA support.
        unsafe {
            accumulate_slab_quad_avx2(row0, row1, row2, row3, x, stride, offset, results)
        };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::features().neon && row0.len() >= 32 {
        // SAFETY: the feature test above established NEON support; `results`
        // holds four runs by the assertion above.
        unsafe {
            neon::accumulate_slab_quad(row0, row1, row2, row3, x, stride, offset, results)
        };
        return;
    }
    for token in 0..tokens {
        let base = token * stride + offset;
        results[token] += dot(row0, &x[base..base + row0.len()]);
        results[tokens + token] += dot(row1, &x[base..base + row1.len()]);
        results[2 * tokens + token] += dot(row2, &x[base..base + row2.len()]);
        results[3 * tokens + token] += dot(row3, &x[base..base + row3.len()]);
    }
}

/// # Safety
/// The caller must have established AVX2 and FMA support, and `results` must hold
/// four runs of one value per token.
#[allow(clippy::too_many_arguments)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_slab_quad_avx2(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    row3: &[f32],
    x: &[f32],
    stride: usize,
    offset: usize,
    results: &mut [f32],
) {
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    let columns = row0.len();
    let vectorised = columns / LANES * LANES;
    let tokens = results.len() / 4;
    let base0 = row0.as_ptr();
    let base1 = row1.as_ptr();
    let base2 = row2.as_ptr();
    let base3 = row3.as_ptr();
    let mut token = 0;
    while token + 2 <= tokens {
        // SAFETY: the two token rows start inside `x`, and the loads below stay
        // inside `vectorised <= columns` of each.
        let (first, second) = unsafe {
            let first = x.as_ptr().add(token * stride + offset);
            (first, first.add(stride))
        };
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut d0 = _mm256_setzero_ps();
        let mut d1 = _mm256_setzero_ps();
        let mut index = 0;
        // SAFETY: every load stays inside `vectorised <= columns` of each row.
        unsafe {
            while index < vectorised {
                let input0 = _mm256_loadu_ps(first.add(index));
                let input1 = _mm256_loadu_ps(second.add(index));
                let weight0 = _mm256_loadu_ps(base0.add(index));
                let weight1 = _mm256_loadu_ps(base1.add(index));
                let weight2 = _mm256_loadu_ps(base2.add(index));
                let weight3 = _mm256_loadu_ps(base3.add(index));
                a0 = _mm256_fmadd_ps(weight0, input0, a0);
                a1 = _mm256_fmadd_ps(weight0, input1, a1);
                b0 = _mm256_fmadd_ps(weight1, input0, b0);
                b1 = _mm256_fmadd_ps(weight1, input1, b1);
                c0 = _mm256_fmadd_ps(weight2, input0, c0);
                c1 = _mm256_fmadd_ps(weight2, input1, c1);
                d0 = _mm256_fmadd_ps(weight3, input0, d0);
                d1 = _mm256_fmadd_ps(weight3, input1, d1);
                index += LANES;
            }
            results[token] += horizontal_sum(a0);
            results[token + 1] += horizontal_sum(a1);
            results[tokens + token] += horizontal_sum(b0);
            results[tokens + token + 1] += horizontal_sum(b1);
            results[2 * tokens + token] += horizontal_sum(c0);
            results[2 * tokens + token + 1] += horizontal_sum(c1);
            results[3 * tokens + token] += horizontal_sum(d0);
            results[3 * tokens + token + 1] += horizontal_sum(d1);
            for tail in index..columns {
                let input0 = *first.add(tail);
                let input1 = *second.add(tail);
                results[token] += *base0.add(tail) * input0;
                results[token + 1] += *base0.add(tail) * input1;
                results[tokens + token] += *base1.add(tail) * input0;
                results[tokens + token + 1] += *base1.add(tail) * input1;
                results[2 * tokens + token] += *base2.add(tail) * input0;
                results[2 * tokens + token + 1] += *base2.add(tail) * input1;
                results[3 * tokens + token] += *base3.add(tail) * input0;
                results[3 * tokens + token + 1] += *base3.add(tail) * input1;
            }
        }
        token += 2;
    }
    while token < tokens {
        let base = token * stride + offset;
        results[token] += dot(row0, &x[base..base + columns]);
        results[tokens + token] += dot(row1, &x[base..base + columns]);
        results[2 * tokens + token] += dot(row2, &x[base..base + columns]);
        results[3 * tokens + token] += dot(row3, &x[base..base + columns]);
        token += 1;
    }
}

/// One output row of a four-bit matrix-vector product, unpacked in registers.
///
/// The rows of a batch go through [`panel_rows`], which expands a slab of values
/// once and reuses it for every token. A single token has nothing to reuse, and
/// expanding 871 million values per pass to read each of them once is what the
/// fused loop avoids: it turns the packed bytes straight into products.
///
/// Each group contributes `(sum of q * x) * scale`, with the integer part summed
/// in `f32`: `q` is exact, so the only rounding is the sum itself.
#[allow(clippy::too_many_arguments)]
fn quantised_dot(
    integers: &[u8],
    scales: &[u8],
    source: &[f32],
    output: usize,
    columns: usize,
    group: usize,
    groups: usize,
    vectorised: bool,
) -> f32 {
    let scale_base = output * groups;
    let integer_base = output * columns / 2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if vectorised {
        // SAFETY: `vectorised` is only true after the AVX2/FMA test.
        return unsafe {
            quantised_dot_avx2(integers, scales, source, scale_base, integer_base, group, groups)
        };
    }
    #[cfg(target_arch = "aarch64")]
    if vectorised {
        // SAFETY: `vectorised` is only true after the NEON test and the group
        // size check that test includes.
        return unsafe {
            neon::quantised_dot(integers, scales, source, scale_base, integer_base, group, groups)
        };
    }
    let _ = vectorised;
    let mut total = 0.0f32;
    for group_index in 0..groups {
        let scale = scale_of(scales, scale_base + group_index);
        let bytes = &integers[integer_base + group_index * group / 2
            ..integer_base + (group_index + 1) * group / 2];
        let source_base = group_index * group;
        // Two chains, one per nibble lane: the low nibbles and the high nibbles
        // are independent sums until the end.
        let mut low_sum = 0.0f32;
        let mut high_sum = 0.0f32;
        for (pair, byte) in bytes.iter().enumerate() {
            let index = source_base + 2 * pair;
            low_sum += f32::from(i16::from(byte & 0x0f) - 8) * source[index];
            high_sum += f32::from(i16::from(byte >> 4) - 8) * source[index + 1];
        }
        total += (low_sum + high_sum) * scale;
    }
    total
}

/// # Safety
/// The caller must have established AVX2 and FMA support.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantised_dot_avx2(
    integers: &[u8],
    scales: &[u8],
    source: &[f32],
    scale_base: usize,
    integer_base: usize,
    group: usize,
    groups: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let mut total = 0.0f32;
    // SAFETY: every load stays inside the group's bytes and the group's values;
    // `group % 32 == 0` was checked by the caller, so the loop covers whole
    // 16-value vectors.
    unsafe {
        let mask = _mm_set1_epi8(0x0f);
        let eight = _mm256_set1_epi32(8);
        let bytes_per_group = group / 2;
        for group_index in 0..groups {
            let scale = scale_of(scales, scale_base + group_index);
            let base = integers.as_ptr().add(integer_base + group_index * bytes_per_group);
            let values = source.as_ptr().add(group_index * group);
            let mut first = _mm256_setzero_ps();
            let mut second = _mm256_setzero_ps();
            let mut byte = 0;
            while byte < bytes_per_group {
                let raw = _mm_loadl_epi64(base.add(byte).cast::<__m128i>());
                let low = _mm_and_si128(raw, mask);
                // Shifting 16-bit lanes moves a byte's high nibble into the next
                // lane's low byte; masking afterwards keeps exactly the high
                // nibble of each byte.
                let high = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
                let interleaved = _mm_unpacklo_epi8(low, high);
                let low_values = _mm256_cvtepu8_epi32(interleaved);
                let high_values = _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(interleaved, interleaved));
                let low_values = _mm256_cvtepi32_ps(_mm256_sub_epi32(low_values, eight));
                let high_values = _mm256_cvtepi32_ps(_mm256_sub_epi32(high_values, eight));
                first = _mm256_fmadd_ps(low_values, _mm256_loadu_ps(values.add(2 * byte)), first);
                second = _mm256_fmadd_ps(
                    high_values,
                    _mm256_loadu_ps(values.add(2 * byte + 8)),
                    second,
                );
                byte += 8;
            }
            total += horizontal_sum(_mm256_add_ps(first, second)) * scale;
        }
    }
    total
}

/// Sum the eight lanes of one vector.
///
/// # Safety
/// Requires AVX, which the caller established by testing AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let mut sum = _mm_add_ps(low, high);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    _mm_cvtss_f32(sum)
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
    // Rows are independent, so the split cannot change any row's result.
    crate::parallel::pool().chunks_mut(width, out, &|row, target| {
        let source = &x[row * width..(row + 1) * width];
        let gate = &gate[row * width..(row + 1) * width];
        let mean_square = source.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        for index in 0..width {
            target[index] = weight[index] * source[index] * inverse * silu(gate[index]);
        }
    });
}

/// Divide a row by the square root of its sum of squares plus `eps`.
pub fn l2_normalize(row: &mut [f32], eps: f32) {
    let sum: f32 = row.iter().map(|value| value * value).sum();
    let inverse = 1.0 / (sum + eps).sqrt();
    for value in row.iter_mut() {
        *value *= inverse;
    }
}

/// Multiply every value by `factor`, in place.
pub fn scale_all(values: &mut [f32], factor: f32) {
    for value in values.iter_mut() {
        *value *= factor;
    }
}

/// `out[i] = sum_row state[row, i] * weight[row]` over a `rows x columns` block.
///
/// The rows are walked outside and the columns inside, so every pass through
/// `state` is a contiguous run. Walking the columns outside instead reads
/// `state[row * columns + i]` with a stride of `columns`, one cache line per
/// element.
///
/// A given `out[i]` is still the sum over rows in increasing order, so this
/// produces the same bits as the strided form.
pub fn transposed_matvec(
    state: &[f32],
    rows: usize,
    columns: usize,
    weight: &[f32],
    out: &mut [f32],
) {
    assert_eq!(state.len(), rows * columns, "state shape");
    assert_eq!(weight.len(), rows, "weight shape");
    assert_eq!(out.len(), columns, "output shape");
    out.fill(0.0);
    for (row, weight) in weight.iter().enumerate().take(rows) {
        let values = &state[row * columns..(row + 1) * columns];
        let weight = *weight;
        for (slot, value) in out.iter_mut().zip(values) {
            *slot += value * weight;
        }
    }
}

/// `state[row, i] += weight[row] * delta[i]` over a `rows x columns` block.
pub fn outer_add(state: &mut [f32], rows: usize, columns: usize, weight: &[f32], delta: &[f32]) {
    assert_eq!(state.len(), rows * columns, "state shape");
    assert_eq!(weight.len(), rows, "weight shape");
    assert_eq!(delta.len(), columns, "delta shape");
    for (row, weight) in weight.iter().enumerate().take(rows) {
        let values = &mut state[row * columns..(row + 1) * columns];
        let weight = *weight;
        for (slot, value) in values.iter_mut().zip(delta) {
            *slot += weight * value;
        }
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

/// Advanced SIMD kernels for 64-bit ARM.
///
/// Four `f32` lanes per register instead of eight, and a fused multiply-add, so
/// the shapes mirror the AVX2 kernels with half the width: the same association
/// order per token, the same split of a group into independent halves, and the
/// same exactness properties.
///
/// 32-bit ARM has no path here. NEON is optional before ARMv8, and the kernels
/// would have to fit sixteen registers rather than thirty-two, which is a
/// different layout rather than the same one with narrower vectors.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// Store sixteen scaled values: `(byte - offset) * factor`.
    ///
    /// # Safety
    /// `target` must have room for sixteen `f32` values.
    #[inline]
    unsafe fn store_scaled(
        values: uint8x16_t,
        target: *mut f32,
        offset: float32x4_t,
        factor: float32x4_t,
    ) {
        // SAFETY: the widening and narrowing helpers below take and return
        // register values only; the single store writes four lanes at a time
        // into the sixteen the caller reserved.
        unsafe {
            let low = vmovl_u8(vget_low_u8(values));
            let high = vmovl_u8(vget_high_u8(values));
            let quarters = [
                vmovl_u16(vget_low_u16(low)),
                vmovl_u16(vget_high_u16(low)),
                vmovl_u16(vget_low_u16(high)),
                vmovl_u16(vget_high_u16(high)),
            ];
            for (index, quarter) in quarters.into_iter().enumerate() {
                let value = vmulq_f32(vsubq_f32(vcvtq_f32_u32(quarter), offset), factor);
                vst1q_f32(target.add(4 * index), value);
            }
        }
    }

    /// Expand grouped symmetric quantisation into `out`.
    ///
    /// # Safety
    /// The caller must have established NEON support and checked the arguments
    /// with [`super::check`].
    pub unsafe fn dequantize(bits: u8, group: usize, scales: &[u8], integers: &[u8], out: &mut [f32]) {
        let groups = out.len() / group;
        let bytes_per_group = group * bits as usize / 8;
        for index in 0..groups {
            let scale = super::scale_of(scales, index);
            let source = &integers[index * bytes_per_group..(index + 1) * bytes_per_group];
            let target = &mut out[index * group..(index + 1) * group];
            // SAFETY: every load reads sixteen bytes the guard below shows are
            // inside `source`, and every store writes the sixteen values those
            // bytes decode to, which are inside `target`.
            unsafe {
                let factor = vdupq_n_f32(scale);
                match bits {
                    8 => {
                        let offset = vdupq_n_f32(128.0);
                        let mut byte = 0;
                        while byte + 16 <= source.len() {
                            let raw = vld1q_u8(source.as_ptr().add(byte));
                            store_scaled(raw, target.as_mut_ptr().add(byte), offset, factor);
                            byte += 16;
                        }
                        for tail in byte..source.len() {
                            target[tail] = f32::from(i16::from(source[tail]) - 128) * scale;
                        }
                    }
                    4 => {
                        let mask = vdupq_n_u8(0x0f);
                        let offset = vdupq_n_f32(8.0);
                        let mut byte = 0;
                        while byte + 16 <= source.len() {
                            let raw = vld1q_u8(source.as_ptr().add(byte));
                            let low = vandq_u8(raw, mask);
                            // A byte's high nibble; the mask keeps the shifted
                            // bits from leaking across lanes.
                            let high = vshrq_n_u8::<4>(raw);
                            let first = vzip1q_u8(low, high);
                            let second = vzip2q_u8(low, high);
                            store_scaled(first, target.as_mut_ptr().add(2 * byte), offset, factor);
                            store_scaled(
                                second,
                                target.as_mut_ptr().add(2 * byte + 16),
                                offset,
                                factor,
                            );
                            byte += 16;
                        }
                        if byte < source.len() {
                            super::unpack_group(4, &source[byte..], &mut target[2 * byte..], scale);
                        }
                    }
                    other => panic!("no vector path for {other}-bit values"),
                }
            }
        }
    }

    /// `sum_i a[i] * b[i]`, four accumulator chains.
    ///
    /// # Safety
    /// The caller must have established NEON support; both slices have the same
    /// length.
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        const LANES: usize = 4;
        let vectorised = a.len() / LANES * LANES;
        // SAFETY: the loops stay inside `vectorised`, a whole number of
        // four-element vectors within both slices; the tail is indexed with
        // bounds checks.
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut index = 0;
            while index + 4 * LANES <= vectorised {
                acc0 = vfmaq_f32(
                    acc0,
                    vld1q_f32(a.as_ptr().add(index)),
                    vld1q_f32(b.as_ptr().add(index)),
                );
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(a.as_ptr().add(index + LANES)),
                    vld1q_f32(b.as_ptr().add(index + LANES)),
                );
                acc2 = vfmaq_f32(
                    acc2,
                    vld1q_f32(a.as_ptr().add(index + 2 * LANES)),
                    vld1q_f32(b.as_ptr().add(index + 2 * LANES)),
                );
                acc3 = vfmaq_f32(
                    acc3,
                    vld1q_f32(a.as_ptr().add(index + 3 * LANES)),
                    vld1q_f32(b.as_ptr().add(index + 3 * LANES)),
                );
                index += 4 * LANES;
            }
            while index < vectorised {
                acc0 = vfmaq_f32(
                    acc0,
                    vld1q_f32(a.as_ptr().add(index)),
                    vld1q_f32(b.as_ptr().add(index)),
                );
                index += LANES;
            }
            let mut total =
                vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            for tail in vectorised..a.len() {
                total += *a.get_unchecked(tail) * *b.get_unchecked(tail);
            }
            total
        }
    }

    /// One output row of a four-bit matrix-vector product, unpacked in registers.
    ///
    /// # Safety
    /// The caller must have established NEON support and checked that
    /// `group % 16 == 0`.
    pub unsafe fn quantised_dot(
        integers: &[u8],
        scales: &[u8],
        source: &[f32],
        scale_base: usize,
        integer_base: usize,
        group: usize,
        groups: usize,
    ) -> f32 {
        let bytes_per_group = group / 2;
        let mut total = 0.0f32;
        // SAFETY: each round reads eight bytes of the group and the sixteen
        // values they decode to, both of which the caller's group length covers.
        unsafe {
            let eight = vdupq_n_f32(8.0);
            let mask = vdup_n_u8(0x0f);
            for group_index in 0..groups {
                let scale = super::scale_of(scales, scale_base + group_index);
                let base = integers.as_ptr().add(integer_base + group_index * bytes_per_group);
                let values = source.as_ptr().add(group_index * group);
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);
                let mut byte = 0;
                while byte + 8 <= bytes_per_group {
                    let raw = vld1_u8(base.add(byte));
                    let low = vand_u8(raw, mask);
                    let high = vshr_n_u8::<4>(raw);
                    // Interleaving the two nibbles puts each byte's two values
                    // next to each other, which is the order the values are
                    // stored in. A 64-bit vector holds eight values, so the pair
                    // of nibble vectors splits into two halves.
                    let first = vzip1_u8(low, high);
                    let second = vzip2_u8(low, high);
                    let first_wide = vmovl_u8(first);
                    let second_wide = vmovl_u8(second);
                    let quarters = [
                        vmovl_u16(vget_low_u16(first_wide)),
                        vmovl_u16(vget_high_u16(first_wide)),
                        vmovl_u16(vget_low_u16(second_wide)),
                        vmovl_u16(vget_high_u16(second_wide)),
                    ];
                    acc0 = vfmaq_f32(
                        acc0,
                        vsubq_f32(vcvtq_f32_u32(quarters[0]), eight),
                        vld1q_f32(values.add(2 * byte)),
                    );
                    acc1 = vfmaq_f32(
                        acc1,
                        vsubq_f32(vcvtq_f32_u32(quarters[1]), eight),
                        vld1q_f32(values.add(2 * byte + 4)),
                    );
                    acc2 = vfmaq_f32(
                        acc2,
                        vsubq_f32(vcvtq_f32_u32(quarters[2]), eight),
                        vld1q_f32(values.add(2 * byte + 8)),
                    );
                    acc3 = vfmaq_f32(
                        acc3,
                        vsubq_f32(vcvtq_f32_u32(quarters[3]), eight),
                        vld1q_f32(values.add(2 * byte + 12)),
                    );
                    byte += 8;
                }
                total += vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
                    * scale;
            }
        }
        total
    }

    /// Add `row · x[token, slab]` into `results` for every token, four at a time.
    ///
    /// # Safety
    /// The caller must have established NEON support; `x` must hold
    /// `results.len()` rows of `stride` values, each at least `offset + row.len()`
    /// long.
    pub unsafe fn accumulate_slab(
        row: &[f32],
        x: &[f32],
        stride: usize,
        offset: usize,
        results: &mut [f32],
    ) {
        const LANES: usize = 4;
        let columns = row.len();
        let vectorised = columns / LANES * LANES;
        let tokens = results.len();
        let base = row.as_ptr();
        let mut token = 0;
        // SAFETY: the token rows start inside `x`, and every load below stays
        // inside `vectorised <= columns` of the row and of each token's row.
        unsafe {
            while token + 4 <= tokens {
                let first = x.as_ptr().add(token * stride + offset);
                let second = first.add(stride);
                let third = first.add(2 * stride);
                let fourth = first.add(3 * stride);
                let mut low0 = vdupq_n_f32(0.0);
                let mut low1 = vdupq_n_f32(0.0);
                let mut low2 = vdupq_n_f32(0.0);
                let mut low3 = vdupq_n_f32(0.0);
                let mut high0 = vdupq_n_f32(0.0);
                let mut high1 = vdupq_n_f32(0.0);
                let mut high2 = vdupq_n_f32(0.0);
                let mut high3 = vdupq_n_f32(0.0);
                let mut index = 0;
                while index + 2 * LANES <= vectorised {
                    let weight0 = vld1q_f32(base.add(index));
                    let weight1 = vld1q_f32(base.add(index + LANES));
                    low0 = vfmaq_f32(low0, weight0, vld1q_f32(first.add(index)));
                    high0 = vfmaq_f32(high0, weight1, vld1q_f32(first.add(index + LANES)));
                    low1 = vfmaq_f32(low1, weight0, vld1q_f32(second.add(index)));
                    high1 = vfmaq_f32(high1, weight1, vld1q_f32(second.add(index + LANES)));
                    low2 = vfmaq_f32(low2, weight0, vld1q_f32(third.add(index)));
                    high2 = vfmaq_f32(high2, weight1, vld1q_f32(third.add(index + LANES)));
                    low3 = vfmaq_f32(low3, weight0, vld1q_f32(fourth.add(index)));
                    high3 = vfmaq_f32(high3, weight1, vld1q_f32(fourth.add(index + LANES)));
                    index += 2 * LANES;
                }
                if index < vectorised {
                    let weight = vld1q_f32(base.add(index));
                    low0 = vfmaq_f32(low0, weight, vld1q_f32(first.add(index)));
                    low1 = vfmaq_f32(low1, weight, vld1q_f32(second.add(index)));
                    low2 = vfmaq_f32(low2, weight, vld1q_f32(third.add(index)));
                    low3 = vfmaq_f32(low3, weight, vld1q_f32(fourth.add(index)));
                    index += LANES;
                }
                results[token] += vaddvq_f32(vaddq_f32(low0, high0));
                results[token + 1] += vaddvq_f32(vaddq_f32(low1, high1));
                results[token + 2] += vaddvq_f32(vaddq_f32(low2, high2));
                results[token + 3] += vaddvq_f32(vaddq_f32(low3, high3));
                for tail in index..columns {
                    let weight = *base.add(tail);
                    results[token] += weight * *first.add(tail);
                    results[token + 1] += weight * *second.add(tail);
                    results[token + 2] += weight * *third.add(tail);
                    results[token + 3] += weight * *fourth.add(tail);
                }
                token += 4;
            }
            // Whatever is left has no quartet to fill; `dot` sums in the same
            // order, so those tokens still get the same bits.
            while token < tokens {
                let start = token * stride + offset;
                results[token] += dot(row, &x[start..start + columns]);
                token += 1;
            }
        }
    }

    /// Two rows' slabs against every token, four tokens per pass.
    ///
    /// # Safety
    /// The caller must have established NEON support, and `results` must hold
    /// two runs of one value per token.
    pub unsafe fn accumulate_slab_pair(
        row0: &[f32],
        row1: &[f32],
        x: &[f32],
        stride: usize,
        offset: usize,
        results: &mut [f32],
    ) {
        const LANES: usize = 4;
        let columns = row0.len();
        let vectorised = columns / LANES * LANES;
        let tokens = results.len() / 2;
        let base0 = row0.as_ptr();
        let base1 = row1.as_ptr();
        let mut token = 0;
        // SAFETY: the token rows start inside `x`; every load stays inside
        // `vectorised <= columns` of both rows and of each token's row.
        unsafe {
            while token + 4 <= tokens {
                let first = x.as_ptr().add(token * stride + offset);
                let second = first.add(stride);
                let third = first.add(2 * stride);
                let fourth = first.add(3 * stride);
                let mut a0 = vdupq_n_f32(0.0);
                let mut a1 = vdupq_n_f32(0.0);
                let mut a2 = vdupq_n_f32(0.0);
                let mut a3 = vdupq_n_f32(0.0);
                let mut b0 = vdupq_n_f32(0.0);
                let mut b1 = vdupq_n_f32(0.0);
                let mut b2 = vdupq_n_f32(0.0);
                let mut b3 = vdupq_n_f32(0.0);
                let mut index = 0;
                while index < vectorised {
                    let weight0 = vld1q_f32(base0.add(index));
                    let weight1 = vld1q_f32(base1.add(index));
                    let input0 = vld1q_f32(first.add(index));
                    let input1 = vld1q_f32(second.add(index));
                    let input2 = vld1q_f32(third.add(index));
                    let input3 = vld1q_f32(fourth.add(index));
                    a0 = vfmaq_f32(a0, weight0, input0);
                    b0 = vfmaq_f32(b0, weight1, input0);
                    a1 = vfmaq_f32(a1, weight0, input1);
                    b1 = vfmaq_f32(b1, weight1, input1);
                    a2 = vfmaq_f32(a2, weight0, input2);
                    b2 = vfmaq_f32(b2, weight1, input2);
                    a3 = vfmaq_f32(a3, weight0, input3);
                    b3 = vfmaq_f32(b3, weight1, input3);
                    index += LANES;
                }
                results[token] += vaddvq_f32(a0);
                results[token + 1] += vaddvq_f32(a1);
                results[token + 2] += vaddvq_f32(a2);
                results[token + 3] += vaddvq_f32(a3);
                results[tokens + token] += vaddvq_f32(b0);
                results[tokens + token + 1] += vaddvq_f32(b1);
                results[tokens + token + 2] += vaddvq_f32(b2);
                results[tokens + token + 3] += vaddvq_f32(b3);
                for tail in index..columns {
                    let weight0 = *base0.add(tail);
                    let weight1 = *base1.add(tail);
                    results[token] += weight0 * *first.add(tail);
                    results[token + 1] += weight0 * *second.add(tail);
                    results[token + 2] += weight0 * *third.add(tail);
                    results[token + 3] += weight0 * *fourth.add(tail);
                    results[tokens + token] += weight1 * *first.add(tail);
                    results[tokens + token + 1] += weight1 * *second.add(tail);
                    results[tokens + token + 2] += weight1 * *third.add(tail);
                    results[tokens + token + 3] += weight1 * *fourth.add(tail);
                }
                token += 4;
            }
            while token < tokens {
                let start = token * stride + offset;
                results[token] += dot(row0, &x[start..start + columns]);
                results[tokens + token] += dot(row1, &x[start..start + columns]);
                token += 1;
            }
        }
    }

    /// Four rows' slabs against every token, two tokens per pass.
    ///
    /// # Safety
    /// The caller must have established NEON support, and `results` must hold
    /// four runs of one value per token.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn accumulate_slab_quad(
        row0: &[f32],
        row1: &[f32],
        row2: &[f32],
        row3: &[f32],
        x: &[f32],
        stride: usize,
        offset: usize,
        results: &mut [f32],
    ) {
        const LANES: usize = 4;
        let columns = row0.len();
        let vectorised = columns / LANES * LANES;
        let tokens = results.len() / 4;
        let base0 = row0.as_ptr();
        let base1 = row1.as_ptr();
        let base2 = row2.as_ptr();
        let base3 = row3.as_ptr();
        let mut token = 0;
        // SAFETY: the token rows start inside `x`; every load stays inside
        // `vectorised <= columns` of each row.
        unsafe {
            while token + 2 <= tokens {
                let first = x.as_ptr().add(token * stride + offset);
                let second = first.add(stride);
                let mut a0 = vdupq_n_f32(0.0);
                let mut a1 = vdupq_n_f32(0.0);
                let mut b0 = vdupq_n_f32(0.0);
                let mut b1 = vdupq_n_f32(0.0);
                let mut c0 = vdupq_n_f32(0.0);
                let mut c1 = vdupq_n_f32(0.0);
                let mut d0 = vdupq_n_f32(0.0);
                let mut d1 = vdupq_n_f32(0.0);
                let mut index = 0;
                while index < vectorised {
                    let input0 = vld1q_f32(first.add(index));
                    let input1 = vld1q_f32(second.add(index));
                    let weight0 = vld1q_f32(base0.add(index));
                    let weight1 = vld1q_f32(base1.add(index));
                    let weight2 = vld1q_f32(base2.add(index));
                    let weight3 = vld1q_f32(base3.add(index));
                    a0 = vfmaq_f32(a0, weight0, input0);
                    a1 = vfmaq_f32(a1, weight0, input1);
                    b0 = vfmaq_f32(b0, weight1, input0);
                    b1 = vfmaq_f32(b1, weight1, input1);
                    c0 = vfmaq_f32(c0, weight2, input0);
                    c1 = vfmaq_f32(c1, weight2, input1);
                    d0 = vfmaq_f32(d0, weight3, input0);
                    d1 = vfmaq_f32(d1, weight3, input1);
                    index += LANES;
                }
                results[token] += vaddvq_f32(a0);
                results[token + 1] += vaddvq_f32(a1);
                results[tokens + token] += vaddvq_f32(b0);
                results[tokens + token + 1] += vaddvq_f32(b1);
                results[2 * tokens + token] += vaddvq_f32(c0);
                results[2 * tokens + token + 1] += vaddvq_f32(c1);
                results[3 * tokens + token] += vaddvq_f32(d0);
                results[3 * tokens + token + 1] += vaddvq_f32(d1);
                for tail in index..columns {
                    let input0 = *first.add(tail);
                    let input1 = *second.add(tail);
                    results[token] += *base0.add(tail) * input0;
                    results[token + 1] += *base0.add(tail) * input1;
                    results[tokens + token] += *base1.add(tail) * input0;
                    results[tokens + token + 1] += *base1.add(tail) * input1;
                    results[2 * tokens + token] += *base2.add(tail) * input0;
                    results[2 * tokens + token + 1] += *base2.add(tail) * input1;
                    results[3 * tokens + token] += *base3.add(tail) * input0;
                    results[3 * tokens + token + 1] += *base3.add(tail) * input1;
                }
                token += 2;
            }
            while token < tokens {
                let start = token * stride + offset;
                results[token] += dot(row0, &x[start..start + columns]);
                results[tokens + token] += dot(row1, &x[start..start + columns]);
                results[2 * tokens + token] += dot(row2, &x[start..start + columns]);
                results[3 * tokens + token] += dot(row3, &x[start..start + columns]);
                token += 1;
            }
        }
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

    /// Expansion is exact, so the two paths must agree bit for bit — including
    /// over a buffer long enough to run the vector rounds more than once.
    fn check_expansion_is_bit_exact() {
        let patterns = scales(&[0.0625, 1.5, 0.25, 2.0, 0.5, 0.125, 1.0, 3.0]);
        // Four-bit: 256 values in eight groups of 32, so 128 bytes and eight
        // rounds of sixteen.
        let integers: Vec<u8> = (0..128u32).map(|index| (index * 37 % 256) as u8).collect();
        let mut slow = [0.0f32; 256];
        let mut fast = [0.0f32; 256];
        dequantize_scalar(4, 32, &patterns, &integers, &mut slow);
        dequantize(4, 32, &patterns, &integers, &mut fast);
        for (a, b) in slow.iter().zip(&fast) {
            assert_eq!(a.to_bits(), b.to_bits(), "four-bit: {a} vs {b}");
        }

        // Eight-bit: the same eight groups, one byte per value.
        let integers: Vec<u8> = (0..256u32).map(|index| (index * 53 % 256) as u8).collect();
        dequantize_scalar(8, 32, &patterns, &integers, &mut slow);
        dequantize(8, 32, &patterns, &integers, &mut fast);
        for (a, b) in slow.iter().zip(&fast) {
            assert_eq!(a.to_bits(), b.to_bits(), "eight-bit: {a} vs {b}");
        }
    }

    /// A reduction has an association order, so the two paths agree to within
    /// the error of summing the same terms in a different order.
    fn check_dot_matches_the_portable_one() {
        for length in [32usize, 33, 64, 129, 1024] {
            let left: Vec<f32> = (0..length).map(|index| (index as f32 * 0.37).sin()).collect();
            let right: Vec<f32> = (0..length).map(|index| (index as f32 * 0.11).cos()).collect();
            let slow = dot_scalar(&left, &right);
            let fast = dot(&left, &right);
            let tolerance = 1e-5 * slow.abs().max(1.0);
            assert!(
                (slow - fast).abs() <= tolerance,
                "{length} values: {slow} vs {fast}"
            );
        }
    }

    fn check_quantised_dot_matches_the_portable_one() {
        let columns = 64;
        let group = 32;
        let groups = columns / group;
        let integers: Vec<u8> = (0..columns / 2)
            .map(|index| (index * 37 % 256) as u8)
            .collect();
        let scale = scales(&[0.03125, 0.5]);
        let source: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.23).sin()).collect();
        let slow = quantised_dot(&integers, &scale, &source, 0, columns, group, groups, false);
        let fast = quantised_dot(&integers, &scale, &source, 0, columns, group, groups, true);
        let tolerance = 1e-5 * slow.abs().max(1.0);
        assert!((slow - fast).abs() <= tolerance, "{slow} vs {fast}");
    }

    /// The batch kernels split a batch into a four-token run and a remainder of
    /// one to three tokens, so a length that is not a multiple of four exercises
    /// both halves of the same call.
    fn check_slab_kernels_match_the_portable_ones() {
        let columns = 68; // not a multiple of the four-lane vectors either
        let tokens = 7;
        let stride = columns;
        let row: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.37).sin()).collect();
        let x: Vec<f32> = (0..tokens * stride)
            .map(|index| (index as f32 * 0.11).cos())
            .collect();
        let close = |slow: f32, fast: f32| (slow - fast).abs() <= 1e-5 * slow.abs().max(1.0);

        // One row.
        let mut slow = vec![0.0f32; tokens];
        let mut fast = vec![0.0f32; tokens];
        for token in 0..tokens {
            let base = token * stride;
            slow[token] += dot_scalar(&row, &x[base..base + columns]);
        }
        accumulate_slab(&row, &x, stride, 0, &mut fast);
        for token in 0..tokens {
            assert!(close(slow[token], fast[token]), "one row, token {token}");
        }

        // Two rows, each with a run of its own.
        let second: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.53).sin()).collect();
        let mut slow = vec![0.0f32; 2 * tokens];
        let mut fast = vec![0.0f32; 2 * tokens];
        for token in 0..tokens {
            let base = token * stride;
            slow[token] += dot_scalar(&row, &x[base..base + columns]);
            slow[tokens + token] += dot_scalar(&second, &x[base..base + columns]);
        }
        accumulate_slab_pair(&row, &second, &x, stride, 0, &mut fast);
        for index in 0..2 * tokens {
            assert!(close(slow[index], fast[index]), "pair, index {index}");
        }

        // Four rows against two tokens at a time, so an odd batch also lands in
        // the single-token remainder.
        let third: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.71).sin()).collect();
        let fourth: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.91).cos()).collect();
        let mut slow = vec![0.0f32; 4 * tokens];
        let mut fast = vec![0.0f32; 4 * tokens];
        for token in 0..tokens {
            let base = token * stride;
            slow[token] += dot_scalar(&row, &x[base..base + columns]);
            slow[tokens + token] += dot_scalar(&second, &x[base..base + columns]);
            slow[2 * tokens + token] += dot_scalar(&third, &x[base..base + columns]);
            slow[3 * tokens + token] += dot_scalar(&fourth, &x[base..base + columns]);
        }
        accumulate_slab_quad(&row, &second, &third, &fourth, &x, stride, 0, &mut fast);
        for index in 0..4 * tokens {
            assert!(close(slow[index], fast[index]), "quad, index {index}");
        }
    }

    /// The vector kernels of a 64-bit ARM build, against the portable ones.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_neon_kernels_match_the_portable_ones() {
        check_expansion_is_bit_exact();
        check_dot_matches_the_portable_one();
        check_quantised_dot_matches_the_portable_one();
        check_slab_kernels_match_the_portable_ones();
    }

    /// The vector kernels of an x86 build, against the portable ones.
    ///
    /// A machine without AVX2 and FMA runs the portable kernel on both sides, so
    /// there is nothing to compare; the guard names that instead of passing
    /// silently as if the wide kernels had been checked.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn the_avx2_kernels_match_the_portable_ones() {
        if !crate::cpu::features().avx2_fma {
            return;
        }
        check_expansion_is_bit_exact();
        check_dot_matches_the_portable_one();
        check_quantised_dot_matches_the_portable_one();
        check_slab_kernels_match_the_portable_ones();
    }
}

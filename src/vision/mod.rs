//! The vision tower: image in, one embedding per merged 2x2 patch out.
//!
//! Twelve blocks of the shape the reference calls a `Qwen3_5VisionBlock`, then a
//! merger that folds each 2x2 block of patches into one vector of the text
//! model's width. Everything here is a plain matrix product, one LayerNorm, one
//! axial rotation and a non-causal attention inside a frame; there is no cache,
//! because the tower sees the image once.
//!
//! The arithmetic follows the reference point by point — see `docs/op-spec.md`
//! §9 — and `dev/dump_vision.py` records the checkpoints it is compared against.

use crate::ops;
use crate::weights::{self, Matrix};

pub mod preprocess;

pub use preprocess::{smart_resize, Error};

/// The constants the tower's shapes are built from.
pub mod constants {
    use crate::weights::manifest;

    /// Side of one patch, in pixels.
    pub const PATCH: usize = manifest::VISION_PATCH;
    /// Frames one patch covers.
    pub const TEMPORAL: usize = manifest::VISION_TEMPORAL_PATCH;
    /// Patches one merge block folds, per side.
    pub const MERGE: usize = manifest::VISION_MERGE;
    /// The multiple every resized side is a multiple of.
    pub const PATCH_STRIDE: usize = PATCH * MERGE;
    /// Pixel count a resized image is scaled up to.
    pub const SHORTEST_EDGE: usize = 65_536;
    /// Pixel count a resized image is scaled down to.
    pub const LONGEST_EDGE: usize = 16_777_216;
    /// Values one patch vector holds: channels, frames, patch side twice.
    pub const PATCH_VALUES: usize = 3 * TEMPORAL * PATCH * PATCH;
    /// Width of the tower's hidden state.
    pub const HIDDEN: usize = manifest::VISION_HIDDEN;
    /// Heads per block.
    pub const HEADS: usize = manifest::VISION_HEADS;
    /// Values per head.
    pub const HEAD_DIM: usize = manifest::VISION_HEAD_DIM;
    /// Width of the feed-forward layer.
    pub const INTERMEDIATE: usize = manifest::VISION_INTERMEDIATE;
    /// Blocks in the tower.
    pub const DEPTH: usize = manifest::VISION_DEPTH;
    /// Width of the merger's hidden layer.
    pub const MERGER_HIDDEN: usize = manifest::VISION_MERGER_HIDDEN;
    /// Width of one merged token, which is the text model's.
    pub const OUT_HIDDEN: usize = manifest::VISION_OUT_HIDDEN;
    /// Side of the learned position grid.
    pub const GRID_SIDE: usize = manifest::VISION_GRID_SIDE;
    /// Rows in the learned position table.
    pub const POSITIONS: usize = manifest::VISION_POSITIONS;
    /// The rotation base.
    pub const ROPE_THETA: f64 = manifest::VISION_ROPE_THETA;
    /// Epsilon of both LayerNorms.
    pub const NORM_EPS: f32 = manifest::VISION_NORM_EPS;
}

/// Where a token sits: the patch grid of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// Frames; one for an image.
    pub temporal: usize,
    /// Patches down.
    pub height: usize,
    /// Patches across.
    pub width: usize,
}

impl Grid {
    /// Tokens the frame contributes.
    pub fn patches(&self) -> usize {
        self.temporal * self.height * self.width
    }
}

/// The tower's answer.
pub struct Encoded {
    /// One row per merged 2x2 block of patches.
    pub values: Vec<f32>,
    /// Rows in `values`.
    pub tokens: usize,
    /// The patch grid the image was resized to.
    pub grid: Grid,
    /// The resized sides, in pixels.
    pub resized: (usize, usize),
}

fn tensor(name: &str) -> weights::Tensor {
    weights::expect(name)
}

fn matrix(name: String) -> Matrix {
    tensor(&name).matrix()
}

fn vector(name: String) -> Vec<f32> {
    tensor(&name).values()
}

/// The patch embedding as a `[hidden, patch values]` matrix.
///
/// The stored shape is five-dimensional (`[out, channels, frames, patch, patch]`),
/// and the last axis alone is not the width, so the generic [`Matrix`] accessor
/// cannot describe it.
fn patch_embed() -> Matrix {
    let tensor = tensor("model.visual.patch_embed.proj.weight");
    Matrix {
        rows: constants::HIDDEN,
        columns: constants::PATCH_VALUES,
        data: tensor.data,
        kind: tensor.spec.kind,
        group: tensor.spec.group,
    }
}

fn linear(x: &[f32], rows: usize, name: String, bias: String, out: &mut [f32]) {
    let weight = matrix(name);
    let bias = vector(bias);
    ops::linear(x, rows, &weight, Some(&bias), out);
}

/// LayerNorm over the last axis, with weight and bias.
fn layer_norm(x: &[f32], rows: usize, weight: &[f32], bias: &[f32], out: &mut [f32]) {
    let width = constants::HIDDEN;
    for row in 0..rows {
        let source = &x[row * width..(row + 1) * width];
        let target = &mut out[row * width..(row + 1) * width];
        let mean = source.iter().sum::<f32>() / width as f32;
        let variance =
            source.iter().map(|value| (value - mean) * (value - mean)).sum::<f32>() / width as f32;
        let inverse = 1.0 / (variance + constants::NORM_EPS).sqrt();
        for index in 0..width {
            target[index] = (source[index] - mean) * inverse * weight[index] + bias[index];
        }
    }
}

/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, the tower's activation.
fn gelu_tanh(value: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_57;
    let cube = value * value * value;
    0.5 * value * (1.0 + (COEFFICIENT * (value + 0.044_715 * cube)).tanh())
}

/// `0.5 x (1 + erf(x / sqrt 2))`, the merger's activation.
fn gelu_exact(value: f32) -> f32 {
    0.5 * value * (1.0 + erf(value * std::f32::consts::FRAC_1_SQRT_2))
}

/// `erf`, by the Abramowitz–Stegun rational form (7.1.26).
///
/// Its absolute error is below 1.5e-7, which is the only place this tower departs
/// from the reference's `erf`; it lands in the merger's activation, whose result
/// is then mixed by one more matrix product.
fn erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

/// The `(row, column)` a patch index stands for.
///
/// The processor emits patches in merge-block order: the block column moves
/// fastest of the block coordinates, and inside a block the patch row is the
/// slower of the two. The measurement behind this is one lit 16x16 source block
/// per run, which shows exactly which output patches respond to it.
pub fn patch_position(index: usize, blocks_across: usize) -> (usize, usize) {
    let merge = constants::MERGE;
    let in_column = index % merge;
    let in_row = (index / merge) % merge;
    let block_column = (index / (merge * merge)) % blocks_across;
    let block_row = index / (merge * merge * blocks_across);
    (
        block_row * merge + in_row,
        block_column * merge + in_column,
    )
}

/// The position embedding of every patch, interpolated from the learned 48x48
/// grid.
///
/// Every axis is sampled at `index * (side - 1) / (size - 1)` — `align_corners`,
/// so the corners of the image land on the corners of the table — with two taps
/// and the weight `1 - |distance|` clamped at zero. The weights are *not*
/// renormalised at the edges, unlike the resize's.
fn position_embedding(grid: Grid) -> Vec<f32> {
    let side = constants::GRID_SIDE;
    let table = tensor("model.visual.pos_embed.weight").matrix();
    let width = constants::HIDDEN;
    let patches = grid.height * grid.width;
    let plan = |size: usize| -> Vec<(Vec<usize>, Vec<f32>)> {
        (0..size)
            .map(|index| {
                let source = if size > 1 {
                    index as f32 * (side - 1) as f32 / (size - 1) as f32
                } else {
                    0.0
                };
                let floor = source.floor();
                let mut taps = Vec::with_capacity(2);
                let mut weights = Vec::with_capacity(2);
                for offset in 0..2 {
                    let tap = floor + offset as f32;
                    let clamped = tap.max(0.0).min((side - 1) as f32) as usize;
                    let weight = (1.0 - (source - floor - offset as f32).abs()).max(0.0);
                    taps.push(clamped);
                    weights.push(weight);
                }
                (taps, weights)
            })
            .collect()
    };
    let rows = plan(grid.height);
    let columns = plan(grid.width);
    let mut out = vec![0.0f32; patches * width];
    let mut row = vec![0.0f32; width];
    for index in 0..patches {
        let (patch_row, patch_column) = patch_position(index, grid.width / constants::MERGE);
        let target = &mut out[index * width..(index + 1) * width];
        for (tap_row, weight_row) in rows[patch_row].0.iter().zip(&rows[patch_row].1) {
            for (tap_column, weight_column) in
                columns[patch_column].0.iter().zip(&columns[patch_column].1)
            {
                let weight = weight_row * weight_column;
                if weight == 0.0 {
                    continue;
                }
                ops::row_values(&table, tap_row * side + tap_column, &mut row);
                for (slot, value) in target.iter_mut().zip(&row) {
                    *slot += weight * value;
                }
            }
        }
    }
    out
}

/// The rotation's cosine and sine for every patch, `heads / 2` values each.
///
/// The frequencies are `1 / theta^(2i / spatial)` for `i < head_dim / 4`; a patch's
/// `(row, column)` are both expressed in them, and the two sixteen-value runs are
/// what the `rotate_half` pairing then reads as one thirty-two-value half.
fn rope(grid: Grid) -> Vec<(Vec<f32>, Vec<f32>)> {
    let spatial = constants::HEAD_DIM / 2;
    let frequencies: Vec<f64> = (0..constants::HEAD_DIM / 4)
        .map(|index| 1.0 / constants::ROPE_THETA.powf(2.0 * index as f64 / spatial as f64))
        .collect();
    let patches = grid.height * grid.width;
    let mut out = Vec::with_capacity(patches);
    for index in 0..patches {
        let (patch_row, patch_column) = patch_position(index, grid.width / constants::MERGE);
        let mut cos = Vec::with_capacity(spatial);
        let mut sin = Vec::with_capacity(spatial);
        for (coordinate, angle) in [(patch_row, &frequencies), (patch_column, &frequencies)] {
            for frequency in angle {
                let value = coordinate as f64 * frequency;
                cos.push(value.cos() as f32);
                sin.push(value.sin() as f32);
            }
        }
        out.push((cos, sin));
    }
    out
}

/// Attention inside each frame: every patch sees every patch of its own frame.
fn attend(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    grid: Grid,
    out: &mut [f32],
) {
    let heads = constants::HEADS;
    let head_dim = constants::HEAD_DIM;
    let patches = grid.patches();
    let scaling = 1.0 / (head_dim as f32).sqrt();
    let frame = grid.height * grid.width;
    let mut scores = vec![0.0f32; frame];
    for head in 0..heads {
        for token in 0..patches {
            let frame_start = token / frame * frame;
            let query_base = (token * heads + head) * head_dim;
            for position in 0..frame {
                let other = (frame_start + position) * heads * head_dim + head * head_dim;
                scores[position] = ops::dot(
                    &query[query_base..query_base + head_dim],
                    &key[other..other + head_dim],
                ) * scaling;
            }
            ops::softmax(&mut scores);
            let target = &mut out[query_base..query_base + head_dim];
            target.fill(0.0);
            for (position, weight) in scores.iter().enumerate() {
                let other = (frame_start + position) * heads * head_dim + head * head_dim;
                for index in 0..head_dim {
                    target[index] += weight * value[other + index];
                }
            }
        }
    }
}

/// Encode one RGB8 image.
///
/// `pixels` is row-major, three bytes per pixel, `height * width * 3` bytes.
pub fn encode(pixels: &[u8], height: usize, width: usize) -> Result<Encoded, Error> {
    let (values, patch_h, patch_w, resized_height, resized_width) =
        preprocess::pixel_values(pixels, height, width)?;
    let grid = Grid {
        temporal: 1,
        height: patch_h,
        width: patch_w,
    };
    let mut encoded = encode_patches(&values, grid);
    encoded.resized = (resized_height, resized_width);
    Ok(encoded)
}

/// Encode patch vectors that were produced elsewhere.
///
/// The rows are [`preprocess::pixel_values`]'s layout: one row per patch, in
/// merge-block order, `(channel, frame, patch row, patch column)` within a row.
pub fn encode_patches(values: &[f32], grid: Grid) -> Encoded {
    let patches = grid.patches();
    let hidden = constants::HIDDEN;
    assert_eq!(values.len(), patches * constants::PATCH_VALUES, "patch rows");

    // --- patch embedding and the position table
    // The stored weight is five-dimensional, so this one call goes through the
    // matrix shape built for it rather than through the name lookup.
    let mut x = vec![0.0f32; patches * hidden];
    let patch_values = patch_embed();
    let patch_bias = vector("model.visual.patch_embed.proj.bias".to_string());
    ops::linear(&values, patches, &patch_values, Some(&patch_bias), &mut x);
    let positions = position_embedding(grid);
    for (slot, position) in x.iter_mut().zip(&positions) {
        *slot += position;
    }

    // --- the rotation is the same for every block
    let rotation = rope(grid);

    let head_dim = constants::HEAD_DIM;
    let rotary = head_dim;
    let half = rotary / 2;
    let mut normalized = vec![0.0f32; patches * hidden];
    let mut packed = vec![0.0f32; patches * 3 * hidden];
    let mut query = vec![0.0f32; patches * hidden];
    let mut key = vec![0.0f32; patches * hidden];
    let mut value = vec![0.0f32; patches * hidden];
    let mut attended = vec![0.0f32; patches * hidden];
    let mut projected = vec![0.0f32; patches * hidden];
    let mut expanded = vec![0.0f32; patches * constants::INTERMEDIATE];
    let mut activated = vec![0.0f32; patches * constants::INTERMEDIATE];

    for block in 0..constants::DEPTH {
        let prefix = format!("model.visual.blocks.{block}");
        let norm1_weight = vector(format!("{prefix}.norm1.weight"));
        let norm1_bias = vector(format!("{prefix}.norm1.bias"));
        layer_norm(&x, patches, &norm1_weight, &norm1_bias, &mut normalized);

        linear(
            &normalized,
            patches,
            format!("{prefix}.attn.qkv.weight"),
            format!("{prefix}.attn.qkv.bias"),
            &mut packed,
        );
        // [token, 3, head, head_dim] -> three [token, head, head_dim] runs
        for token in 0..patches {
            for part in 0..3 {
                let source = token * 3 * hidden + part * hidden;
                let target = token * hidden;
                let destination = match part {
                    0 => &mut query,
                    1 => &mut key,
                    _ => &mut value,
                };
                destination[target..target + hidden]
                    .copy_from_slice(&packed[source..source + hidden]);
            }
        }
        for token in 0..patches {
            let (cos, sin) = &rotation[token];
            for head in 0..constants::HEADS {
                let base = token * hidden + head * head_dim;
                ops::apply_rope(&mut query[base..base + head_dim], rotary, &cos[..half], &sin[..half]);
                ops::apply_rope(&mut key[base..base + head_dim], rotary, &cos[..half], &sin[..half]);
            }
        }
        attend(&query, &key, &value, grid, &mut attended);
        linear(
            &attended,
            patches,
            format!("{prefix}.attn.proj.weight"),
            format!("{prefix}.attn.proj.bias"),
            &mut projected,
        );
        for (slot, update) in x.iter_mut().zip(&projected) {
            *slot += update;
        }

        let norm2_weight = vector(format!("{prefix}.norm2.weight"));
        let norm2_bias = vector(format!("{prefix}.norm2.bias"));
        layer_norm(&x, patches, &norm2_weight, &norm2_bias, &mut normalized);
        linear(
            &normalized,
            patches,
            format!("{prefix}.mlp.linear_fc1.weight"),
            format!("{prefix}.mlp.linear_fc1.bias"),
            &mut expanded,
        );
        for (slot, value) in activated.iter_mut().zip(&expanded) {
            *slot = gelu_tanh(*value);
        }
        linear(
            &activated,
            patches,
            format!("{prefix}.mlp.linear_fc2.weight"),
            format!("{prefix}.mlp.linear_fc2.bias"),
            &mut projected,
        );
        for (slot, update) in x.iter_mut().zip(&projected) {
            *slot += update;
        }
    }

    // --- the merger: normalise each patch, fold 2x2 of them into one row, then
    // two projections with the exact GELU between them.
    let merge = constants::MERGE;
    let merged = patches / (merge * merge);
    let merger_norm_weight = vector("model.visual.merger.norm.weight".to_string());
    let merger_norm_bias = vector("model.visual.merger.norm.bias".to_string());
    layer_norm(&x, patches, &merger_norm_weight, &merger_norm_bias, &mut normalized);
    let mut folded = vec![0.0f32; merged * constants::MERGER_HIDDEN];
    for index in 0..merged {
        let source = index * merge * merge * hidden;
        folded[index * constants::MERGER_HIDDEN..(index + 1) * constants::MERGER_HIDDEN]
            .copy_from_slice(&normalized[source..source + constants::MERGER_HIDDEN]);
    }
    let mut merger_hidden = vec![0.0f32; merged * constants::MERGER_HIDDEN];
    linear(
        &folded,
        merged,
        "model.visual.merger.linear_fc1.weight".to_string(),
        "model.visual.merger.linear_fc1.bias".to_string(),
        &mut merger_hidden,
    );
    for value in merger_hidden.iter_mut() {
        *value = gelu_exact(*value);
    }
    let mut out = vec![0.0f32; merged * constants::OUT_HIDDEN];
    linear(
        &merger_hidden,
        merged,
        "model.visual.merger.linear_fc2.weight".to_string(),
        "model.visual.merger.linear_fc2.bias".to_string(),
        &mut out,
    );

    Encoded {
        values: out,
        tokens: merged,
        grid,
        resized: (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_patch_order_is_merge_block_major() {
        // A 4x4 patch grid, so two merge blocks across. Index 0..3 are the first
        // block's four patches, then the next blocks along the row.
        assert_eq!(patch_position(0, 2), (0, 0));
        assert_eq!(patch_position(1, 2), (0, 1));
        assert_eq!(patch_position(2, 2), (1, 0));
        assert_eq!(patch_position(3, 2), (1, 1));
        assert_eq!(patch_position(4, 2), (0, 2));
        assert_eq!(patch_position(5, 2), (0, 3));
        assert_eq!(patch_position(8, 2), (2, 0));
    }

    #[test]
    fn the_exact_gelu_tracks_its_definition() {
        // gelu(0) = 0, and the two branches of `erf` agree at the origin.
        assert!(gelu_exact(0.0).abs() < 1e-7);
        // A value with a known answer: 0.5 * 1 * (1 + erf(1/sqrt2)) = 0.8413447.
        let value = gelu_exact(1.0);
        assert!((value - 0.841_344_7).abs() < 1e-5, "{value}");
    }
}

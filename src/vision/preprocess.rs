//! Image preprocessing: from an RGB8 buffer to the tower's patch vectors.
//!
//! The rule was measured against `Qwen2VLImageProcessorFast` and is pinned by
//! `dev/dump_vision.py`'s fixtures; the readings are in `dev/README.md`. Two
//! things about it are easy to get wrong and are part of the contract:
//!
//! * the resize is bicubic with the `a = -0.5` kernel (the one PIL uses), not the
//!   `a = -0.75` kernel torchvision's `F.interpolate` uses, and
//! * the two passes each round to a byte before the next one reads them, because
//!   the processor keeps its intermediate buffer in `uint8`. Skipping that
//!   rounding moves a saturated image by 14 levels.
//!
//! Sizes come out of the container's configuration: the multiple of 32 is
//! `patch_size * merge_size`, the pixel bounds are `size.shortest_edge` and
//! `size.longest_edge`.

use super::constants;

/// The prefix of every weight name in this tower.
pub const PREFIX: &str = "model.visual";

/// The largest ratio between the two sides the processor accepts.
const MAX_ASPECT: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The buffer does not hold `height * width * 3` bytes.
    WrongLength { expected: usize, actual: usize },
    /// One side is zero.
    Empty,
    /// The sides differ by more than 200:1.
    Aspect { height: usize, width: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::WrongLength { expected, actual } => {
                write!(formatter, "{actual} bytes for an image of {expected}")
            }
            Error::Empty => write!(formatter, "an image needs two positive sides"),
            Error::Aspect { height, width } => {
                write!(formatter, "the sides {height} and {width} differ by more than 200:1")
            }
        }
    }
}

impl std::error::Error for Error {}

/// The resized sides, both multiples of [`constants::PATCH_STRIDE`].
///
/// A side is scaled so that the pixel count lands in `[SHORTEST_EDGE,
/// LONGEST_EDGE]`, rounded to the nearest multiple of the stride first and then
/// adjusted with a floor (when too large) or a ceiling (when too small).
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize), Error> {
    if height == 0 || width == 0 {
        return Err(Error::Empty);
    }
    let (larger, smaller) = if height > width { (height, width) } else { (width, height) };
    if larger > smaller * MAX_ASPECT {
        return Err(Error::Aspect { height, width });
    }
    let stride = constants::PATCH_STRIDE;
    let round = |side: usize| ((side + stride / 2) / stride) * stride;
    let mut resized_height = round(height);
    let mut resized_width = round(width);
    let pixels = height * width;
    if resized_height * resized_width > constants::LONGEST_EDGE {
        let beta = ((pixels as f64) / constants::LONGEST_EDGE as f64).sqrt();
        resized_height = stride.max(((height as f64 / beta) as usize / stride) * stride);
        resized_width = stride.max(((width as f64 / beta) as usize / stride) * stride);
    } else if resized_height * resized_width < constants::SHORTEST_EDGE {
        let beta = (constants::SHORTEST_EDGE as f64 / pixels as f64).sqrt();
        resized_height = (height as f64 * beta / stride as f64).ceil() as usize * stride;
        resized_width = (width as f64 * beta / stride as f64).ceil() as usize * stride;
    }
    Ok((resized_height, resized_width))
}

/// The cubic kernel PIL uses for `BICUBIC`, with `a = -0.5`.
fn cubic(distance: f64) -> f64 {
    const A: f64 = -0.5;
    let x = distance.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// The taps and normalised weights of one axis.
///
/// The sample position is the half-pixel centre `(index + 0.5) * in / out - 0.5`;
/// the four taps are `floor(position) + {-1, 0, 1, 2}`, a tap outside the source
/// is dropped rather than clamped, and the kept weights are renormalised. That
/// last pair of choices is what the border measurement showed.
fn axis_plan(out_size: usize, in_size: usize) -> Vec<(Vec<usize>, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let mut plan = Vec::with_capacity(out_size);
    for index in 0..out_size {
        let center = (index as f64 + 0.5) * scale - 0.5;
        let base = center.floor();
        let mut taps = Vec::with_capacity(4);
        let mut weights = Vec::with_capacity(4);
        let mut total = 0.0;
        for offset in [-1.0, 0.0, 1.0, 2.0] {
            let tap = base + offset;
            if tap < 0.0 || tap >= in_size as f64 {
                continue;
            }
            let weight = cubic(center - tap);
            taps.push(tap as usize);
            weights.push(weight);
            total += weight;
        }
        for weight in weights.iter_mut() {
            *weight /= total;
        }
        plan.push((taps, weights));
    }
    plan
}

/// Round and clamp one pass, the way the processor's `uint8` buffer does.
fn store(value: f64) -> u8 {
    let rounded = value.round();
    if rounded <= 0.0 {
        0
    } else if rounded >= 255.0 {
        255
    } else {
        rounded as u8
    }
}

/// Resize an RGB8 image, returning `0..=255` values at the new size.
fn resize(
    pixels: &[u8],
    height: usize,
    width: usize,
    resized_height: usize,
    resized_width: usize,
) -> Vec<u8> {
    let rows = axis_plan(resized_height, height);
    let columns = axis_plan(resized_width, width);
    let mut horizontal = vec![0u8; height * resized_width * 3];
    for y in 0..height {
        let source = &pixels[y * width * 3..(y + 1) * width * 3];
        let target = &mut horizontal[y * resized_width * 3..(y + 1) * resized_width * 3];
        for (index, (taps, weights)) in columns.iter().enumerate() {
            for channel in 0..3 {
                let mut total = 0.0;
                for (tap, weight) in taps.iter().zip(weights) {
                    total += weight * f64::from(source[tap * 3 + channel]);
                }
                target[index * 3 + channel] = store(total);
            }
        }
    }
    let mut vertical = vec![0u8; resized_height * resized_width * 3];
    for (index, (taps, weights)) in rows.iter().enumerate() {
        let target = &mut vertical[index * resized_width * 3..(index + 1) * resized_width * 3];
        for channel in 0..3 {
            for x in 0..resized_width {
                let mut total = 0.0;
                for (tap, weight) in taps.iter().zip(weights) {
                    total += weight * f64::from(horizontal[(tap * resized_width + x) * 3 + channel]);
                }
                target[x * 3 + channel] = store(total);
            }
        }
    }
    vertical
}

/// The patch vector of one 16x16 (by two frames) tile.
///
/// The layout is `(channel, temporal, patch_row, patch_column)`, and an image has
/// one frame that is written twice — the temporal kernel is 2 wide and the
/// processor duplicates the frame to fill it.
const PATCH_VALUES: usize = constants::PATCH_VALUES;

/// Normalise and flatten an image into the tower's input.
///
/// Returns the values in patch order (merge-block major) and the patch grid.
pub fn pixel_values(
    pixels: &[u8],
    height: usize,
    width: usize,
) -> Result<(Vec<f32>, usize, usize, usize, usize), Error> {
    let expected = height * width * 3;
    if pixels.len() != expected {
        return Err(Error::WrongLength {
            expected,
            actual: pixels.len(),
        });
    }
    let (resized_height, resized_width) = smart_resize(height, width)?;
    let resized = resize(pixels, height, width, resized_height, resized_width);

    let patch = constants::PATCH;
    let merge = constants::MERGE;
    let patch_h = resized_height / patch;
    let patch_w = resized_width / patch;
    let mut values = vec![0.0f32; patch_h * patch_w * PATCH_VALUES];
    let frame_stride = resized_width * 3;
    for block_row in 0..patch_h / merge {
        for block_col in 0..patch_w / merge {
            for in_row in 0..merge {
                for in_col in 0..merge {
                    let row = block_row * merge + in_row;
                    let column = block_col * merge + in_col;
                    let index = ((block_row * (patch_w / merge) + block_col) * merge + in_row) * merge
                        + in_col;
                    let target = &mut values[index * PATCH_VALUES..(index + 1) * PATCH_VALUES];
                    // (channel, temporal, patch_row, patch_column)
                    let mut slot = 0;
                    for channel in 0..3 {
                        for temporal in 0..2 {
                            let _ = temporal;
                            for patch_row in 0..patch {
                                let source = (row * patch + patch_row) * frame_stride
                                    + column * patch * 3
                                    + channel;
                                for patch_column in 0..patch {
                                    // `/255` then `(x - mean) / std` with mean and
                                    // std both 0.5.
                                    let raw = f32::from(resized[source + patch_column * 3]);
                                    target[slot] = (raw / 255.0 - 0.5) / 0.5;
                                    slot += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((values, patch_h, patch_w, resized_height, resized_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stride_rounding_matches_the_reference() {
        // Measured: 100x70 -> 320x224 and 64x64 -> 256x256, both scaled up to the
        // shortest edge because the pixel count is below it.
        assert_eq!(smart_resize(100, 70), Ok((320, 224)));
        assert_eq!(smart_resize(64, 64), Ok((256, 256)));
        // 1024x768 is 786,432 pixels, between the bounds: only the stride rounds.
        assert_eq!(smart_resize(1024, 768), Ok((1024, 768)));
    }

    #[test]
    fn an_extreme_aspect_is_rejected() {
        assert_eq!(
            smart_resize(1000, 4),
            Err(Error::Aspect {
                height: 1000,
                width: 4
            })
        );
    }

    #[test]
    fn the_image_buffer_has_to_match_the_sides() {
        assert_eq!(
            pixel_values(&[0; 10], 2, 2),
            Err(Error::WrongLength {
                expected: 12,
                actual: 10
            })
        );
    }
}

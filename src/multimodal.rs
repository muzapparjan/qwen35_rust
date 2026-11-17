//! Prompts that carry an image: where the image tokens sit, what positions they
//! get, and how the tower's rows replace their embeddings.
//!
//! A prompt is a flat token sequence in which one run of [`IMAGE_PLACEHOLDER`]
//! placeholders stands for each image. The processor emits those placeholders and
//! the tower emits exactly one row per merged patch, so the two line up in order.
//! What is *not* a token index is the position: every image token carries the grid
//! coordinate of its merged patch, and the text after the image starts from the
//! image's height rather than from its token count. The reference calls this the
//! multimodal rotation; the rules are in `docs/op-spec.md` §8.3 and are measured
//! by `dev/dump_vision.py`'s end-to-end fixture.

use crate::model::{Cache, Model, Trace};
use crate::vision;
use crate::weights::manifest::IMAGE_TOKEN_ID;

pub const IMAGE_PLACEHOLDER: u32 = IMAGE_TOKEN_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The prompt has a different number of image runs than the caller supplied.
    ImageCount { runs: usize, given: usize },
    /// A run of placeholders is not the size the grid says it should be.
    RunLength {
        run: usize,
        expected: usize,
        actual: usize,
    },
    /// The tower's output does not match the grid.
    RowCount { expected: usize, actual: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ImageCount { runs, given } => {
                write!(formatter, "the prompt has {runs} image runs, {given} were given")
            }
            Error::RunLength {
                run,
                expected,
                actual,
            } => write!(
                formatter,
                "image run {run} holds {actual} placeholders, its grid needs {expected}"
            ),
            Error::RowCount { expected, actual } => {
                write!(formatter, "{actual} merged rows for a grid that needs {expected}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// One image: its merged rows and the grid they came from.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    pub values: &'a [f32],
    pub grid: vision::Grid,
}

/// The merged patch count of one image.
fn merged_tokens(grid: &vision::Grid) -> usize {
    let merge = vision::constants::MERGE;
    grid.temporal * (grid.height / merge) * (grid.width / merge)
}

/// The runs of consecutive image placeholders, as `(start, length)`.
fn image_runs(ids: &[u32]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < ids.len() {
        if ids[index] == IMAGE_PLACEHOLDER {
            let start = index;
            while index < ids.len() && ids[index] == IMAGE_PLACEHOLDER {
                index += 1;
            }
            runs.push((start, index - start));
        } else {
            index += 1;
        }
    }
    runs
}

/// The positions of every token, and the offset later tokens need.
///
/// A text run continues from the current position one index at a time. An image
/// run takes the grid coordinates of its merged patches in raster order, shifted
/// by the current position, and then advances the current position by the grid's
/// larger side over the merge size — so the text after an image reads as following
/// it spatially rather than as following a token count.
///
/// The offset is what the reference adds to `past + index` when it generates the
/// next token: the prompt's largest position plus one, minus the prompt's length.
pub fn positions(ids: &[u32], images: &[vision::Grid]) -> Result<(Vec<[u32; 3]>, i64), Error> {
    let runs = image_runs(ids);
    if runs.len() != images.len() {
        return Err(Error::ImageCount {
            runs: runs.len(),
            given: images.len(),
        });
    }
    let merge = vision::constants::MERGE;
    let mut out = Vec::with_capacity(ids.len());
    let mut current = 0u32;
    let mut previous = 0usize;
    for (image_index, ((start, length), grid)) in runs.iter().zip(images).enumerate() {
        // The text before this run.
        for _ in previous..*start {
            out.push([current; 3]);
            current += 1;
        }
        let expected = merged_tokens(grid);
        if *length != expected {
            return Err(Error::RunLength {
                run: image_index,
                expected,
                actual: *length,
            });
        }
        let rows = grid.height / merge;
        let columns = grid.width / merge;
        for frame in 0..grid.temporal {
            let temporal = current + frame as u32;
            for row in 0..rows {
                for column in 0..columns {
                    out.push([
                        temporal,
                        current + row as u32,
                        current + column as u32,
                    ]);
                }
            }
        }
        current += (grid.height.max(grid.width) / merge) as u32;
        previous = start + length;
    }
    for _ in previous..ids.len() {
        out.push([current; 3]);
        current += 1;
    }
    let largest = out.iter().map(|position| position.iter().copied().max().unwrap()).max().unwrap_or(0);
    let delta = i64::from(largest) + 1 - ids.len() as i64;
    Ok((out, delta))
}

/// Expand one placeholder per image into the run its grid needs.
///
/// The chat template writes a single `<|image_pad|>` where the image goes; the
/// processor is what turns it into one token per merged patch. A caller that
/// renders the template itself has to do the same, so this does it and reports a
/// placeholder that has no grid behind it.
pub fn expand_placeholders(ids: &[u32], grids: &[vision::Grid]) -> Result<Vec<u32>, Error> {
    let mut out = Vec::with_capacity(ids.len());
    let mut remaining = grids.len();
    let mut next = 0;
    for id in ids {
        if *id == IMAGE_PLACEHOLDER {
            if next >= grids.len() {
                return Err(Error::ImageCount {
                    runs: next + 1,
                    given: grids.len(),
                });
            }
            out.extend(std::iter::repeat_n(IMAGE_PLACEHOLDER, merged_tokens(&grids[next])));
            next += 1;
            remaining -= 1;
        } else {
            out.push(*id);
        }
    }
    if remaining > 0 {
        return Err(Error::ImageCount {
            runs: grids.len() - remaining,
            given: grids.len(),
        });
    }
    Ok(out)
}

/// The prompt's embeddings, with each image run replaced by its rows.
pub fn embeddings(ids: &[u32], images: &[Image<'_>]) -> Result<(Vec<f32>, i64), Error> {
    let grids: Vec<vision::Grid> = images.iter().map(|image| image.grid).collect();
    let (_, delta) = positions(ids, &grids)?;
    let hidden = crate::weights::manifest::HIDDEN_SIZE;
    let embed = crate::weights::expect("model.language_model.embed_tokens.weight").matrix();
    let mut out = vec![0.0f32; ids.len() * hidden];
    for (token, id) in ids.iter().enumerate() {
        crate::ops::row_values(
            &embed,
            *id as usize,
            &mut out[token * hidden..(token + 1) * hidden],
        );
    }
    let runs = image_runs(ids);
    for ((start, length), image) in runs.iter().zip(images) {
        let expected = merged_tokens(&image.grid);
        if image.values.len() != expected * hidden {
            return Err(Error::RowCount {
                expected: expected * hidden,
                actual: image.values.len(),
            });
        }
        let _ = length;
        let target = &mut out[*start * hidden..(*start + expected) * hidden];
        target.copy_from_slice(image.values);
    }
    Ok((out, delta))
}

impl Model {
    /// Run a prompt that may contain images and return the logits of its last
    /// position.
    ///
    /// The cache learns the position offset the image's grid implies, so later
    /// single-token calls continue from the right place without the caller having
    /// to remember anything.
    pub fn forward_prompt<T: Trace>(
        &self,
        ids: &[u32],
        images: &[Image<'_>],
        cache: &mut Cache,
        trace: &mut T,
    ) -> Result<Vec<f32>, Error> {
        let (embeds, delta) = embeddings(ids, images)?;
        let grids: Vec<vision::Grid> = images.iter().map(|image| image.grid).collect();
        let (positions, _) = positions(ids, &grids)?;
        cache.position_delta = delta;
        Ok(self.forward_embeds(&embeds, ids.len(), &positions, cache, trace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_text_prompt_gets_the_token_indices() {
        let ids = [10u32, 11, 12];
        let (positions, delta) = positions(&ids, &[]).expect("positions");
        assert_eq!(positions, vec![[0; 3], [1; 3], [2; 3]]);
        assert_eq!(delta, 0);
    }

    #[test]
    fn an_image_shifts_the_text_behind_it_by_its_grid() {
        // Four placeholders for a 4x4 patch grid (two merged rows and columns),
        // then two text tokens.
        let grid = vision::Grid {
            temporal: 1,
            height: 4,
            width: 4,
        };
        let mut ids = vec![IMAGE_PLACEHOLDER; 4];
        ids.extend([7u32, 8]);
        let (positions, delta) = positions(&ids, &[grid]).expect("positions");
        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                // `max(4, 4) / 2 = 2`, so the text continues at 2.
                [2; 3],
                [3; 3],
            ]
        );
        // Largest position 3, six tokens, so the next generated token is at 4.
        assert_eq!(delta, 3 + 1 - 6);
    }

    #[test]
    fn a_mismatched_run_is_rejected() {
        let grid = vision::Grid {
            temporal: 1,
            height: 4,
            width: 4,
        };
        let ids = vec![IMAGE_PLACEHOLDER; 3];
        assert_eq!(
            positions(&ids, &[grid]),
            Err(Error::RunLength {
                run: 0,
                expected: 4,
                actual: 3
            })
        );
    }
}

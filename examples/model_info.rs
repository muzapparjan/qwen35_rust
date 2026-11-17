//! What the embedded container holds.
//!
//! The checksums are not decoration. An embedded static that nothing reads is
//! dropped by the linker, so a binary size measured without touching the bytes
//! would not be the size of the model.

use qwen35::weights::{self, manifest};

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut value: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    value
}

fn main() {
    let blob = weights::blob();
    println!("container bytes    : {}", blob.len());
    println!("recorded sha256    : {}", manifest::BLOB_SHA256);
    println!(
        "magic              : {}",
        String::from_utf8_lossy(&blob[..4])
    );
    println!("header bytes       : {}", manifest::HEADER_BYTES);
    println!(
        "tensors            : {} ({} in the model, {} mtp)",
        manifest::TENSOR_COUNT,
        manifest::MODEL_TENSOR_COUNT,
        manifest::TENSOR_COUNT - manifest::MODEL_TENSOR_COUNT
    );
    println!(
        "parameters         : {} ({} in the model)",
        manifest::PARAM_COUNT,
        manifest::MODEL_PARAM_COUNT
    );
    println!(
        "decoder layers     : {} linear, {} full attention",
        manifest::LINEAR_LAYERS.len(),
        manifest::FULL_LAYERS.len()
    );

    let embed = weights::expect("model.language_model.embed_tokens.weight");
    println!(
        "embedding          : {:?} {}",
        embed.spec.shape, embed.spec.kind
    );
    let raw = embed.raw();
    println!("embed head fnv1a64 : {:016x}", fnv1a64(&raw[..4096]));
    println!(
        "embed tail fnv1a64 : {:016x}",
        fnv1a64(&raw[raw.len() - 4096..])
    );

    let last = manifest::TENSORS.last().expect("the container is not empty");
    println!("last tensor        : {}", last.name);
    println!(
        "blob head fnv1a64  : {:016x}",
        fnv1a64(&blob[..4096])
    );
    println!(
        "blob tail fnv1a64  : {:016x}",
        fnv1a64(&blob[blob.len() - 4096..])
    );
    println!(
        "blob alignment     : {} bytes",
        (blob.as_ptr() as usize) % manifest::ALIGNMENT
    );
}

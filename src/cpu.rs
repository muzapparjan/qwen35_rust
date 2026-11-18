//! Which instruction sets this machine has, detected once.
//!
//! Detection is a one-time `is_x86_feature_detected!` per feature; the result is
//! cached so the kernels do not pay for it per call.

/// Vector instruction sets the kernels can use.
///
/// The names are the ones the kernels are written against, not the vendor: AVX2
/// and Advanced SIMD (NEON) are both wide floating-point extensions with a fused
/// multiply-add, and both are implemented by more than one vendor.
#[derive(Clone, Copy, Debug)]
pub struct Features {
    /// AVX2 and FMA together. Every AVX2 CPU in practice has FMA, but both are
    /// tested: the kernels that use FMA would be wrong without it.
    pub avx2_fma: bool,
    /// Advanced SIMD with a fused multiply-add on 64-bit ARM. It is part of the
    /// architecture from ARMv8-A on, so it needs no runtime test there.
    pub neon: bool,
}

static FEATURES: std::sync::OnceLock<Features> = std::sync::OnceLock::new();

/// The features available on this machine.
///
/// `--features force-scalar` reports none of them, which is how the portable
/// path is measured end to end.
pub fn features() -> &'static Features {
    FEATURES.get_or_init(|| {
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(feature = "force-scalar")
        ))]
        {
            Features {
                avx2_fma: std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma"),
                neon: false,
            }
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
        {
            // NEON is mandatory in ARMv8-A, which is what every `aarch64` target
            // is, so there is nothing to detect.
            Features {
                avx2_fma: false,
                neon: true,
            }
        }
        #[cfg(not(all(
            any(
                target_arch = "x86",
                target_arch = "x86_64",
                all(target_arch = "aarch64", not(feature = "force-scalar"))
            ),
            not(feature = "force-scalar")
        )))]
        {
            Features {
                avx2_fma: false,
                neon: false,
            }
        }
    })
}

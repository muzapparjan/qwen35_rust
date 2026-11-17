//! Which instruction sets this machine has, detected once.
//!
//! Detection is a one-time `is_x86_feature_detected!` per feature; the result is
//! cached so the kernels do not pay for it per call.

/// Instruction sets the kernels can use.
#[derive(Clone, Copy, Debug)]
pub struct Features {
    /// AVX2 and FMA together. Every AVX2 CPU in practice has FMA, but both are
    /// tested: the kernels that use FMA would be wrong without it.
    pub avx2_fma: bool,
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
            }
        }
        #[cfg(not(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(feature = "force-scalar")
        )))]
        {
            Features { avx2_fma: false }
        }
    })
}

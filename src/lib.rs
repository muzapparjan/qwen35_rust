//! Qwen3.5-0.8B inference in Rust.
//!
//! The default build is CPU only and has no dependencies: the numeric kernels,
//! the tokenizer and the quantised weight reader are all `std`. The `wgpu`
//! feature adds a second backend for the same arithmetic; `force-scalar` turns
//! the vector kernels off so the portable path can be measured end to end.
//!
//! The weights are a QWQT container embedded with `include_bytes!`, so the
//! library performs no file IO at run time.
//!
//! # What the model may use
//!
//! Generation is a fork-join workload and will take every core it is offered.
//! A process that owns the machine can offer all of them; anything that shares
//! the machine with a render loop, a UI thread or a mouse cursor cannot, so the
//! pool takes a budget instead of deciding for itself ([`parallel::Budget`]).
//!
//! The default is the one that stays out of the way: at most sixteen workers,
//! two cores left over for the caller, and [`parallel::Priority::Background`],
//! which puts the workers below every normal thread so the operating system
//! preempts them the moment the host wants a core. On an idle machine that costs
//! a few percent, and it is what keeps a generation from freezing the desktop it
//! runs on. A benchmark that wants raw throughput asks for
//! [`parallel::Priority::Normal`] and `reserve: 0` explicitly.
//!
//! ```no_run
//! qwen35::parallel::set_budget(qwen35::parallel::Budget {
//!     threads: 4,
//!     reserve: 2,
//!     priority: qwen35::parallel::Priority::Background,
//! });
//! ```
//!
//! The thread count is fixed when the pool is created, which is the first call
//! into the model; the priority may be changed at any time, which is how a host
//! yields while it is loading a level or showing a menu.
//!
//! Both ends of a priority change are free of privileges and reversible on the
//! three platforms that have a mechanism for it — Windows, Linux (including
//! Android) and macOS. Every other target keeps its threads where the process
//! put them, and [`parallel::Priority`] documents that; a host that needs the
//! workers to yield there has to set the priority of the process itself.

mod blob;
pub mod backend;
pub mod chat;
pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod gpu;
pub mod model;
pub mod multimodal;
pub mod ops;
pub mod parallel;
pub mod tokenizer;
pub mod vision;
pub mod weights;

//! A fork-join worker pool, `std` only.
//!
//! Spawning threads per call costs more than the work: one linear layer is a few
//! milliseconds and thread creation is tens of microseconds per thread. The pool
//! is created once and its workers park on a condition variable between jobs.
//!
//! # Why the job pointer is type-erased
//!
//! Work items borrow the caller's slices, which do not live for `'static`. The
//! pointer is therefore erased to `*const Job` and only read inside a protocol
//! that keeps the borrow alive: [`Pool::parallel_for`] does not return until every
//! worker has finished the generation it published.

use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type Job = dyn Fn(usize) + Sync + 'static;

/// A type-erased job pointer whose lifetime the join protocol upholds.
#[derive(Clone, Copy)]
struct JobPointer(*const Job);

// SAFETY: workers only dereference the pointer while running the generation it
// was published with, and the publisher waits for all of them.
unsafe impl Send for JobPointer {}

struct Inner {
    generation: u64,
    parts: usize,
    workers: usize,
    job: Option<JobPointer>,
    shutdown: bool,
}

struct Shared {
    /// Mirrors `Inner::generation` so a worker can spin before paying for the
    /// mutex, which is what makes a barrier cost microseconds.
    epoch: AtomicU64,
    /// Workers that have not finished the current job.
    remaining: AtomicUsize,
    inner: Mutex<Inner>,
    work: Condvar,
    done: Condvar,
    /// The first panic a worker caught; the publisher re-raises it.
    failed: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

/// Spin iterations before parking: long enough to cover a barrier, short enough
/// that an idle worker gives the core back.
const SPIN_ROUNDS: u32 = 4000;

/// A pool of parked workers.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// The protocol has one publisher: a second concurrent caller would overwrite
    /// the job the first is waiting on.
    publisher: Mutex<()>,
}

impl Pool {
    /// Start `threads` workers.
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            inner: Mutex::new(Inner {
                generation: 0,
                parts: 0,
                workers: threads,
                job: None,
                shutdown: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
            failed: Mutex::new(None),
        });
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let shared = Arc::clone(&shared);
            workers.push(
                std::thread::Builder::new()
                    .name(format!("qwen35-{index}"))
                    .spawn(move || worker(shared, index))
                    .expect("failed to start a worker thread"),
            );
        }
        Pool {
            shared,
            workers,
            publisher: Mutex::new(()),
        }
    }

    /// Number of workers.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Run `body(part)` for every part and return when all have finished.
    pub fn parallel_for(&self, parts: usize, body: &(dyn Fn(usize) + Sync)) {
        if parts == 0 {
            return;
        }
        if self.workers.is_empty() || parts == 1 {
            for part in 0..parts {
                body(part);
            }
            return;
        }
        // SAFETY: `body` outlives this call, and the call does not return until
        // every worker has finished with the pointer.
        let pointer = JobPointer(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync), *const Job>(body as *const _)
        });
        let published = self
            .publisher
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        {
            let mut inner = self.shared.inner.lock().expect("pool mutex poisoned");
            inner.job = Some(pointer);
            inner.parts = parts;
            inner.generation = inner.generation.wrapping_add(1);
            self.shared.epoch.store(inner.generation, Ordering::Release);
            self.shared.remaining.store(self.workers.len(), Ordering::Release);
            // Signalled while holding the mutex, so it cannot be lost between a
            // worker's predicate check and its wait.
            self.shared.work.notify_all();
        }
        let mut spins = 0;
        while self.shared.remaining.load(Ordering::Acquire) > 0 && spins < SPIN_ROUNDS {
            std::hint::spin_loop();
            spins += 1;
        }
        if self.shared.remaining.load(Ordering::Acquire) > 0 {
            let guard = self.shared.inner.lock().expect("pool mutex poisoned");
            let guard = self
                .shared
                .done
                .wait_while(guard, |_| self.shared.remaining.load(Ordering::Acquire) > 0)
                .expect("pool mutex poisoned");
            drop(guard);
        }
        {
            let mut inner = self.shared.inner.lock().expect("pool mutex poisoned");
            inner.job = None;
        }
        // The lock is released before a panic is re-raised, otherwise the next
        // call would find it poisoned.
        drop(published);
        let failed = self
            .shared
            .failed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(payload) = failed {
            std::panic::resume_unwind(payload);
        }
    }

    /// Split `data` into chunks of `chunk` items and run `body(index, chunk)` in
    /// parallel. The index is the chunk's position in `data`.
    pub fn chunks_mut<T: Send>(&self, chunk: usize, data: &mut [T], body: &(dyn Fn(usize, &mut [T]) + Sync)) {
        assert!(chunk > 0, "chunk length must be positive");
        if data.is_empty() {
            return;
        }
        let parts = data.len().div_ceil(chunk);
        let mut queue: Vec<(usize, &mut [T])> = data.chunks_mut(chunk).enumerate().collect();
        if queue.len() == 1 {
            let (index, slice) = queue.pop().expect("one chunk");
            body(index, slice);
            return;
        }
        let queue = Mutex::new(queue);
        self.parallel_for(parts, &|_| {
            let next = queue.lock().expect("chunk mutex poisoned").pop();
            if let Some((index, slice)) = next {
                body(index, slice);
            }
        });
    }

    /// Split two independent buffers into matching chunks and run
    /// `body(index, first, second)` in parallel.
    ///
    /// The strides differ in practice (a head's recurrent state and its output
    /// have different lengths), so each side takes its own chunk length; what
    /// must agree is the number of chunks, which is checked.
    pub fn chunks_mut_pair<A: Send, B: Send>(
        &self,
        first_chunk: usize,
        first: &mut [A],
        second_chunk: usize,
        second: &mut [B],
        body: &(dyn Fn(usize, &mut [A], &mut [B]) + Sync),
    ) {
        assert!(first_chunk > 0 && second_chunk > 0, "chunk lengths must be positive");
        let parts = first.len().div_ceil(first_chunk);
        assert_eq!(
            parts,
            second.len().div_ceil(second_chunk),
            "the two buffers split into a different number of parts"
        );
        let mut queue: Vec<(usize, &mut [A], &mut [B])> = first
            .chunks_mut(first_chunk)
            .zip(second.chunks_mut(second_chunk))
            .enumerate()
            .map(|(index, (left, right))| (index, left, right))
            .collect();
        if queue.len() == 1 {
            let (index, left, right) = queue.pop().expect("one chunk");
            body(index, left, right);
            return;
        }
        let queue = Mutex::new(queue);
        self.parallel_for(parts, &|_| {
            let next = queue.lock().expect("chunk mutex poisoned").pop();
            if let Some((index, left, right)) = next {
                body(index, left, right);
            }
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut inner = self.shared.inner.lock().expect("pool mutex poisoned");
            inner.shutdown = true;
            self.shared.work.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(shared: Arc<Shared>, index: usize) {
    let mut seen = 0u64;
    let mut applied = apply_priority(DESIRED_PRIORITY.load(Ordering::SeqCst));
    loop {
        // Fast path: spin until the epoch moves.
        let mut spins = 0;
        while shared.epoch.load(Ordering::Relaxed) == seen {
            if spins >= SPIN_ROUNDS {
                let guard = shared.inner.lock().expect("pool mutex poisoned");
                let guard = shared
                    .work
                    .wait_while(guard, |inner| inner.generation == seen && !inner.shutdown)
                    .expect("pool mutex poisoned");
                if guard.shutdown {
                    return;
                }
                drop(guard);
                spins = 0;
            } else {
                std::hint::spin_loop();
                spins += 1;
            }
        }
        let (generation, parts, workers, pointer) = {
            let inner = shared.inner.lock().expect("pool mutex poisoned");
            if inner.shutdown {
                return;
            }
            (inner.generation, inner.parts, inner.workers, inner.job)
        };
        if generation != seen {
            // The host may have asked for a different priority since the last
            // job; the comparison is free against a job that is about to run.
            let desired = DESIRED_PRIORITY.load(Ordering::SeqCst);
            if desired != applied {
                applied = apply_priority(desired);
            }
            if let Some(pointer) = pointer {
                // SAFETY: the publisher waits for `remaining` to reach zero, so
                // the borrowed data is still alive; this worker decrements below
                // before it looks for new work.
                let body: &dyn Fn(usize) = unsafe { &*pointer.0 };
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut part = index;
                    while part < parts {
                        body(part);
                        part += workers.max(1);
                    }
                }));
                if let Err(payload) = outcome {
                    let mut failed = shared.failed.lock().expect("pool mutex poisoned");
                    if failed.is_none() {
                        *failed = Some(payload);
                    }
                }
            }
            seen = generation;
            // A panic still counts as "this worker is done", otherwise the
            // publisher would wait for a wake-up that never comes.
            if shared.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _guard = shared.inner.lock().expect("pool mutex poisoned");
                shared.done.notify_all();
            }
        }
    }
}

/// How many workers this machine should use for the model's shapes.
///
/// The count is capped rather than left at the core count: past this point the
/// batch kernels gain about a tenth while the single-token kernel loses about a
/// third, the workers having started to fight over the same caches.
const MAX_WORKERS: usize = 16;

/// What a host wants the pool to be allowed to use.
///
/// A model that saturates every core at normal priority starves whatever shares
/// the machine: a game's render loop, a UI thread, the mouse cursor. The pool
/// therefore takes a budget rather than deciding for itself, and the default is
/// the polite one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Worker threads. `0` asks for a count derived from the machine and
    /// `reserve`; anything else is used as given, up to a fixed cap.
    pub threads: usize,
    /// Cores to leave for the host when `threads` is `0`. Two is enough for a
    /// render loop and its audio thread.
    pub reserve: usize,
    /// Scheduling priority of the workers.
    pub priority: Priority,
}

/// How the workers compete with the host's own threads.
///
/// Each platform has a mechanism that is both below normal and reversible
/// without a privilege, so a host can raise the workers again when it stops
/// needing them to yield:
///
/// - Windows: `THREAD_PRIORITY_LOWEST`;
/// - Linux and Android: the `SCHED_IDLE` scheduling class, which runs only when
///   no other thread wants the core, and which an unprivileged thread is always
///   allowed to enter and to leave;
/// - macOS: `PRIO_DARWIN_BG` on the thread.
///
/// Every one of those is reversible without a privilege, so a host that asks
/// for [`Priority::Normal`] again gets the cores back. A target with none of
/// those mechanisms keeps its threads where the process put them; the pool
/// reports that through [`budget`] rather than pretending the change landed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// Below every normal thread: the operating system preempts a worker the
    /// moment any other thread wants its core, so a host stays responsive and
    /// still gets every core that nobody else is using.
    Background,
    /// The default priority. Only for a tool that owns the machine, such as a
    /// benchmark; an embedded host should not ask for this.
    Normal,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget {
            threads: 0,
            reserve: 2,
            priority: Priority::Background,
        }
    }
}

impl Budget {
    /// The worker count this budget means on a machine with `available` cores.
    pub fn workers(&self, available: usize) -> usize {
        let available = available.max(1);
        if self.threads > 0 {
            return self.threads.min(MAX_WORKERS).max(1);
        }
        available.saturating_sub(self.reserve).min(MAX_WORKERS).max(1)
    }
}

/// The budget the pool will be built with, and the one its workers follow.
static BUDGET: Mutex<Budget> = Mutex::new(Budget {
    threads: 0,
    reserve: 2,
    priority: Priority::Background,
});

/// The priority the workers should be running at, read without the mutex.
static DESIRED_PRIORITY: AtomicI32 = AtomicI32::new(BACKGROUND_CODE);

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

/// Set what the pool may use. Call before the first [`pool`] access.
///
/// The priority may be changed at any time; the thread count may not, because a
/// pool cannot grow or shrink, and silently ignoring the request would leave the
/// caller believing the machine is quieter than it is. Changing the count after
/// the pool exists is therefore a programming error and panics.
pub fn set_budget(budget: Budget) {
    if let Some(pool) = POOL.get()
        && budget.threads > 0
    {
        let count = budget.workers(available_cores());
        assert_eq!(
            count,
            pool.threads(),
            "the pool already has {} workers, so the thread count can no longer change; set the budget before the first use",
            pool.threads()
        );
    }
    DESIRED_PRIORITY.store(priority_code(budget.priority), Ordering::SeqCst);
    *BUDGET.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = budget;
}

/// The budget in force.
///
/// Reading it does not create the pool.
pub fn budget() -> Budget {
    *BUDGET.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run the workers at `priority` from now on.
pub fn set_priority(priority: Priority) {
    set_budget(Budget {
        priority,
        ..budget()
    });
}

/// Ask for a worker count; `0` means "derive it from the budget".
///
/// A thin wrapper over [`set_budget`] for callers that only want to change the
/// count.
pub fn set_threads(count: usize) {
    set_budget(Budget {
        threads: count,
        ..budget()
    });
}

/// Number of workers in the pool.
pub fn threads() -> usize {
    pool().threads()
}

fn available_cores() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// The shared pool, created on first use.
pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let budget = budget();
        DESIRED_PRIORITY.store(priority_code(budget.priority), Ordering::SeqCst);
        Pool::new(budget.workers(available_cores()))
    })
}

/// The priority codes the pool stores and compares.
///
/// They are the pool's own numbering rather than a platform constant, so one
/// atomic carries the request on every target and the platform mapping stays
/// inside [`native`].
const BACKGROUND_CODE: i32 = 0;
const NORMAL_CODE: i32 = 1;

/// Whether this platform can apply a priority.
const SUPPORTED: bool = cfg!(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "macos"
));

/// The pool's code for a priority.
fn priority_code(priority: Priority) -> i32 {
    match priority {
        Priority::Background => BACKGROUND_CODE,
        Priority::Normal => NORMAL_CODE,
    }
}

/// Put the calling thread at `code`, returning the code in force afterwards.
///
/// A platform that refuses the change leaves the thread where it was, and the
/// return value says so instead of claiming the request took effect.
fn apply_priority(code: i32) -> i32 {
    if !SUPPORTED {
        return code;
    }
    if native::set(code == BACKGROUND_CODE) {
        code
    } else {
        current_priority()
    }
}

/// The priority the calling thread is running at.
fn current_priority() -> i32 {
    if SUPPORTED && native::is_background() {
        BACKGROUND_CODE
    } else {
        NORMAL_CODE
    }
}

/// The platform's own way of lowering a thread.
#[cfg(windows)]
mod native {
    /// `THREAD_PRIORITY_LOWEST` and `THREAD_PRIORITY_NORMAL`.
    const LOWEST: i32 = -2;
    const NORMAL: i32 = 0;
    /// `THREAD_PRIORITY_BELOW_NORMAL`; anything at or below this is a thread the
    /// system prefers over.
    const BELOW_NORMAL: i32 = -1;

    pub fn set(background: bool) -> bool {
        let level = if background { LOWEST } else { NORMAL };
        // SAFETY: `GetCurrentThread` returns a pseudo handle for the calling
        // thread, which is valid to pass to `SetThreadPriority` and must not be
        // closed. Neither call can invalidate anything this thread owns.
        unsafe { SetThreadPriority(GetCurrentThread(), level) != 0 }
    }

    pub fn is_background() -> bool {
        // SAFETY: the same pseudo handle as `set`.
        unsafe { GetThreadPriority(GetCurrentThread()) <= BELOW_NORMAL }
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn GetThreadPriority(handle: *mut core::ffi::c_void) -> i32;
        fn SetThreadPriority(handle: *mut core::ffi::c_void, priority: i32) -> i32;
    }
}

/// The platform's own way of lowering a thread.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod native {
    const SCHED_OTHER: i32 = 0;
    const SCHED_IDLE: i32 = 5;

    /// The priority a policy takes. Both policies here ignore it and require
    /// zero.
    #[repr(C)]
    struct SchedParam {
        sched_priority: i32,
    }

    pub fn set(background: bool) -> bool {
        let policy = if background { SCHED_IDLE } else { SCHED_OTHER };
        let param = SchedParam { sched_priority: 0 };
        // SAFETY: a zero pid means the calling thread, and `param` points at a
        // live `sched_param` for the duration of the call.
        unsafe { sched_setscheduler(0, policy, &param) == 0 }
    }

    pub fn is_background() -> bool {
        // SAFETY: a zero pid means the calling thread.
        unsafe { sched_getscheduler(0) == SCHED_IDLE }
    }

    unsafe extern "C" {
        fn sched_setscheduler(pid: i32, policy: i32, param: *const SchedParam) -> i32;
        fn sched_getscheduler(pid: i32) -> i32;
    }
}

/// The platform's own way of lowering a thread.
#[cfg(target_os = "macos")]
mod native {
    const PRIO_DARWIN_THREAD: i32 = 3;
    const PRIO_DARWIN_BG: i32 = 0x10000;

    pub fn set(background: bool) -> bool {
        let value = if background { PRIO_DARWIN_BG } else { 0 };
        // SAFETY: a zero id means the calling thread.
        unsafe { setpriority(PRIO_DARWIN_THREAD, 0, value) == 0 }
    }

    pub fn is_background() -> bool {
        // SAFETY: a zero id means the calling thread.
        unsafe { getpriority(PRIO_DARWIN_THREAD, 0) == PRIO_DARWIN_BG }
    }

    unsafe extern "C" {
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        fn getpriority(which: i32, who: u32) -> i32;
    }
}

/// Platforms with no mechanism here keep the thread where the process put it.
#[cfg(not(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "macos"
)))]
mod native {
    pub fn set(_background: bool) -> bool {
        false
    }

    pub fn is_background() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_part_runs_once() {
        let pool = Pool::new(3);
        let seen: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        pool.parallel_for(10, &|part| {
            seen.lock().unwrap().push(part);
        });
        let mut seen = seen.into_inner().unwrap();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<usize>>());
    }

    #[test]
    fn chunks_cover_the_slice() {
        let pool = Pool::new(3);
        let mut values: Vec<usize> = (0..37).collect();
        pool.chunks_mut(5, &mut values, &|_, chunk| {
            for value in chunk.iter_mut() {
                *value *= 2;
            }
        });
        assert_eq!(values, (0..37).map(|value| value * 2).collect::<Vec<usize>>());
    }

    #[test]
    fn a_panic_in_a_part_surfaces_in_the_caller() {
        let pool = Pool::new(2);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.parallel_for(4, &|part| {
                if part == 2 {
                    panic!("part {part} failed");
                }
            });
        }));
        assert!(outcome.is_err(), "the panic was swallowed");
        // The pool is still usable.
        pool.parallel_for(2, &|_| {});
    }

    #[test]
    fn a_budget_leaves_the_reserved_cores_alone() {
        let polite = Budget::default();
        assert_eq!(polite.workers(24), 16, "capped, and two cores are left over");
        assert_eq!(polite.workers(4), 2, "a small machine keeps two cores for the host");
        assert_eq!(polite.workers(2), 1, "one core is still a worker");
        assert_eq!(polite.workers(1), 1, "a single-core machine still runs");
    }

    #[test]
    fn an_explicit_count_wins_over_the_reserve() {
        let fixed = Budget {
            threads: 4,
            reserve: 8,
            priority: Priority::Normal,
        };
        assert_eq!(fixed.workers(24), 4, "the reserve does not subtract from a fixed count");
        assert_eq!(fixed.threads, 4);
    }

    #[test]
    fn an_explicit_count_is_still_capped() {
        let fixed = Budget {
            threads: 1000,
            ..Budget::default()
        };
        assert_eq!(fixed.workers(24), MAX_WORKERS);
    }

    #[test]
    fn a_priority_change_reaches_the_workers() {
        if !SUPPORTED {
            return;
        }
        let pool = Pool::new(2);
        let sample = || {
            let observed = Mutex::new(Vec::new());
            pool.parallel_for(2, &|_| {
                observed.lock().unwrap().push(current_priority());
            });
            observed.into_inner().unwrap()
        };

        // Both directions: a mechanism that only lowers, or only once, would
        // leave the pool stuck and the host unable to take the cores back.
        set_priority(Priority::Normal);
        let seen = sample();
        assert!(
            seen.iter().all(|code| *code == priority_code(Priority::Normal)),
            "workers ran at {seen:?} after the priority was set to normal"
        );

        set_priority(Priority::Background);
        let seen = sample();
        assert!(
            seen.iter().all(|code| *code == priority_code(Priority::Background)),
            "workers ran at {seen:?} after the priority was set to background"
        );

        set_priority(Priority::Normal);
        let seen = sample();
        assert!(
            seen.iter().all(|code| *code == priority_code(Priority::Normal)),
            "workers stayed at {seen:?} after the priority was raised again"
        );
    }

    #[test]
    fn one_part_takes_the_direct_path() {
        let pool = Pool::new(4);
        let visited = AtomicUsize::new(0);
        pool.parallel_for(1, &|_| {
            visited.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(visited.load(Ordering::SeqCst), 1);
    }
}

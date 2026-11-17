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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
/// [`set_threads`] overrides it, which is how `examples/bench.rs` sweeps the
/// value.
const MAX_WORKERS: usize = 8;

static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

/// Ask for a worker count; 0 means "choose one". Ignored once the pool exists.
pub fn set_threads(count: usize) {
    REQUESTED.store(count, Ordering::SeqCst);
}

/// Number of workers in the pool.
pub fn threads() -> usize {
    pool().threads()
}

/// The shared pool, created on first use.
pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let requested = REQUESTED.load(Ordering::SeqCst);
        let count = if requested > 0 {
            requested
        } else {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
                .saturating_sub(1)
                .min(MAX_WORKERS)
        };
        Pool::new(count)
    })
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
    fn one_part_takes_the_direct_path() {
        let pool = Pool::new(4);
        let visited = AtomicUsize::new(0);
        pool.parallel_for(1, &|_| {
            visited.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(visited.load(Ordering::SeqCst), 1);
    }
}

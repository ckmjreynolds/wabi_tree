//! Measures what the entry API costs over the direct calls it stands in for (#32).
//!
//! Each benchmark keeps the tree at a steady size: keys are timed in chunks, and the tree is
//! restored, untimed, after each chunk. Present keys are even codes and absent keys odd codes, so
//! a miss is always a miss. Comparing `entry` against `direct` on the same map shows the cost of
//! the entry API's extra walk from the root.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::{BTreeMap, btree_map};
use std::hint::black_box;
use std::time::{Duration, Instant};
use wabi_tree::{OSBTreeMap, osbtree_map};

/// The most keys timed between two restores of the tree.
const MAX_CHUNK: usize = 10_000;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 2
    }
}

trait BenchKey: Ord + Clone {
    fn from_code(code: u64) -> Self;
}

impl BenchKey for i64 {
    fn from_code(code: u64) -> Self {
        code as i64
    }
}

impl BenchKey for String {
    fn from_code(code: u64) -> Self {
        format!("{code:020}")
    }
}

trait BenchMap<K: BenchKey>: Sized {
    const NAME: &'static str;
    fn build(keys: &[K]) -> Self;
    fn insert(&mut self, key: K);
    fn remove(&mut self, key: &K);
    fn entry_insert(&mut self, key: K);
    fn entry_remove(&mut self, key: K);
    fn entry_count(&mut self, key: K);
    fn direct_count(&mut self, key: K);
}

macro_rules! impl_bench_map {
    ($map:ident, $entry:path, $name:literal) => {
        impl<K: BenchKey> BenchMap<K> for $map<K, u64> {
            const NAME: &'static str = $name;

            fn build(keys: &[K]) -> Self {
                keys.iter().map(|k| (k.clone(), 0)).collect()
            }

            fn insert(&mut self, key: K) {
                self.insert(key, 0);
            }

            fn remove(&mut self, key: &K) {
                $map::remove(self, key);
            }

            fn entry_insert(&mut self, key: K) {
                *self.entry(key).or_insert(0) += 1;
            }

            fn entry_remove(&mut self, key: K) {
                use $entry as Entry;
                if let Entry::Occupied(o) = self.entry(key) {
                    black_box(o.remove());
                }
            }

            fn entry_count(&mut self, key: K) {
                *self.entry(key).or_default() += 1;
            }

            fn direct_count(&mut self, key: K) {
                match self.get_mut(&key) {
                    Some(v) => *v += 1,
                    None => {
                        self.insert(key, 1);
                    }
                }
            }
        }
    };
}

impl_bench_map!(OSBTreeMap, osbtree_map::Entry, "OSBTreeMap");
impl_bench_map!(BTreeMap, btree_map::Entry, "BTreeMap");

/// A tree of `n` present keys, plus the keys in shuffled order.
struct Fixture<K, M> {
    map: M,
    present: Vec<K>,
    cursor: usize,
    rng: Lcg,
}

impl<K: BenchKey, M: BenchMap<K>> Fixture<K, M> {
    fn new(n: usize) -> Self {
        let mut rng = Lcg(12345);
        let mut codes: Vec<u64> = (0..n).map(|_| rng.next() & !1).collect();
        codes.sort_unstable();
        codes.dedup();
        let mut sorted: Vec<K> = codes.iter().map(|&c| K::from_code(c)).collect();
        sorted.sort();
        let map = M::build(&sorted);
        // Fisher-Yates, so hits visit the tree in random order.
        for i in (1..sorted.len()).rev() {
            sorted.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        Self {
            map,
            present: sorted,
            cursor: 0,
            rng,
        }
    }

    fn chunk_len(&self, remaining: u64) -> usize {
        (remaining as usize).min(MAX_CHUNK).min(self.present.len() / 2).max(1)
    }

    fn absent_keys(&mut self, len: usize) -> Vec<K> {
        (0..len).map(|_| K::from_code(self.rng.next() | 1)).collect()
    }

    fn present_keys(&mut self, len: usize) -> Vec<K> {
        (0..len)
            .map(|_| {
                self.cursor = (self.cursor + 1) % self.present.len();
                self.present[self.cursor].clone()
            })
            .collect()
    }

    /// Times `op` on `iters` keys from `keys`, running `restore` untimed after each chunk.
    fn time(
        &mut self,
        iters: u64,
        mut keys: impl FnMut(&mut Self, usize) -> Vec<K>,
        mut op: impl FnMut(&mut M, K),
        mut restore: impl FnMut(&mut M, &[K]),
    ) -> Duration {
        let mut total = Duration::ZERO;
        let mut done = 0;
        while done < iters {
            let len = self.chunk_len(iters - done);
            let batch = keys(self, len);
            let restore_keys = batch.clone();
            let start = Instant::now();
            for key in batch {
                op(&mut self.map, key);
            }
            total += start.elapsed();
            restore(&mut self.map, &restore_keys);
            done += len as u64;
        }
        total
    }
}

fn bench_size<K: BenchKey, M: BenchMap<K>>(c: &mut Criterion, key_name: &str, n: usize) {
    let mut fx = Fixture::<K, M>::new(n);
    let id = |form: &str| BenchmarkId::new(format!("{}/{form}", M::NAME), n);
    let undo_insert = |m: &mut M, ks: &[K]| ks.iter().for_each(|k| m.remove(k));
    let undo_remove = |m: &mut M, ks: &[K]| ks.iter().for_each(|k| m.insert(k.clone()));

    let mut group = c.benchmark_group(format!("map_entry_insert_miss/{key_name}"));
    group.bench_function(id("entry"), |b| {
        b.iter_custom(|iters| fx.time(iters, Fixture::absent_keys, M::entry_insert, undo_insert))
    });
    group.bench_function(id("direct"), |b| {
        b.iter_custom(|iters| fx.time(iters, Fixture::absent_keys, M::insert, undo_insert))
    });
    group.finish();

    let mut group = c.benchmark_group(format!("map_entry_remove_hit/{key_name}"));
    group.bench_function(id("entry"), |b| {
        b.iter_custom(|iters| fx.time(iters, Fixture::present_keys, M::entry_remove, undo_remove))
    });
    group.bench_function(id("direct"), |b| {
        b.iter_custom(|iters| fx.time(iters, Fixture::present_keys, |m, k| m.remove(&k), undo_remove))
    });
    group.finish();

    // Half hits, half misses: the `*map.entry(k).or_default() += 1` counting idiom.
    let mixed = |fx: &mut Fixture<K, M>, len: usize| {
        let mut keys = fx.present_keys(len / 2);
        keys.extend(fx.absent_keys(len - len / 2));
        keys
    };
    // Removing every key in the batch also drops the present ones, so put those back.
    let undo_mixed = |m: &mut M, ks: &[K]| {
        ks.iter().for_each(|k| m.remove(k));
        ks[..ks.len() / 2].iter().for_each(|k| m.insert(k.clone()));
    };
    let mut group = c.benchmark_group(format!("map_entry_count_mixed/{key_name}"));
    group.bench_function(id("entry"), |b| b.iter_custom(|iters| fx.time(iters, mixed, M::entry_count, undo_mixed)));
    group.bench_function(id("direct"), |b| b.iter_custom(|iters| fx.time(iters, mixed, M::direct_count, undo_mixed)));
    group.finish();
}

fn bench_entry(c: &mut Criterion) {
    for n in [10_000, 1_000_000, 10_000_000] {
        bench_size::<i64, OSBTreeMap<i64, u64>>(c, "i64", n);
        bench_size::<i64, BTreeMap<i64, u64>>(c, "i64", n);
    }
    for n in [10_000, 1_000_000] {
        bench_size::<String, OSBTreeMap<String, u64>>(c, "String", n);
        bench_size::<String, BTreeMap<String, u64>>(c, "String", n);
    }
}

criterion_group! {
    name = entry_benches;
    config = Criterion::default().sample_size(20).measurement_time(Duration::from_secs(3));
    targets = bench_entry
}
criterion_main!(entry_benches);

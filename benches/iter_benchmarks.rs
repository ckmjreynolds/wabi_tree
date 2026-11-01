//! Iterator positioning benchmarks: `skip`, `rev().skip`, `step_by`, and a plain full walk.
//!
//! `skip(n)` and `step_by(s)` are implemented in std on top of `Iterator::nth`, so these measure
//! how cheaply an iterator can jump forward or backward by rank (issue #18).
//!
//! Run with `cargo bench --bench iter_benchmarks`.
//!
//! Medians for `OSBTreeMap` with N = 1,000,000 (Apple Silicon, 2026-11-01), before and after #18
//! made `nth`/`nth_back` jump by rank:
//!
//! | benchmark                 |  before |   after |
//! |---------------------------|--------:|--------:|
//! | `iter_skip/10`            | 26.7 ns | 6.39 ns |
//! | `iter_skip/1000`          | 2.34 µs | 40.8 ns |
//! | `iter_skip/900000`        | 2.26 ms | 79.6 ns |
//! | `iter_skip_back/900000`   | 2.05 ms | 43.6 ns |
//! | `range_skip/700000`       | 2.00 ms |  146 ns |
//! | `iter_step_by/2`          | 2.26 ms | 1.88 ms |
//! | `iter_step_by/65`         | 2.18 ms |  158 µs |
//! | `iter_step_by/129`        | 1.98 ms |  142 µs |
//! | `iter_step_by/1000`       | 2.17 ms | 41.0 µs |
//! | `iter_full`               | 1.68 ms | 1.76 ms |
//!
//! `iter_full` is within code-layout noise: `BTreeMap`'s own `iter_full`, which #18 does not
//! touch, moved by up to 35% between the two builds.

// `skip(n).next()` is deliberate: it is the call pattern from issue #18 being measured.
#![allow(clippy::iter_skip_next)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::BTreeMap;
use std::hint::black_box;
use wabi_tree::OSBTreeMap;

const N: usize = 1_000_000;
const SKIPS: [usize; 5] = [10, 100, 1_000, 100_000, 900_000];
const STEPS: [usize; 4] = [2, 65, 129, 1_000];
const RANGE: core::ops::Range<i64> = 100_000..900_000;
const RANGE_SKIPS: [usize; 4] = [10, 1_000, 100_000, 700_000];

fn maps() -> (OSBTreeMap<i64, i64>, BTreeMap<i64, i64>) {
    let os_map = (0..N as i64).map(|k| (k, k)).collect();
    let bt_map = (0..N as i64).map(|k| (k, k)).collect();
    (os_map, bt_map)
}

fn bench_iter_skip(c: &mut Criterion) {
    let (os_map, bt_map) = maps();
    let mut group = c.benchmark_group("iter_skip");
    for n in SKIPS {
        group.bench_with_input(BenchmarkId::new("OSBTreeMap", n), &n, |b, &n| {
            b.iter(|| os_map.iter().skip(black_box(n)).next());
        });
        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter(|| bt_map.iter().skip(black_box(n)).next());
        });
    }
    group.finish();
}

fn bench_iter_skip_back(c: &mut Criterion) {
    let (os_map, bt_map) = maps();
    let mut group = c.benchmark_group("iter_skip_back");
    for n in SKIPS {
        group.bench_with_input(BenchmarkId::new("OSBTreeMap", n), &n, |b, &n| {
            b.iter(|| os_map.iter().rev().skip(black_box(n)).next());
        });
        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter(|| bt_map.iter().rev().skip(black_box(n)).next());
        });
    }
    group.finish();
}

fn bench_range_skip(c: &mut Criterion) {
    let (os_map, bt_map) = maps();
    let mut group = c.benchmark_group("range_skip");
    for n in RANGE_SKIPS {
        group.bench_with_input(BenchmarkId::new("OSBTreeMap", n), &n, |b, &n| {
            b.iter(|| os_map.range(RANGE).skip(black_box(n)).next());
        });
        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &n, |b, &n| {
            b.iter(|| bt_map.range(RANGE).skip(black_box(n)).next());
        });
    }
    group.finish();
}

fn bench_iter_step_by(c: &mut Criterion) {
    let (os_map, bt_map) = maps();
    let mut group = c.benchmark_group("iter_step_by");
    for s in STEPS {
        group.bench_with_input(BenchmarkId::new("OSBTreeMap", s), &s, |b, &s| {
            b.iter(|| os_map.iter().step_by(black_box(s)).count());
        });
        group.bench_with_input(BenchmarkId::new("BTreeMap", s), &s, |b, &s| {
            b.iter(|| bt_map.iter().step_by(black_box(s)).count());
        });
    }
    group.finish();
}

fn bench_iter_full(c: &mut Criterion) {
    let (os_map, bt_map) = maps();
    let mut group = c.benchmark_group("iter_full");
    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| os_map.iter().fold(0i64, |acc, (_, &v)| acc.wrapping_add(v)));
    });
    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| bt_map.iter().fold(0i64, |acc, (_, &v)| acc.wrapping_add(v)));
    });
    group.finish();
}

criterion_group!(
    iter_benches,
    bench_iter_skip,
    bench_iter_skip_back,
    bench_range_skip,
    bench_iter_step_by,
    bench_iter_full,
);

criterion_main!(iter_benches);

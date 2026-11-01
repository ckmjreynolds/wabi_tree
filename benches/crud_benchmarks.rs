use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::{BTreeMap, BTreeSet};
use wabi_tree::{OSBTreeMap, OSBTreeSet};

const N: usize = 10_000;

// ─── Helper functions to generate key sequences ─────────────────────────────

fn ordered_keys(n: usize) -> Vec<i64> {
    (0..n as i64).collect()
}

fn reverse_ordered_keys(n: usize) -> Vec<i64> {
    (0..n as i64).rev().collect()
}

fn random_keys(n: usize) -> Vec<i64> {
    // Use a simple LCG for deterministic pseudo-random sequence
    let mut keys = Vec::with_capacity(n);
    let mut x: u64 = 12345;
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        keys.push((x >> 33) as i64);
    }
    keys
}

// ─── Map Benchmarks ─────────────────────────────────────────────────────────

fn bench_map_insert_ordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_insert_ordered");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| {
            let mut map = OSBTreeMap::new();
            for i in 0..N as i64 {
                map.insert(i, i);
            }
            map
        });
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..N as i64 {
                map.insert(i, i);
            }
            map
        });
    });

    group.finish();
}

fn bench_map_insert_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_insert_reverse");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| {
            let mut map = OSBTreeMap::new();
            for i in (0..N as i64).rev() {
                map.insert(i, i);
            }
            map
        });
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in (0..N as i64).rev() {
                map.insert(i, i);
            }
            map
        });
    });

    group.finish();
}

fn bench_map_insert_random(c: &mut Criterion) {
    let keys = random_keys(N);
    let mut group = c.benchmark_group("map_insert_random");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| {
            let mut map = OSBTreeMap::new();
            for &k in &keys {
                map.insert(k, k);
            }
            map
        });
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for &k in &keys {
                map.insert(k, k);
            }
            map
        });
    });

    group.finish();
}

fn bench_map_get_ordered(c: &mut Criterion) {
    let keys = ordered_keys(N);
    let os_map: OSBTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();
    let bt_map: BTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();

    let mut group = c.benchmark_group("map_get_ordered");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &k in &keys {
                if let Some(&v) = os_map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            sum
        });
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &k in &keys {
                if let Some(&v) = bt_map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            sum
        });
    });

    group.finish();
}

fn bench_map_get_reverse(c: &mut Criterion) {
    let keys = ordered_keys(N);
    let os_map: OSBTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();
    let bt_map: BTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();
    let reverse_keys = reverse_ordered_keys(N);

    let mut group = c.benchmark_group("map_get_reverse");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &k in &reverse_keys {
                if let Some(&v) = os_map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            sum
        });
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &k in &reverse_keys {
                if let Some(&v) = bt_map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            sum
        });
    });

    group.finish();
}

fn bench_map_get_random(c: &mut Criterion) {
    let keys = random_keys(N);
    let os_map: OSBTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();
    let bt_map: BTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();

    let mut group = c.benchmark_group("map_get_random");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &k in &keys {
                if let Some(&v) = os_map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            sum
        });
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &k in &keys {
                if let Some(&v) = bt_map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            sum
        });
    });

    group.finish();
}

fn bench_map_remove_ordered(c: &mut Criterion) {
    let keys = ordered_keys(N);

    let mut group = c.benchmark_group("map_remove_ordered");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter_batched(
            || keys.iter().map(|&k| (k, k)).collect::<OSBTreeMap<i64, i64>>(),
            |mut map| {
                for &k in &keys {
                    map.remove(&k);
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter_batched(
            || keys.iter().map(|&k| (k, k)).collect::<BTreeMap<i64, i64>>(),
            |mut map| {
                for &k in &keys {
                    map.remove(&k);
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_map_remove_reverse(c: &mut Criterion) {
    let keys = ordered_keys(N);
    let reverse_keys = reverse_ordered_keys(N);

    let mut group = c.benchmark_group("map_remove_reverse");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter_batched(
            || keys.iter().map(|&k| (k, k)).collect::<OSBTreeMap<i64, i64>>(),
            |mut map| {
                for &k in &reverse_keys {
                    map.remove(&k);
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter_batched(
            || keys.iter().map(|&k| (k, k)).collect::<BTreeMap<i64, i64>>(),
            |mut map| {
                for &k in &reverse_keys {
                    map.remove(&k);
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_map_remove_random(c: &mut Criterion) {
    let keys = random_keys(N);

    let mut group = c.benchmark_group("map_remove_random");

    group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
        b.iter_batched(
            || keys.iter().map(|&k| (k, k)).collect::<OSBTreeMap<i64, i64>>(),
            |mut map| {
                for &k in &keys {
                    map.remove(&k);
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
        b.iter_batched(
            || keys.iter().map(|&k| (k, k)).collect::<BTreeMap<i64, i64>>(),
            |mut map| {
                for &k in &keys {
                    map.remove(&k);
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ─── Set Benchmarks ─────────────────────────────────────────────────────────

fn bench_set_insert_ordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_insert_ordered");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter(|| {
            let mut set = OSBTreeSet::new();
            for i in 0..N as i64 {
                set.insert(i);
            }
            set
        });
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter(|| {
            let mut set = BTreeSet::new();
            for i in 0..N as i64 {
                set.insert(i);
            }
            set
        });
    });

    group.finish();
}

fn bench_set_insert_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_insert_reverse");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter(|| {
            let mut set = OSBTreeSet::new();
            for i in (0..N as i64).rev() {
                set.insert(i);
            }
            set
        });
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter(|| {
            let mut set = BTreeSet::new();
            for i in (0..N as i64).rev() {
                set.insert(i);
            }
            set
        });
    });

    group.finish();
}

fn bench_set_insert_random(c: &mut Criterion) {
    let keys = random_keys(N);
    let mut group = c.benchmark_group("set_insert_random");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter(|| {
            let mut set = OSBTreeSet::new();
            for &k in &keys {
                set.insert(k);
            }
            set
        });
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter(|| {
            let mut set = BTreeSet::new();
            for &k in &keys {
                set.insert(k);
            }
            set
        });
    });

    group.finish();
}

fn bench_set_contains_ordered(c: &mut Criterion) {
    let keys = ordered_keys(N);
    let os_set: OSBTreeSet<i64> = keys.iter().copied().collect();
    let bt_set: BTreeSet<i64> = keys.iter().copied().collect();

    let mut group = c.benchmark_group("set_contains_ordered");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter(|| {
            let mut count = 0usize;
            for &k in &keys {
                if os_set.contains(&k) {
                    count += 1;
                }
            }
            count
        });
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter(|| {
            let mut count = 0usize;
            for &k in &keys {
                if bt_set.contains(&k) {
                    count += 1;
                }
            }
            count
        });
    });

    group.finish();
}

fn bench_set_contains_reverse(c: &mut Criterion) {
    let keys = ordered_keys(N);
    let os_set: OSBTreeSet<i64> = keys.iter().copied().collect();
    let bt_set: BTreeSet<i64> = keys.iter().copied().collect();
    let reverse_keys = reverse_ordered_keys(N);

    let mut group = c.benchmark_group("set_contains_reverse");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter(|| {
            let mut count = 0usize;
            for &k in &reverse_keys {
                if os_set.contains(&k) {
                    count += 1;
                }
            }
            count
        });
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter(|| {
            let mut count = 0usize;
            for &k in &reverse_keys {
                if bt_set.contains(&k) {
                    count += 1;
                }
            }
            count
        });
    });

    group.finish();
}

fn bench_set_contains_random(c: &mut Criterion) {
    let keys = random_keys(N);
    let os_set: OSBTreeSet<i64> = keys.iter().copied().collect();
    let bt_set: BTreeSet<i64> = keys.iter().copied().collect();

    let mut group = c.benchmark_group("set_contains_random");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter(|| {
            let mut count = 0usize;
            for &k in &keys {
                if os_set.contains(&k) {
                    count += 1;
                }
            }
            count
        });
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter(|| {
            let mut count = 0usize;
            for &k in &keys {
                if bt_set.contains(&k) {
                    count += 1;
                }
            }
            count
        });
    });

    group.finish();
}

fn bench_set_remove_ordered(c: &mut Criterion) {
    let keys = ordered_keys(N);

    let mut group = c.benchmark_group("set_remove_ordered");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter_batched(
            || keys.iter().copied().collect::<OSBTreeSet<i64>>(),
            |mut set| {
                for &k in &keys {
                    set.remove(&k);
                }
                set
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter_batched(
            || keys.iter().copied().collect::<BTreeSet<i64>>(),
            |mut set| {
                for &k in &keys {
                    set.remove(&k);
                }
                set
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_set_remove_reverse(c: &mut Criterion) {
    let keys = ordered_keys(N);
    let reverse_keys = reverse_ordered_keys(N);

    let mut group = c.benchmark_group("set_remove_reverse");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter_batched(
            || keys.iter().copied().collect::<OSBTreeSet<i64>>(),
            |mut set| {
                for &k in &reverse_keys {
                    set.remove(&k);
                }
                set
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter_batched(
            || keys.iter().copied().collect::<BTreeSet<i64>>(),
            |mut set| {
                for &k in &reverse_keys {
                    set.remove(&k);
                }
                set
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_set_remove_random(c: &mut Criterion) {
    let keys = random_keys(N);

    let mut group = c.benchmark_group("set_remove_random");

    group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
        b.iter_batched(
            || keys.iter().copied().collect::<OSBTreeSet<i64>>(),
            |mut set| {
                for &k in &keys {
                    set.remove(&k);
                }
                set
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
        b.iter_batched(
            || keys.iter().copied().collect::<BTreeSet<i64>>(),
            |mut set| {
                for &k in &keys {
                    set.remove(&k);
                }
                set
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ─── Set Relation Benchmarks ────────────────────────────────────────────────

fn bench_set_is_subset(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_is_subset");

    let all: Vec<i64> = ordered_keys(N);
    let evens: Vec<i64> = all.iter().copied().filter(|k| k % 2 == 0).collect();
    let sparse: Vec<i64> = all.iter().copied().step_by(100).collect();
    // Larger than `all` overall and covering its span, but missing one element near the end
    // of it: the range count rejects it in O(log n), while std looks up nearly every element.
    let missing_one: Vec<i64> =
        all.iter().copied().filter(|&k| k != N as i64 - 2).chain(N as i64..2 * N as i64).collect();
    let mut near_miss = evens.clone();
    *near_miss.last_mut().unwrap() += 1;
    // (case name, subset candidate, superset candidate)
    let cases: [(&str, &[i64], &[i64]); 4] = [
        ("stitch_true", &evens, &all),
        ("search_true", &sparse, &all),
        ("range_count_false", &all, &missing_one),
        ("stitch_late_miss", &near_miss, &all),
    ];

    for (name, sub, sup) in cases {
        let os_sub: OSBTreeSet<i64> = sub.iter().copied().collect();
        let os_sup: OSBTreeSet<i64> = sup.iter().copied().collect();
        let bt_sub: BTreeSet<i64> = sub.iter().copied().collect();
        let bt_sup: BTreeSet<i64> = sup.iter().copied().collect();
        assert_eq!(os_sub.is_subset(&os_sup), bt_sub.is_subset(&bt_sup));

        group.bench_function(BenchmarkId::new("OSBTreeSet", name), |b| {
            b.iter(|| std::hint::black_box(&os_sub).is_subset(std::hint::black_box(&os_sup)));
        });
        group.bench_function(BenchmarkId::new("BTreeSet", name), |b| {
            b.iter(|| std::hint::black_box(&bt_sub).is_subset(std::hint::black_box(&bt_sup)));
        });
    }

    group.finish();
}

// ─── Bulk construction (FromIterator) ────────────────────────────────────────

fn key_orders() -> [(&'static str, Vec<i64>); 3] {
    [("ordered", ordered_keys(N)), ("reverse", reverse_ordered_keys(N)), ("random", random_keys(N))]
}

fn bench_map_collect(c: &mut Criterion) {
    for (order, keys) in key_orders() {
        let mut group = c.benchmark_group(format!("map_collect_{order}"));
        group.bench_function(BenchmarkId::new("OSBTreeMap", N), |b| {
            b.iter(|| keys.iter().map(|&k| (k, k)).collect::<OSBTreeMap<i64, i64>>());
        });
        group.bench_function(BenchmarkId::new("BTreeMap", N), |b| {
            b.iter(|| keys.iter().map(|&k| (k, k)).collect::<BTreeMap<i64, i64>>());
        });
        group.finish();
    }
}

fn bench_set_collect(c: &mut Criterion) {
    for (order, keys) in key_orders() {
        let mut group = c.benchmark_group(format!("set_collect_{order}"));
        group.bench_function(BenchmarkId::new("OSBTreeSet", N), |b| {
            b.iter(|| keys.iter().copied().collect::<OSBTreeSet<i64>>());
        });
        group.bench_function(BenchmarkId::new("BTreeSet", N), |b| {
            b.iter(|| keys.iter().copied().collect::<BTreeSet<i64>>());
        });
        group.finish();
    }
}

// ─── Criterion Groups ───────────────────────────────────────────────────────

criterion_group!(map_insert_benches, bench_map_insert_ordered, bench_map_insert_reverse, bench_map_insert_random,);

criterion_group!(map_get_benches, bench_map_get_ordered, bench_map_get_reverse, bench_map_get_random,);

criterion_group!(map_remove_benches, bench_map_remove_ordered, bench_map_remove_reverse, bench_map_remove_random,);

criterion_group!(set_insert_benches, bench_set_insert_ordered, bench_set_insert_reverse, bench_set_insert_random,);

criterion_group!(
    set_contains_benches,
    bench_set_contains_ordered,
    bench_set_contains_reverse,
    bench_set_contains_random,
);

criterion_group!(set_remove_benches, bench_set_remove_ordered, bench_set_remove_reverse, bench_set_remove_random,);

criterion_group!(set_relation_benches, bench_set_is_subset);

criterion_group!(collect_benches, bench_map_collect, bench_set_collect);

criterion_main!(
    map_insert_benches,
    map_get_benches,
    map_remove_benches,
    set_insert_benches,
    set_contains_benches,
    set_remove_benches,
    set_relation_benches,
    collect_benches,
);

//! Tests that exercise every `unsafe` code path in the crate, sized to run quickly under Miri.
//!
//! Run with `cargo +nightly miri test --test miri` (Stacked Borrows) and again with
//! `MIRIFLAGS=-Zmiri-tree-borrows` (Tree Borrows). The trees are large enough to span several
//! leaves and two levels of the B+tree.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use std::collections::BTreeMap;

use wabi_tree::{OSBTreeMap, OSBTreeSet, Rank};

const N: u32 = 1_000;

fn map() -> OSBTreeMap<u32, u32> {
    (0..N).map(|i| (i, i)).collect()
}

/// Yields `next` and `next_back` alternately, the pattern most likely to expose an iterator
/// whose front and back cursors alias or cross incorrectly.
fn alternate<I: DoubleEndedIterator>(mut iter: I) -> Vec<I::Item> {
    let mut out = Vec::new();
    loop {
        let item = if out.len() % 2 == 0 {
            iter.next()
        } else {
            iter.next_back()
        };
        match item {
            Some(item) => out.push(item),
            None => return out,
        }
    }
}

#[test]
fn iter_mut_references_can_all_be_held_and_written() {
    let mut map = map();
    let refs: Vec<&mut u32> = map.iter_mut().map(|(_, v)| v).collect();
    for r in refs {
        *r += 1;
    }
    assert!(map.iter().all(|(k, v)| *v == k + 1));
}

#[test]
fn values_mut_references_can_all_be_held_and_written() {
    let mut map = map();
    let refs: Vec<&mut u32> = map.values_mut().collect();
    for r in refs {
        *r *= 2;
    }
    assert!(map.iter().all(|(k, v)| *v == k * 2));
}

#[test]
fn range_mut_references_can_all_be_held_and_written() {
    let mut map = map();
    let refs: Vec<(&u32, &mut u32)> = map.range_mut(100..900).collect();
    assert_eq!(refs.len(), 800);
    for (_, v) in refs {
        *v = N;
    }
    assert!(map.iter().all(|(k, v)| (*v == N) == (100..900).contains(k)));
}

// The double-ended tests below are one per iterator, rather than one test, so that nextest can
// run them in parallel; under Tree Borrows each takes tens of seconds.

#[test]
fn double_ended_iter_meets_in_the_middle() {
    assert_eq!(alternate(map().iter()).len(), N as usize);
}

#[test]
fn double_ended_range_meets_in_the_middle() {
    assert_eq!(alternate(map().range(10..=990)).len(), 981);
}

#[test]
fn double_ended_keys_meets_in_the_middle() {
    assert_eq!(alternate(map().keys()).len(), N as usize);
}

#[test]
fn double_ended_values_meets_in_the_middle() {
    assert_eq!(alternate(map().values()).len(), N as usize);
}

#[test]
fn double_ended_iter_mut_meets_in_the_middle() {
    let mut map = map();
    let expected: BTreeMap<u32, u32> = (0..N).map(|i| (i, i)).collect();
    let mut seen: Vec<(u32, u32)> = alternate(map.iter_mut()).into_iter().map(|(k, v)| (*k, *v)).collect();
    seen.sort_unstable();
    assert!(seen.into_iter().eq(expected));
}

#[test]
fn double_ended_range_mut_meets_in_the_middle() {
    let mut map = map();
    let mut seen: Vec<u32> = alternate(map.range_mut(..500)).into_iter().map(|(k, _)| *k).collect();
    seen.sort_unstable();
    assert!(seen.into_iter().eq(0..500));
}

#[test]
fn double_ended_values_mut_meets_in_the_middle() {
    assert_eq!(alternate(map().values_mut()).len(), N as usize);
}

#[test]
fn cloned_iterators_are_independent() {
    let map = map();
    let mut iter = map.iter();
    iter.next();
    let clone = iter.clone();
    assert!(iter.eq(clone));

    let mut range = map.range(5..50);
    range.next_back();
    let clone = range.clone();
    assert!(range.eq(clone));
}

#[test]
fn default_iterators_are_empty() {
    assert_eq!(wabi_tree::osbtree_map::Iter::<u32, u32>::default().next(), None);
    assert_eq!(wabi_tree::osbtree_map::Iter::<u32, u32>::default().next_back(), None);
    assert_eq!(wabi_tree::osbtree_map::IterMut::<u32, u32>::default().next(), None);
    assert_eq!(wabi_tree::osbtree_map::IterMut::<u32, u32>::default().next_back(), None);
    assert_eq!(wabi_tree::osbtree_map::Range::<u32, u32>::default().next(), None);
    assert_eq!(wabi_tree::osbtree_map::Range::<u32, u32>::default().next_back(), None);
    assert_eq!(wabi_tree::osbtree_map::RangeMut::<u32, u32>::default().next(), None);
    assert_eq!(wabi_tree::osbtree_map::RangeMut::<u32, u32>::default().next_back(), None);
    assert_eq!(wabi_tree::osbtree_map::ValuesMut::<u32, u32>::default().next(), None);
    assert_eq!(wabi_tree::osbtree_set::Iter::<u32>::default().next(), None);
    assert_eq!(wabi_tree::osbtree_set::Range::<u32>::default().next(), None);
}

#[test]
fn entries_and_rank_access_mutate_in_place() {
    let mut map = map();
    *map.entry(N).or_insert(0) += 7;
    *map.entry(3).or_insert(0) += 7;
    let held = map.entry(4).or_default();
    *held = 42;
    if let wabi_tree::osbtree_map::Entry::Occupied(o) = map.entry(5) {
        *o.into_mut() = 55;
    }
    *map.get_by_rank_mut(6).unwrap().1 = 66;
    map[Rank(7)] = 77;
    *map.get_with_rank_mut(&8).unwrap().2 = 88;
    assert_eq!(map.remove_by_rank(9), Some((9, 9)));
    assert_eq!(map.remove_with_rank(&10), Some((9, 10, 10)));

    assert_eq!(map.get(&N), Some(&7));
    assert_eq!(map.get(&3), Some(&10));
    assert_eq!(map.get(&4), Some(&42));
    assert_eq!(map.get(&5), Some(&55));
    assert_eq!(map.get(&6), Some(&66));
    assert_eq!(map.get(&7), Some(&77));
    assert_eq!(map.get_with_rank(&8), Some((8, &8, &88)));
    assert_eq!(map.get(&9), None);
}

#[test]
fn mutation_after_removals_and_clone() {
    let mut map = map();
    map.retain(|k, _| k % 3 != 0);
    let mut clone = map.clone();
    for (_, v) in &mut clone {
        *v += 1;
    }
    assert!(map.iter().zip(clone.iter()).all(|((k1, v1), (k2, v2))| k1 == k2 && *v2 == v1 + 1));

    let drained: Vec<(u32, u32)> = clone.into_iter().collect();
    assert_eq!(drained.len(), map.len());
}

#[test]
fn set_iterators() {
    let a: OSBTreeSet<u32> = (0..N).filter(|x| x % 2 == 0).collect();
    let b: OSBTreeSet<u32> = (0..N).filter(|x| x % 3 == 0).collect();

    assert_eq!(alternate(a.iter()).len(), a.len());
    assert_eq!(alternate(a.range(100..200)).len(), 50);
    assert_eq!(a.intersection(&b).count(), (0..N).filter(|x| x % 6 == 0).count());
    assert_eq!(a.union(&b).count(), (0..N).filter(|x| x % 2 == 0 || x % 3 == 0).count());
    assert_eq!(a.difference(&b).count(), (0..N).filter(|x| x % 2 == 0 && x % 3 != 0).count());
    assert_eq!(a.symmetric_difference(&b).count(), (0..N).filter(|x| (x % 2 == 0) != (x % 3 == 0)).count());
    assert_eq!(a.into_iter().next_back(), Some(N - 2));
}

/// Jumps of every size (within a leaf, to a sibling leaf, and through the tree) while holding
/// every entry already returned, then writes through all of them.
fn jump_hold_and_write<'a, I: DoubleEndedIterator<Item = (&'a u32, &'a mut u32)>>(mut iter: I) -> Vec<u32> {
    let mut held = Vec::new();
    for n in [0, 5, 100, 300] {
        held.extend(iter.nth(n));
        held.extend(iter.nth_back(n));
    }
    held.extend(iter);
    held.into_iter()
        .map(|(k, v)| {
            *v += 1;
            *k
        })
        .collect()
}

#[test]
fn nth_on_iter_mut_keeps_references_valid() {
    let mut map = map();
    let written = jump_hold_and_write(map.iter_mut());
    assert!(map.iter().all(|(k, v)| *v == k + u32::from(written.contains(k))));
}

#[test]
fn nth_on_range_mut_keeps_references_valid() {
    let mut map = map();
    let written = jump_hold_and_write(map.range_mut(10..990));
    assert!(written.iter().all(|k| (10..990).contains(k)));
    assert!(map.iter().all(|(k, v)| *v == k + u32::from(written.contains(k))));
}

#[test]
fn terminals_on_mutable_iterators_return_valid_references() {
    let mut map = map();
    let (k, v) = map.iter_mut().last().unwrap();
    *v += 1;
    let k = *k;
    let (_, v) = map.range_mut(10..990).max().unwrap();
    *v += 1;
    let v = map.values_mut().min().unwrap();
    *v += 1;
    assert_eq!(map[&k], k + 1);
    assert_eq!(map[&989], 990);
    assert_eq!(map.range_mut(..).count(), map.len());
}

/// `collect` sorts, drops discarded duplicates and clones separators. Heap-allocated keys and
/// values let Miri catch a leak or double drop on any of those paths.
#[test]
fn collect_with_duplicates_and_heap_keys() {
    let entries = (0..N).rev().map(|i| (format!("{:04}", i % 600), i.to_string()));
    let map: OSBTreeMap<String, String> = entries.clone().collect();
    let model: BTreeMap<String, String> = entries.collect();
    assert!(map.iter().eq(model.iter()));
    assert_eq!(alternate(map.iter()).len(), model.len());

    let set: OSBTreeSet<String> = model.into_keys().rev().collect();
    assert_eq!(set.len(), 600);
    assert_eq!(set.get_by_rank(599).map(String::as_str), Some("0599"));
}

/// Entries insert and remove through the path their search recorded (#32), including inserts
/// that split leaves and removals that merge them.
#[test]
fn entries_split_and_merge_through_recorded_path() {
    use wabi_tree::osbtree_map::Entry;

    let mut map = OSBTreeMap::new();
    for i in 0..400u32 {
        *map.entry(i * 7 % 400).or_insert(0) += i;
    }
    if let Entry::Vacant(e) = map.entry(1000) {
        assert_eq!(e.insert_entry(5).remove_entry(), (1000, 5));
    }
    for i in (0..400u32).step_by(2) {
        match map.entry(i) {
            Entry::Occupied(e) => assert_eq!(e.remove_entry().0, i),
            Entry::Vacant(_) => panic!("{i} was inserted"),
        }
    }
    assert_eq!(map.len(), 200);
    assert!(map.keys().copied().eq((1..400).step_by(2)));
}

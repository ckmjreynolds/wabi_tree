use std::collections::BTreeMap;

use proptest::prelude::*;
use wabi_tree::osbtree_map;
use wabi_tree::{OSBTreeMap, Rank};

/// The number of operations to perform in each proptest case.
const TEST_SIZE: usize = 10_000;

/// Generates a vector of random keys in the range suitable for causing collisions.
fn key_strategy() -> impl Strategy<Value = i64> {
    // Use a range that's smaller than TEST_SIZE to ensure key collisions
    -20_000i64..20_000i64
}

fn value_strategy() -> impl Strategy<Value = i64> {
    any::<i64>()
}

// ─── Operations enum for driving randomized tests ────────────────────────────

#[derive(Debug, Clone)]
enum MapOp {
    Insert(i64, i64),
    Remove(i64),
    Get(i64),
    ContainsKey(i64),
    GetKeyValue(i64),
    FirstKeyValue,
    LastKeyValue,
    PopFirst,
    PopLast,
}

fn map_op_strategy() -> impl Strategy<Value = MapOp> {
    prop_oneof![
        5 => (key_strategy(), value_strategy()).prop_map(|(k, v)| MapOp::Insert(k, v)),
        3 => key_strategy().prop_map(MapOp::Remove),
        2 => key_strategy().prop_map(MapOp::Get),
        1 => key_strategy().prop_map(MapOp::ContainsKey),
        1 => key_strategy().prop_map(MapOp::GetKeyValue),
        1 => Just(MapOp::FirstKeyValue),
        1 => Just(MapOp::LastKeyValue),
        1 => Just(MapOp::PopFirst),
        1 => Just(MapOp::PopLast),
    ]
}

// ─── Core CRUD operations ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Replays a random sequence of insert/remove/get operations on both
    /// OSBTreeMap and BTreeMap and asserts identical results at every step.
    #[test]
    fn map_ops_match_btreemap(ops in proptest::collection::vec(map_op_strategy(), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        for op in &ops {
            match op {
                MapOp::Insert(k, v) => {
                    let os_result = os_map.insert(*k, *v);
                    let bt_result = bt_map.insert(*k, *v);
                    prop_assert_eq!(os_result, bt_result, "insert({}, {})", k, v);
                }
                MapOp::Remove(k) => {
                    let os_result = os_map.remove(k);
                    let bt_result = bt_map.remove(k);
                    prop_assert_eq!(os_result, bt_result, "remove({})", k);
                }
                MapOp::Get(k) => {
                    let os_result = os_map.get(k);
                    let bt_result = bt_map.get(k);
                    prop_assert_eq!(os_result, bt_result, "get({})", k);
                }
                MapOp::ContainsKey(k) => {
                    let os_result = os_map.contains_key(k);
                    let bt_result = bt_map.contains_key(k);
                    prop_assert_eq!(os_result, bt_result, "contains_key({})", k);
                }
                MapOp::GetKeyValue(k) => {
                    let os_result = os_map.get_key_value(k);
                    let bt_result = bt_map.get_key_value(k);
                    prop_assert_eq!(os_result, bt_result, "get_key_value({})", k);
                }
                MapOp::FirstKeyValue => {
                    let os_result = os_map.first_key_value();
                    let bt_result = bt_map.first_key_value();
                    prop_assert_eq!(os_result, bt_result, "first_key_value");
                }
                MapOp::LastKeyValue => {
                    let os_result = os_map.last_key_value();
                    let bt_result = bt_map.last_key_value();
                    prop_assert_eq!(os_result, bt_result, "last_key_value");
                }
                MapOp::PopFirst => {
                    let os_result = os_map.pop_first();
                    let bt_result = bt_map.pop_first();
                    prop_assert_eq!(os_result, bt_result, "pop_first");
                }
                MapOp::PopLast => {
                    let os_result = os_map.pop_last();
                    let bt_result = bt_map.pop_last();
                    prop_assert_eq!(os_result, bt_result, "pop_last");
                }
            }
            prop_assert_eq!(os_map.len(), bt_map.len(), "len mismatch after {:?}", op);
            prop_assert_eq!(os_map.is_empty(), bt_map.is_empty(), "is_empty mismatch after {:?}", op);
        }
    }

    /// Tests that iteration order matches BTreeMap after random insertions.
    #[test]
    fn iter_matches_btreemap(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        for (k, v) in &entries {
            os_map.insert(*k, *v);
            bt_map.insert(*k, *v);
        }

        // Forward iteration
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "iter() mismatch");

        // Reverse iteration
        let os_rev: Vec<_> = os_map.iter().rev().map(|(&k, &v)| (k, v)).collect();
        let bt_rev: Vec<_> = bt_map.iter().rev().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_rev, &bt_rev, "iter().rev() mismatch");

        // Keys
        let os_keys: Vec<_> = os_map.keys().copied().collect();
        let bt_keys: Vec<_> = bt_map.keys().copied().collect();
        prop_assert_eq!(&os_keys, &bt_keys, "keys() mismatch");

        // Values
        let os_vals: Vec<_> = os_map.values().copied().collect();
        let bt_vals: Vec<_> = bt_map.values().copied().collect();
        prop_assert_eq!(&os_vals, &bt_vals, "values() mismatch");

        // into_iter
        let os_into: Vec<_> = os_map.clone().into_iter().collect();
        let bt_into: Vec<_> = bt_map.clone().into_iter().collect();
        prop_assert_eq!(&os_into, &bt_into, "into_iter() mismatch");

        // into_keys
        let os_into_keys: Vec<_> = os_map.clone().into_keys().collect();
        let bt_into_keys: Vec<_> = bt_map.clone().into_keys().collect();
        prop_assert_eq!(&os_into_keys, &bt_into_keys, "into_keys() mismatch");

        // into_values
        let os_into_vals: Vec<_> = os_map.clone().into_values().collect();
        let bt_into_vals: Vec<_> = bt_map.clone().into_values().collect();
        prop_assert_eq!(&os_into_vals, &bt_into_vals, "into_values() mismatch");
    }

    /// Tests ExactSizeIterator and DoubleEndedIterator behavior.
    #[test]
    fn iter_size_and_double_ended(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();

        let iter = os_map.iter();
        let len = iter.len();
        prop_assert_eq!(len, os_map.len(), "ExactSizeIterator len mismatch");

        // Alternating front/back should yield all elements
        let mut from_front = Vec::new();
        let mut from_back = Vec::new();
        let mut iter = os_map.iter();
        let mut toggle = true;
        loop {
            if toggle {
                if let Some(item) = iter.next() {
                    from_front.push(item);
                } else {
                    break;
                }
            } else if let Some(item) = iter.next_back() {
                from_back.push(item);
            } else {
                break;
            }
            toggle = !toggle;
        }
        prop_assert_eq!(from_front.len() + from_back.len(), os_map.len());
    }

    /// Tests range queries match BTreeMap.
    #[test]
    fn range_matches_btreemap(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        for (k, v) in &entries {
            os_map.insert(*k, *v);
            bt_map.insert(*k, *v);
        }

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Inclusive range
        let os_range: Vec<_> = os_map.range(lo..=hi).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(lo..=hi).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..={}) mismatch", lo, hi);

        // Exclusive end
        let os_range: Vec<_> = os_map.range(lo..hi).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(lo..hi).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..{}) mismatch", lo, hi);

        // From start
        let os_range: Vec<_> = os_map.range(lo..).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(lo..).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..) mismatch", lo);

        // Up to end
        let os_range: Vec<_> = os_map.range(..=hi).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(..=hi).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range(..={}) mismatch", hi);

        // Unbounded
        let os_range: Vec<_> = os_map.range(..).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(..).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range(..) mismatch");

        // Reverse range
        let os_range_rev: Vec<_> = os_map.range(lo..=hi).rev().map(|(&k, &v)| (k, v)).collect();
        let bt_range_rev: Vec<_> = bt_map.range(lo..=hi).rev().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range_rev, &bt_range_rev, "range({}..={}).rev() mismatch", lo, hi);
    }

    /// Tests get_mut and range_mut behave correctly.
    #[test]
    fn get_mut_matches_btreemap(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        keys_to_mutate in proptest::collection::vec(key_strategy(), 100),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        for (k, v) in &entries {
            os_map.insert(*k, *v);
            bt_map.insert(*k, *v);
        }

        for k in &keys_to_mutate {
            if let Some(v) = os_map.get_mut(k) {
                *v += 1;
            }
            if let Some(v) = bt_map.get_mut(k) {
                *v += 1;
            }
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "get_mut mismatch");
    }

    /// Tests retain matches BTreeMap.
    #[test]
    fn retain_matches_btreemap(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        os_map.retain(|k, _v| k % 3 != 0);
        bt_map.retain(|k, _v| k % 3 != 0);

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "retain mismatch");
        prop_assert_eq!(os_map.len(), bt_map.len(), "retain len mismatch");
    }

    /// Tests append matches BTreeMap.
    #[test]
    fn append_matches_btreemap(
        entries_a in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        entries_b in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
    ) {
        let mut os_a: OSBTreeMap<i64, i64> = entries_a.iter().cloned().collect();
        let mut os_b: OSBTreeMap<i64, i64> = entries_b.iter().cloned().collect();
        let mut bt_a: BTreeMap<i64, i64> = entries_a.iter().cloned().collect();
        let mut bt_b: BTreeMap<i64, i64> = entries_b.iter().cloned().collect();

        os_a.append(&mut os_b);
        bt_a.append(&mut bt_b);

        prop_assert_eq!(os_b.len(), 0, "append did not empty source");
        prop_assert_eq!(os_a.len(), bt_a.len(), "append len mismatch");

        let os_items: Vec<_> = os_a.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_a.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "append content mismatch");
    }

    /// Tests split_off matches BTreeMap.
    #[test]
    fn split_off_matches_btreemap(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        split_key in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let os_right = os_map.split_off(&split_key);
        let bt_right = bt_map.split_off(&split_key);

        let os_left_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_left_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_left_items, &bt_left_items, "split_off left mismatch");

        let os_right_items: Vec<_> = os_right.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_right_items: Vec<_> = bt_right.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_right_items, &bt_right_items, "split_off right mismatch");
    }

    /// Tests that clear produces an empty map.
    #[test]
    fn clear_empties_map(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        os_map.clear();
        prop_assert!(os_map.is_empty());
        prop_assert_eq!(os_map.len(), 0);
        prop_assert_eq!(os_map.iter().count(), 0);
    }

    /// Tests the Entry API matches BTreeMap behavior.
    #[test]
    fn entry_api_matches_btreemap(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        entry_keys in proptest::collection::vec(key_strategy(), TEST_SIZE / 2),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = initial.iter().cloned().collect();

        for k in &entry_keys {
            // or_insert
            let os_val = *os_map.entry(*k).or_insert(999);
            let bt_val = *bt_map.entry(*k).or_insert(999);
            prop_assert_eq!(os_val, bt_val, "entry({}).or_insert", k);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "entry API content mismatch");
    }

    /// Tests and_modify + or_insert pattern.
    #[test]
    fn entry_and_modify_or_insert(
        keys in proptest::collection::vec(key_strategy(), TEST_SIZE),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        for k in &keys {
            os_map.entry(*k).and_modify(|v| *v += 1).or_insert(1);
            bt_map.entry(*k).and_modify(|v| *v += 1).or_insert(1);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "and_modify/or_insert mismatch");
    }

    /// Tests or_insert_with matches BTreeMap.
    #[test]
    fn entry_or_insert_with(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        keys in proptest::collection::vec(key_strategy(), TEST_SIZE / 2),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = initial.iter().cloned().collect();

        for k in &keys {
            let os_val = *os_map.entry(*k).or_insert_with(|| k.wrapping_mul(2));
            let bt_val = *bt_map.entry(*k).or_insert_with(|| k.wrapping_mul(2));
            prop_assert_eq!(os_val, bt_val, "or_insert_with({}) value mismatch", k);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "or_insert_with content mismatch");
    }

    /// Tests or_insert_with_key matches BTreeMap.
    #[test]
    fn entry_or_insert_with_key(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        keys in proptest::collection::vec(key_strategy(), TEST_SIZE / 2),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = initial.iter().cloned().collect();

        for k in &keys {
            let os_val = *os_map.entry(*k).or_insert_with_key(|key| key.wrapping_add(100));
            let bt_val = *bt_map.entry(*k).or_insert_with_key(|key| key.wrapping_add(100));
            prop_assert_eq!(os_val, bt_val, "or_insert_with_key({}) value mismatch", k);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "or_insert_with_key content mismatch");
    }

    /// Tests or_default matches BTreeMap.
    #[test]
    fn entry_or_default(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        keys in proptest::collection::vec(key_strategy(), TEST_SIZE / 2),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = initial.iter().cloned().collect();

        for k in &keys {
            let os_val = *os_map.entry(*k).or_default();
            let bt_val = *bt_map.entry(*k).or_default();
            prop_assert_eq!(os_val, bt_val, "or_default({}) value mismatch", k);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "or_default content mismatch");
    }

    /// Tests insert_entry behavior.
    #[test]
    fn entry_insert_entry(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        insertions in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();

        for (k, v) in &insertions {
            let os_entry = os_map.entry(*k).insert_entry(*v);
            // Verify the entry has the correct key and value
            prop_assert_eq!(*os_entry.key(), *k, "insert_entry key mismatch");
            prop_assert_eq!(*os_entry.get(), *v, "insert_entry value mismatch");
        }

        // Verify all insertions are in the map with correct values
        // (later insertions overwrite earlier ones for duplicate keys)
        let expected: BTreeMap<i64, i64> = insertions.iter().cloned().collect();
        for (k, v) in &expected {
            prop_assert_eq!(os_map.get(k), Some(v), "insert_entry final value mismatch for key {}", k);
        }
    }

    /// Tests VacantEntry::into_key returns the correct key.
    #[test]
    fn vacant_entry_into_key(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        new_keys in proptest::collection::vec(key_strategy(), 100),
    ) {
        let os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();

        for k in &new_keys {
            if !os_map.contains_key(k) {
                // Create a fresh map for each test to get a VacantEntry
                let mut test_map = os_map.clone();
                if let wabi_tree::osbtree_map::Entry::Vacant(v) = test_map.entry(*k) {
                    let returned_key = v.into_key();
                    prop_assert_eq!(returned_key, *k, "into_key() returned wrong key");
                }
            }
        }
    }

    /// Tests FromIterator and From<[T; N]>.
    #[test]
    fn from_iter_matches_btreemap(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "FromIterator mismatch");
    }

    /// Tests Clone produces an equal map.
    #[test]
    fn clone_produces_equal_map(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let cloned = os_map.clone();

        prop_assert_eq!(os_map.len(), cloned.len());
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let cl_items: Vec<_> = cloned.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &cl_items, "clone content mismatch");
    }

    /// Tests PartialEq / Eq.
    #[test]
    fn eq_matches_btreemap(
        entries_a in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        entries_b in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeMap<i64, i64> = entries_a.iter().cloned().collect();
        let os_b: OSBTreeMap<i64, i64> = entries_b.iter().cloned().collect();
        let bt_a: BTreeMap<i64, i64> = entries_a.iter().cloned().collect();
        let bt_b: BTreeMap<i64, i64> = entries_b.iter().cloned().collect();

        prop_assert_eq!(os_a == os_b, bt_a == bt_b, "equality mismatch");
    }

    /// Tests Ord / PartialOrd.
    #[test]
    fn ord_matches_btreemap(
        entries_a in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        entries_b in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeMap<i64, i64> = entries_a.iter().cloned().collect();
        let os_b: OSBTreeMap<i64, i64> = entries_b.iter().cloned().collect();
        let bt_a: BTreeMap<i64, i64> = entries_a.iter().cloned().collect();
        let bt_b: BTreeMap<i64, i64> = entries_b.iter().cloned().collect();

        prop_assert_eq!(os_a.cmp(&os_b), bt_a.cmp(&bt_b), "Ord mismatch");
        prop_assert_eq!(os_a.partial_cmp(&os_b), bt_a.partial_cmp(&bt_b), "PartialOrd mismatch");
    }

    /// Tests Index<&Q> panics/returns same as BTreeMap.
    #[test]
    fn index_by_key_matches_btreemap(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        for (k, _) in &entries {
            prop_assert_eq!(os_map[k], bt_map[k], "Index[&{}] mismatch", k);
        }
    }

    /// Tests extract_if matches the expected behavior.
    #[test]
    fn extract_if_matches_expected(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let original_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let os_extracted: Vec<_> = os_map.extract_if(.., |k, _v| k % 2 == 0).collect();

        // Verify extracted elements have even keys and are in sorted order
        let mut prev_key = None;
        for (k, v) in &os_extracted {
            prop_assert!(k % 2 == 0, "extracted key {} should be even", k);
            prop_assert_eq!(original_map.get(k), Some(v), "extracted value mismatch for key {}", k);
            if let Some(prev) = prev_key {
                prop_assert!(k > &prev, "extracted keys should be in sorted order");
            }
            prev_key = Some(*k);
        }

        // Verify remaining elements have odd keys
        for (k, v) in os_map.iter() {
            prop_assert!(k % 2 != 0, "remaining key {} should be odd", k);
            prop_assert_eq!(original_map.get(k), Some(v), "remaining value mismatch for key {}", k);
        }

        // Verify total count matches
        prop_assert_eq!(os_extracted.len() + os_map.len(), original_map.len(), "total count mismatch");
    }
}

// ─── Order-statistic operations (compared against Vec) ───────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests get_by_rank against a sorted Vec oracle.
    #[test]
    fn get_by_rank_matches_vec(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let sorted: Vec<(i64, i64)> = BTreeMap::from_iter(entries.iter().cloned())
            .into_iter()
            .collect();

        prop_assert_eq!(os_map.len(), sorted.len());

        for (rank, (ek, ev)) in sorted.iter().enumerate() {
            let os_result = os_map.get_by_rank(rank);
            let expected = Some((ek, ev));
            prop_assert_eq!(
                os_result, expected,
                "get_by_rank({}) mismatch: got {:?}, expected {:?}", rank, os_result, expected
            );
        }

        // Out of bounds should return None
        prop_assert_eq!(os_map.get_by_rank(sorted.len()), None);
        prop_assert_eq!(os_map.get_by_rank(sorted.len() + 100), None);
    }

    /// Tests get_by_rank_mut against a sorted Vec oracle.
    #[test]
    fn get_by_rank_mut_matches_vec(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let sorted: Vec<(i64, i64)> = BTreeMap::from_iter(entries.iter().cloned())
            .into_iter()
            .collect();

        // Verify keys match, then mutate via rank
        for (rank, (expected_k, _)) in sorted.iter().enumerate() {
            if let Some((k, v)) = os_map.get_by_rank_mut(rank) {
                prop_assert_eq!(*k, *expected_k, "get_by_rank_mut({}) key mismatch", rank);
                *v = rank as i64; // mutate
            } else {
                prop_assert!(false, "get_by_rank_mut({}) returned None unexpectedly", rank);
            }
        }

        // Verify mutations stuck
        for (rank, _) in sorted.iter().enumerate() {
            let (_, v) = os_map.get_by_rank(rank).unwrap();
            prop_assert_eq!(*v, rank as i64, "mutation at rank {} did not persist", rank);
        }
    }

    /// Tests rank_of against a sorted Vec oracle.
    #[test]
    fn rank_of_matches_vec(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let sorted: Vec<(i64, i64)> = BTreeMap::from_iter(entries.iter().cloned())
            .into_iter()
            .collect();

        // Every key in the map should have the correct rank
        for (expected_rank, (k, _)) in sorted.iter().enumerate() {
            let rank = os_map.rank_of(k);
            prop_assert_eq!(rank, Some(expected_rank), "rank_of({})", k);
        }

        // Keys not in the map should return None
        for probe in [i64::MIN, i64::MAX, 99999, -99999] {
            if !os_map.contains_key(&probe) {
                prop_assert_eq!(os_map.rank_of(&probe), None, "rank_of({}) should be None", probe);
            }
        }
    }

    /// Tests Index<Rank> and IndexMut<Rank>.
    #[test]
    fn index_by_rank_matches_vec(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let sorted: Vec<(i64, i64)> = BTreeMap::from_iter(entries.iter().cloned())
            .into_iter()
            .collect();

        // Index<Rank> for reading
        for (rank, (_, expected_v)) in sorted.iter().enumerate() {
            prop_assert_eq!(os_map[Rank(rank)], *expected_v, "Index[Rank({})]", rank);
        }

        // IndexMut<Rank> for writing
        if !sorted.is_empty() {
            os_map[Rank(0)] = 42;
            prop_assert_eq!(os_map[Rank(0)], 42, "IndexMut[Rank(0)]");
        }
    }

    /// Tests that rank_of and get_by_rank are consistent with each other.
    #[test]
    fn rank_of_get_by_rank_roundtrip(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();

        for rank in 0..os_map.len() {
            let (k, _v) = os_map.get_by_rank(rank).unwrap();
            let recovered_rank = os_map.rank_of(k).unwrap();
            prop_assert_eq!(recovered_rank, rank, "roundtrip rank mismatch at rank {}", rank);
        }
    }

    /// Tests order-statistic operations after a mix of inserts and removes.
    #[test]
    fn order_stats_after_mutations(ops in proptest::collection::vec(map_op_strategy(), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        for op in &ops {
            match op {
                MapOp::Insert(k, v) => {
                    os_map.insert(*k, *v);
                    bt_map.insert(*k, *v);
                }
                MapOp::Remove(k) => {
                    os_map.remove(k);
                    bt_map.remove(k);
                }
                _ => {}
            }
        }

        let sorted: Vec<(i64, i64)> = bt_map.into_iter().collect();
        prop_assert_eq!(os_map.len(), sorted.len());

        // Spot-check ranks at various positions
        let check_positions = [0, 1, sorted.len() / 4, sorted.len() / 2, sorted.len() * 3 / 4, sorted.len().saturating_sub(1)];
        for &pos in &check_positions {
            if pos < sorted.len() {
                let os_result = os_map.get_by_rank(pos);
                let expected = Some((&sorted[pos].0, &sorted[pos].1));
                prop_assert_eq!(os_result, expected, "get_by_rank({}) after mutations", pos);

                let rank = os_map.rank_of(&sorted[pos].0);
                prop_assert_eq!(rank, Some(pos), "rank_of after mutations at pos {}", pos);
            }
        }
    }
}

// ─── Extend and iter_mut ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests Extend matches BTreeMap.
    #[test]
    fn extend_matches_btreemap(
        initial in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
        extra in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE / 2),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = initial.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = initial.iter().cloned().collect();

        os_map.extend(extra.iter().cloned());
        bt_map.extend(extra.iter().cloned());

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "extend mismatch");
    }

    /// Tests iter_mut produces the same sequence and allows mutation.
    #[test]
    fn iter_mut_matches(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Mutate all values
        for (_, v) in os_map.iter_mut() {
            *v = v.wrapping_add(1);
        }
        for (_, v) in bt_map.iter_mut() {
            *v = v.wrapping_add(1);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "iter_mut mismatch");
    }

    /// Tests IterMut double-ended traversal with alternating next/next_back.
    #[test]
    fn iter_mut_double_ended_traversal(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Collect keys using alternating next/next_back, mutating values as we go
        let mut os_keys = Vec::new();
        let mut bt_keys = Vec::new();

        {
            let mut os_iter = os_map.iter_mut();
            let mut bt_iter = bt_map.iter_mut();

            let mut toggle = true;
            loop {
                if toggle {
                    match (os_iter.next(), bt_iter.next()) {
                        (Some((os_k, os_v)), Some((bt_k, bt_v))) => {
                            prop_assert_eq!(*os_k, *bt_k, "iter_mut next() key mismatch");
                            prop_assert_eq!(*os_v, *bt_v, "iter_mut next() value mismatch");
                            os_keys.push(*os_k);
                            bt_keys.push(*bt_k);
                            // Mutate the value
                            *os_v = os_v.wrapping_add(100);
                            *bt_v = bt_v.wrapping_add(100);
                        }
                        (None, None) => break,
                        (os, bt) => {
                            prop_assert!(false, "iter_mut next() mismatch: os={:?}, bt={:?}",
                                os.map(|(k, _)| k), bt.map(|(k, _)| k));
                        }
                    }
                } else {
                    match (os_iter.next_back(), bt_iter.next_back()) {
                        (Some((os_k, os_v)), Some((bt_k, bt_v))) => {
                            prop_assert_eq!(*os_k, *bt_k, "iter_mut next_back() key mismatch");
                            prop_assert_eq!(*os_v, *bt_v, "iter_mut next_back() value mismatch");
                            os_keys.push(*os_k);
                            bt_keys.push(*bt_k);
                            // Mutate the value
                            *os_v = os_v.wrapping_add(200);
                            *bt_v = bt_v.wrapping_add(200);
                        }
                        (None, None) => break,
                        (os, bt) => {
                            prop_assert!(false, "iter_mut next_back() mismatch: os={:?}, bt={:?}",
                                os.map(|(k, _)| k), bt.map(|(k, _)| k));
                        }
                    }
                }
                toggle = !toggle;
            }
        }

        // Verify total elements match
        prop_assert_eq!(os_keys.len(), bt_keys.len(), "iter_mut double-ended total count mismatch");
        prop_assert_eq!(os_keys.len(), os_map.len(), "iter_mut should visit all elements");

        // Verify mutations were applied correctly
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "iter_mut double-ended mutations mismatch");

        // Verify no duplicates
        let mut os_keys_sorted = os_keys.clone();
        os_keys_sorted.sort();
        let dedup_len = os_keys_sorted.len();
        os_keys_sorted.dedup();
        prop_assert_eq!(os_keys_sorted.len(), dedup_len, "iter_mut yielded duplicate keys");
    }

    /// Tests values_mut produces the same result.
    #[test]
    fn values_mut_matches(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        for v in os_map.values_mut() {
            *v = v.wrapping_mul(2);
        }
        for v in bt_map.values_mut() {
            *v = v.wrapping_mul(2);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "values_mut mismatch");
    }

    /// Tests range_mut matches expected behavior.
    #[test]
    fn range_mut_matches(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        for (_, v) in os_map.range_mut(lo..=hi) {
            *v = v.wrapping_add(100);
        }
        for (_, v) in bt_map.range_mut(lo..=hi) {
            *v = v.wrapping_add(100);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "range_mut mismatch");
    }
}

// ─── first_entry / last_entry ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests first_entry and last_entry.
    #[test]
    fn first_last_entry_matches(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // first_entry
        if let Some(entry) = os_map.first_entry() {
            let bt_first = bt_map.first_key_value().unwrap();
            prop_assert_eq!(entry.key(), bt_first.0, "first_entry key");
            prop_assert_eq!(entry.get(), bt_first.1, "first_entry value");
        } else {
            prop_assert!(bt_map.is_empty());
        }

        // last_entry
        if let Some(entry) = os_map.last_entry() {
            let bt_last = bt_map.last_key_value().unwrap();
            prop_assert_eq!(entry.key(), bt_last.0, "last_entry key");
            prop_assert_eq!(entry.get(), bt_last.1, "last_entry value");
        } else {
            prop_assert!(bt_map.is_empty());
        }
    }

    /// Tests first_entry mutation via get_mut and insert.
    #[test]
    fn first_entry_mutation(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Mutate first entry using get_mut
        if let Some(mut entry) = os_map.first_entry() {
            *entry.get_mut() = 999_999;
        }
        if let Some(mut entry) = bt_map.first_entry() {
            *entry.get_mut() = 999_999;
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "first_entry get_mut mismatch");

        // Mutate first entry using insert
        if let Some(mut entry) = os_map.first_entry() {
            let old = entry.insert(888_888);
            prop_assert_eq!(old, 999_999, "first_entry insert should return old value");
        }
        if let Some(mut entry) = bt_map.first_entry() {
            entry.insert(888_888);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "first_entry insert mismatch");
    }

    /// Tests last_entry mutation via get_mut and insert.
    #[test]
    fn last_entry_mutation(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Mutate last entry using get_mut
        if let Some(mut entry) = os_map.last_entry() {
            *entry.get_mut() = 999_999;
        }
        if let Some(mut entry) = bt_map.last_entry() {
            *entry.get_mut() = 999_999;
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "last_entry get_mut mismatch");

        // Mutate last entry using insert
        if let Some(mut entry) = os_map.last_entry() {
            let old = entry.insert(888_888);
            prop_assert_eq!(old, 999_999, "last_entry insert should return old value");
        }
        if let Some(mut entry) = bt_map.last_entry() {
            entry.insert(888_888);
        }

        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "last_entry insert mismatch");
    }

    /// Tests first_entry remove and remove_entry.
    #[test]
    fn first_entry_remove(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Remove first entry
        let os_result = os_map.first_entry().map(|e| e.remove_entry());
        let bt_result = bt_map.first_entry().map(|e| e.remove_entry());
        prop_assert_eq!(os_result, bt_result, "first_entry remove_entry mismatch");

        // Verify remaining maps match
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "first_entry remove content mismatch");
    }

    /// Tests last_entry remove and remove_entry.
    #[test]
    fn last_entry_remove(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Remove last entry
        let os_result = os_map.last_entry().map(|e| e.remove_entry());
        let bt_result = bt_map.last_entry().map(|e| e.remove_entry());
        prop_assert_eq!(os_result, bt_result, "last_entry remove_entry mismatch");

        // Verify remaining maps match
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "last_entry remove content mismatch");
    }

    /// Tests remove_entry matches BTreeMap.
    #[test]
    fn remove_entry_matches_btreemap(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        keys_to_remove in proptest::collection::vec(key_strategy(), TEST_SIZE / 5),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        for k in &keys_to_remove {
            let os_result = os_map.remove_entry(k);
            let bt_result = bt_map.remove_entry(k);
            prop_assert_eq!(os_result, bt_result, "remove_entry({})", k);
        }

        prop_assert_eq!(os_map.len(), bt_map.len());
    }
}

// ─── Hash consistency ────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests that equal maps produce equal hashes.
    #[test]
    fn hash_consistent_for_equal_maps(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        use std::hash::{DefaultHasher, Hash, Hasher};

        let os_map1: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let os_map2: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        os_map1.hash(&mut h1);
        os_map2.hash(&mut h2);

        prop_assert_eq!(h1.finish(), h2.finish(), "equal maps should have equal hashes");
    }
}

// ─── Range edge cases (empty ranges, leaf boundaries, tuple bounds) ──────────

use core::ops::Bound;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests range with tuple bounds using Excluded/Included combinations matches BTreeMap.
    #[test]
    fn range_tuple_bounds_match_btreemap(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // (Included, Included)
        let os_range: Vec<_> = os_map.range((Bound::Included(lo), Bound::Included(hi))).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range((Bound::Included(lo), Bound::Included(hi))).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Included({}))) mismatch", lo, hi);

        // (Included, Excluded)
        let os_range: Vec<_> = os_map.range((Bound::Included(lo), Bound::Excluded(hi))).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range((Bound::Included(lo), Bound::Excluded(hi))).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Excluded({}))) mismatch", lo, hi);

        // (Excluded, Included)
        let os_range: Vec<_> = os_map.range((Bound::Excluded(lo), Bound::Included(hi))).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range((Bound::Excluded(lo), Bound::Included(hi))).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Excluded({}), Included({}))) mismatch", lo, hi);

        // (Excluded, Excluded) - only valid if lo < hi
        if lo < hi {
            let os_range: Vec<_> = os_map.range((Bound::Excluded(lo), Bound::Excluded(hi))).map(|(&k, &v)| (k, v)).collect();
            let bt_range: Vec<_> = bt_map.range((Bound::Excluded(lo), Bound::Excluded(hi))).map(|(&k, &v)| (k, v)).collect();
            prop_assert_eq!(&os_range, &bt_range, "range((Excluded({}), Excluded({}))) mismatch", lo, hi);
        }

        // (Unbounded, Included)
        let os_range: Vec<_> = os_map.range((Bound::<i64>::Unbounded, Bound::Included(hi))).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range((Bound::<i64>::Unbounded, Bound::Included(hi))).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Unbounded, Included({}))) mismatch", hi);

        // (Included, Unbounded)
        let os_range: Vec<_> = os_map.range((Bound::Included(lo), Bound::<i64>::Unbounded)).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range((Bound::Included(lo), Bound::<i64>::Unbounded)).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Unbounded)) mismatch", lo);
    }

    /// Tests range(k..k) produces empty range (empty range at any key).
    #[test]
    fn range_empty_at_key_matches_btreemap(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        key in key_strategy(),
    ) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        // range(k..k) should always be empty
        let os_range: Vec<_> = os_map.range(key..key).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(key..key).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..{}) should be empty", key, key);
        prop_assert!(os_range.is_empty(), "range(k..k) must be empty");

        // Also test with explicit bounds
        let os_range: Vec<_> = os_map.range((Bound::Included(key), Bound::Excluded(key))).map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range((Bound::Included(key), Bound::Excluded(key))).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Excluded({}))) should be empty", key, key);
    }

    /// Tests range_mut only mutates values within the specified bounds.
    #[test]
    fn range_mut_only_affects_in_range(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Mutate values in range
        for (_, v) in os_map.range_mut(lo..=hi) {
            *v = 999_999;
        }

        // Verify only in-range values were changed
        for (k, v) in os_map.iter() {
            if *k >= lo && *k <= hi {
                prop_assert_eq!(*v, 999_999, "in-range key {} should be mutated", k);
            } else {
                let expected = bt_map.get(k).unwrap();
                prop_assert_eq!(v, expected, "out-of-range key {} should not be mutated", k);
            }
        }
    }

    /// Tests range next_back doesn't escape bounds.
    #[test]
    fn range_next_back_respects_bounds(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Collect using next_back only
        let os_range: Vec<_> = os_map.range(lo..=hi).rev().map(|(&k, &v)| (k, v)).collect();
        let bt_range: Vec<_> = bt_map.range(lo..=hi).rev().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..={}).rev() mismatch", lo, hi);

        // Verify all collected keys are in bounds
        for (k, _) in &os_range {
            prop_assert!(*k >= lo && *k <= hi, "key {} is outside range {}..={}", k, lo, hi);
        }
    }

    /// Tests extract_if with bounded range only removes keys in range.
    #[test]
    fn extract_if_bounded_only_removes_in_range(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let original_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Extract all even keys in range
        let os_extracted: Vec<_> = os_map.extract_if(lo..=hi, |k, _| k % 2 == 0).collect();

        // Verify extracted keys are in bounds and even, in sorted order
        let mut prev_key = None;
        for (k, v) in &os_extracted {
            prop_assert!(*k >= lo && *k <= hi, "extracted key {} is outside range {}..={}", k, lo, hi);
            prop_assert!(k % 2 == 0, "extracted key {} should be even", k);
            prop_assert_eq!(original_map.get(k), Some(v), "extracted value mismatch for key {}", k);
            if let Some(prev) = prev_key {
                prop_assert!(k > &prev, "extracted keys should be in sorted order");
            }
            prev_key = Some(*k);
        }

        // Verify remaining keys are either outside range or odd
        for (k, v) in os_map.iter() {
            let in_range = *k >= lo && *k <= hi;
            if in_range {
                prop_assert!(k % 2 != 0, "remaining key {} in range should be odd", k);
            }
            prop_assert_eq!(original_map.get(k), Some(v), "remaining value mismatch for key {}", k);
        }
    }

    /// Tests extract_if with early drop (iterator not exhausted) retains unvisited keys.
    #[test]
    fn extract_if_early_drop_retains_unvisited(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let original_len = os_map.len();

        // Take only the first 10 items from extract_if, then drop
        let os_extracted: Vec<_> = os_map.extract_if(.., |_, _| true).take(10).collect();

        // Extracted count should be at most 10
        prop_assert!(os_extracted.len() <= 10, "should extract at most 10 items");

        // Remaining items should be original_len - extracted_len
        prop_assert_eq!(os_map.len(), original_len - os_extracted.len(), "remaining count mismatch");

        // Extracted items should be the first N items in sorted order
        let sorted_unique: Vec<_> = {
            let map: BTreeMap<_, _> = entries.iter().cloned().collect();
            map.keys().copied().collect()
        };
        for (i, (k, _)) in os_extracted.iter().enumerate() {
            prop_assert_eq!(*k, sorted_unique[i], "extracted key at position {} should match sorted order", i);
        }
    }

    /// Tests interleaved next/next_back for Range iterator matches BTreeMap behavior.
    /// This specifically tests crossing detection across leaf boundaries.
    #[test]
    fn range_interleaved_next_next_back(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Collect using alternating next/next_back
        let mut os_from_front = Vec::new();
        let mut os_from_back = Vec::new();
        let mut bt_from_front = Vec::new();
        let mut bt_from_back = Vec::new();

        let mut os_iter = os_map.range(lo..=hi);
        let mut bt_iter = bt_map.range(lo..=hi);

        let mut toggle = true;
        loop {
            if toggle {
                match (os_iter.next(), bt_iter.next()) {
                    (Some(os_item), Some(bt_item)) => {
                        prop_assert_eq!(os_item, bt_item, "interleaved range next() mismatch");
                        os_from_front.push(*os_item.0);
                        bt_from_front.push(*bt_item.0);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "next() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            } else {
                match (os_iter.next_back(), bt_iter.next_back()) {
                    (Some(os_item), Some(bt_item)) => {
                        prop_assert_eq!(os_item, bt_item, "interleaved range next_back() mismatch");
                        os_from_back.push(*os_item.0);
                        bt_from_back.push(*bt_item.0);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "next_back() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            }
            toggle = !toggle;
        }

        // Verify total elements match
        let os_total = os_from_front.len() + os_from_back.len();
        let bt_total = bt_from_front.len() + bt_from_back.len();
        prop_assert_eq!(os_total, bt_total, "interleaved range total count mismatch");

        // Verify no duplicates by checking combined keys
        let mut os_all: Vec<_> = os_from_front.iter().chain(os_from_back.iter()).copied().collect();
        os_all.sort();
        let os_dedup_len = os_all.len();
        os_all.dedup();
        prop_assert_eq!(os_all.len(), os_dedup_len, "range iterator yielded duplicate keys");
    }

    /// Tests Range iterator is properly fused (once None, always None).
    #[test]
    fn range_fused_iterator(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        let mut iter = os_map.range(lo..=hi);

        // Exhaust the iterator
        while iter.next().is_some() {}

        // Verify FusedIterator: once None, always None
        for _ in 0..10 {
            prop_assert_eq!(iter.next(), None, "FusedIterator violation: next() returned Some after None");
            prop_assert_eq!(iter.next_back(), None, "FusedIterator violation: next_back() returned Some after None");
        }
    }

    /// Tests interleaved next/next_back for RangeMut iterator matches BTreeMap behavior.
    #[test]
    fn range_mut_interleaved_next_next_back(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let mut bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Collect keys using alternating next/next_back
        let mut os_keys = Vec::new();
        let mut bt_keys = Vec::new();

        {
            let mut os_iter = os_map.range_mut(lo..=hi);
            let mut bt_iter = bt_map.range_mut(lo..=hi);

            let mut toggle = true;
            loop {
                if toggle {
                    match (os_iter.next(), bt_iter.next()) {
                        (Some((os_k, os_v)), Some((bt_k, bt_v))) => {
                            prop_assert_eq!(*os_k, *bt_k, "interleaved range_mut next() key mismatch");
                            prop_assert_eq!(*os_v, *bt_v, "interleaved range_mut next() value mismatch");
                            os_keys.push(*os_k);
                            bt_keys.push(*bt_k);
                            // Mutate to verify mutation works
                            *os_v = os_v.wrapping_add(1);
                            *bt_v = bt_v.wrapping_add(1);
                        }
                        (None, None) => break,
                        (os, bt) => {
                            prop_assert!(false, "range_mut next() mismatch: os={:?}, bt={:?}", os.map(|(k, _)| k), bt.map(|(k, _)| k));
                        }
                    }
                } else {
                    match (os_iter.next_back(), bt_iter.next_back()) {
                        (Some((os_k, os_v)), Some((bt_k, bt_v))) => {
                            prop_assert_eq!(*os_k, *bt_k, "interleaved range_mut next_back() key mismatch");
                            prop_assert_eq!(*os_v, *bt_v, "interleaved range_mut next_back() value mismatch");
                            os_keys.push(*os_k);
                            bt_keys.push(*bt_k);
                            *os_v = os_v.wrapping_add(1);
                            *bt_v = bt_v.wrapping_add(1);
                        }
                        (None, None) => break,
                        (os, bt) => {
                            prop_assert!(false, "range_mut next_back() mismatch: os={:?}, bt={:?}", os.map(|(k, _)| k), bt.map(|(k, _)| k));
                        }
                    }
                }
                toggle = !toggle;
            }
        }

        // Verify total elements match
        prop_assert_eq!(os_keys.len(), bt_keys.len(), "interleaved range_mut total count mismatch");

        // Verify mutations were applied correctly
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&os_items, &bt_items, "range_mut mutations mismatch");
    }

    /// Tests RangeMut iterator is properly fused.
    #[test]
    fn range_mut_fused_iterator(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        let mut iter = os_map.range_mut(lo..=hi);

        // Exhaust via alternating pattern
        let mut toggle = true;
        loop {
            if toggle {
                if iter.next().is_none() { break; }
            } else if iter.next_back().is_none() {
                break;
            }
            toggle = !toggle;
        }

        // Verify FusedIterator: once None, always None
        for _ in 0..10 {
            prop_assert_eq!(iter.next(), None, "RangeMut FusedIterator violation: next() returned Some after None");
            prop_assert_eq!(iter.next_back(), None, "RangeMut FusedIterator violation: next_back() returned Some after None");
        }
    }

    /// Tests Range iterator with heavy back-to-front consumption pattern.
    /// This tests the scenario where back retreats multiple leaves before front advances.
    #[test]
    fn range_heavy_next_back_pattern(
        entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE),
        lo in key_strategy(),
        hi in key_strategy(),
    ) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        let mut os_iter = os_map.range(lo..=hi);
        let mut bt_iter = bt_map.range(lo..=hi);

        // Consume mostly from back (3 from back, 1 from front pattern)
        let mut os_items = Vec::new();
        let mut bt_items = Vec::new();
        let mut count = 0;

        loop {
            let (os_item, bt_item) = if count % 4 == 0 {
                (os_iter.next(), bt_iter.next())
            } else {
                (os_iter.next_back(), bt_iter.next_back())
            };

            match (os_item, bt_item) {
                (Some(os), Some(bt)) => {
                    prop_assert_eq!(os, bt, "heavy next_back pattern mismatch at count {}", count);
                    os_items.push(*os.0);
                    bt_items.push(*bt.0);
                }
                (None, None) => break,
                (os, bt) => {
                    prop_assert!(false, "heavy next_back pattern termination mismatch: os={:?}, bt={:?}", os, bt);
                }
            }
            count += 1;
        }

        prop_assert_eq!(os_items.len(), bt_items.len(), "heavy next_back total count mismatch");
    }
}

// ─── Invalid range bounds panic tests ─────────────────────────────────────────

/// Tests that range with start > end panics just like BTreeMap.
#[test]
#[should_panic]
fn range_start_greater_than_end_panics() {
    let map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // This should panic because 5 > 3
    // Use tuple bounds to avoid clippy::reversed_empty_ranges lint
    let _: Vec<_> = map.range((Bound::Included(5), Bound::Included(3))).collect();
}

/// Tests that range_mut with start > end panics just like BTreeMap.
#[test]
#[should_panic]
fn range_mut_start_greater_than_end_panics() {
    let mut map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // This should panic because 5 > 3
    // Use tuple bounds to avoid clippy::reversed_empty_ranges lint
    let _: Vec<_> = map.range_mut((Bound::Included(5), Bound::Included(3))).collect();
}

/// Tests that range with (Excluded(x), Excluded(x)) for same x panics.
#[test]
#[should_panic]
fn range_excluded_excluded_same_bound_panics() {
    let map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // (Excluded(2), Excluded(2)) is an invalid range
    let _: Vec<_> = map.range((Bound::Excluded(2), Bound::Excluded(2))).collect();
}

/// Tests that range with (Excluded(x), Included(y)) where x > y panics.
#[test]
#[should_panic]
fn range_excluded_included_inverted_panics() {
    let map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // (Excluded(5), Included(3)) is an invalid range because 5 > 3
    let _: Vec<_> = map.range((Bound::Excluded(5), Bound::Included(3))).collect();
}

// ─── Out-of-bounds Rank indexing panic tests ──────────────────────────────────

/// Tests that Index<Rank> panics for out-of-bounds rank on non-empty map.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_rank_out_of_bounds_panics() {
    let map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // Map has 3 elements, so Rank(3) is out of bounds
    let _ = map[Rank(3)];
}

/// Tests that IndexMut<Rank> panics for out-of-bounds rank.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_mut_rank_out_of_bounds_panics() {
    let mut map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // Map has 3 elements, so Rank(3) is out of bounds
    map[Rank(3)] = 999;
}

/// Tests that Index<Rank> panics on empty map.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_rank_empty_map_panics() {
    let map: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    let _ = map[Rank(0)];
}

/// Tests that Index<Rank> panics for very large out-of-bounds rank.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_rank_large_out_of_bounds_panics() {
    let map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2)].into_iter().collect();
    let _ = map[Rank(1000)];
}

// ─── Index<&Q> panic tests ────────────────────────────────────────────────────

/// Tests that Index<&Q> panics for missing key on non-empty map.
#[test]
#[should_panic(expected = "no entry found for key")]
fn index_missing_key_panics() {
    let map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    // Key 999 does not exist
    let _ = map[&999];
}

/// Tests that Index<&Q> panics on empty map.
#[test]
#[should_panic(expected = "no entry found for key")]
fn index_key_empty_map_panics() {
    let map: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    let _ = map[&1];
}

/// Tests that Index<&Q> panics for key that was removed.
#[test]
#[should_panic(expected = "no entry found for key")]
fn index_removed_key_panics() {
    let mut map: OSBTreeMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    map.remove(&2);
    let _ = map[&2];
}

// ─── Consuming iterator interleaved tests ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests into_iter with interleaved next/next_back matches BTreeMap.
    #[test]
    fn into_iter_interleaved_next_next_back(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let mut os_iter = os_map.into_iter();
        let mut bt_iter = bt_map.into_iter();

        let mut os_items = Vec::new();
        let mut bt_items = Vec::new();

        let mut toggle = true;
        loop {
            if toggle {
                match (os_iter.next(), bt_iter.next()) {
                    (Some(os_item), Some(bt_item)) => {
                        prop_assert_eq!(os_item, bt_item, "into_iter interleaved next() mismatch");
                        os_items.push(os_item.0);
                        bt_items.push(bt_item.0);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "into_iter next() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            } else {
                match (os_iter.next_back(), bt_iter.next_back()) {
                    (Some(os_item), Some(bt_item)) => {
                        prop_assert_eq!(os_item, bt_item, "into_iter interleaved next_back() mismatch");
                        os_items.push(os_item.0);
                        bt_items.push(bt_item.0);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "into_iter next_back() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            }
            toggle = !toggle;
        }

        prop_assert_eq!(os_items.len(), bt_items.len(), "into_iter interleaved total count mismatch");

        // Verify no duplicates
        let mut os_items_sorted = os_items.clone();
        os_items_sorted.sort();
        let dedup_len = os_items_sorted.len();
        os_items_sorted.dedup();
        prop_assert_eq!(os_items_sorted.len(), dedup_len, "into_iter yielded duplicate keys");
    }

    /// Tests into_keys with interleaved next/next_back matches BTreeMap.
    #[test]
    fn into_keys_interleaved_next_next_back(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let mut os_iter = os_map.into_keys();
        let mut bt_iter = bt_map.into_keys();

        let mut os_keys = Vec::new();
        let mut bt_keys = Vec::new();

        let mut toggle = true;
        loop {
            if toggle {
                match (os_iter.next(), bt_iter.next()) {
                    (Some(os_key), Some(bt_key)) => {
                        prop_assert_eq!(os_key, bt_key, "into_keys interleaved next() mismatch");
                        os_keys.push(os_key);
                        bt_keys.push(bt_key);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "into_keys next() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            } else {
                match (os_iter.next_back(), bt_iter.next_back()) {
                    (Some(os_key), Some(bt_key)) => {
                        prop_assert_eq!(os_key, bt_key, "into_keys interleaved next_back() mismatch");
                        os_keys.push(os_key);
                        bt_keys.push(bt_key);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "into_keys next_back() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            }
            toggle = !toggle;
        }

        prop_assert_eq!(os_keys.len(), bt_keys.len(), "into_keys interleaved total count mismatch");
    }

    /// Tests into_values with interleaved next/next_back matches BTreeMap.
    #[test]
    fn into_values_interleaved_next_next_back(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1..TEST_SIZE)) {
        let os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let bt_map: BTreeMap<i64, i64> = entries.iter().cloned().collect();

        let mut os_iter = os_map.into_values();
        let mut bt_iter = bt_map.into_values();

        let mut os_values = Vec::new();
        let mut bt_values = Vec::new();

        let mut toggle = true;
        loop {
            if toggle {
                match (os_iter.next(), bt_iter.next()) {
                    (Some(os_val), Some(bt_val)) => {
                        os_values.push(os_val);
                        bt_values.push(bt_val);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "into_values next() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            } else {
                match (os_iter.next_back(), bt_iter.next_back()) {
                    (Some(os_val), Some(bt_val)) => {
                        os_values.push(os_val);
                        bt_values.push(bt_val);
                    }
                    (None, None) => break,
                    (os, bt) => {
                        prop_assert!(false, "into_values next_back() mismatch: os={:?}, bt={:?}", os, bt);
                    }
                }
            }
            toggle = !toggle;
        }

        prop_assert_eq!(os_values.len(), bt_values.len(), "into_values interleaved total count mismatch");
    }
}

// ─── extract_if mutation semantics tests ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests extract_if where predicate mutates values and returns both true/false.
    /// Verifies that:
    /// 1. Mutations are applied to both extracted and retained values
    /// 2. The correct elements are extracted based on the predicate result
    #[test]
    fn extract_if_mutates_values(entries in proptest::collection::vec((key_strategy(), value_strategy()), 1000)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let original_map: std::collections::HashMap<_, _> = entries.iter().cloned().collect();

        // Predicate that mutates the value (doubles it) and returns true for even keys
        let os_extracted: Vec<_> = os_map.extract_if(.., |k, v| {
            *v = v.wrapping_mul(2); // Mutate value
            k % 2 == 0 // Extract even keys
        }).collect();

        // Verify extracted elements have even keys and mutated values
        for (k, v) in &os_extracted {
            prop_assert!(k % 2 == 0, "extracted key {} should be even", k);
            if let Some(&original) = original_map.get(k) {
                prop_assert_eq!(*v, original.wrapping_mul(2), "extracted value for key {} should be doubled", k);
            }
        }

        // Verify that retained values (odd keys) were also mutated
        for (&k, &v) in os_map.iter() {
            prop_assert!(k % 2 != 0, "remaining key {} should be odd", k);
            if let Some(&original) = original_map.get(&k) {
                prop_assert_eq!(v, original.wrapping_mul(2), "retained key {} value should be doubled", k);
            }
        }
    }

    /// Tests extract_if where predicate conditionally mutates based on key.
    #[test]
    fn extract_if_conditional_mutation(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();

        // Predicate that:
        // - For keys divisible by 3: mutate to 999 and extract (return true)
        // - For keys divisible by 5 (but not 3): mutate to 555 and retain (return false)
        // - Others: don't mutate, retain (return false)
        let os_extracted: Vec<_> = os_map.extract_if(.., |k, v| {
            if k % 3 == 0 {
                *v = 999;
                true
            } else if k % 5 == 0 {
                *v = 555;
                false
            } else {
                false
            }
        }).collect();

        // Verify extracted elements have keys divisible by 3 and value 999
        for (k, v) in &os_extracted {
            prop_assert!(k % 3 == 0, "extracted key {} should be divisible by 3", k);
            prop_assert_eq!(*v, 999, "extracted value for key {} should be 999", k);
        }

        // Verify specific mutation patterns in retained values
        for (&k, &v) in os_map.iter() {
            prop_assert!(k % 3 != 0, "remaining key {} should not be divisible by 3", k);
            if k % 5 == 0 {
                // Keys divisible by 5 but not 3 should be mutated to 555
                prop_assert_eq!(v, 555, "key {} divisible by 5 (not 3) should be 555", k);
            }
        }
    }

    /// Tests extract_if where predicate always returns false but mutates all values.
    #[test]
    fn extract_if_mutate_all_extract_none(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let original_map: std::collections::HashMap<_, _> = entries.iter().cloned().collect();

        // Predicate that mutates all values but never extracts
        let os_extracted: Vec<_> = os_map.extract_if(.., |_k, v| {
            *v = v.wrapping_add(100);
            false // Never extract
        }).collect();

        // Should extract nothing
        prop_assert!(os_extracted.is_empty(), "should extract nothing");

        // But all values should be mutated
        prop_assert_eq!(os_map.len(), original_map.len(), "map length should be unchanged");

        // Verify mutations were applied
        for (&k, &v) in os_map.iter() {
            if let Some(&original) = original_map.get(&k) {
                prop_assert_eq!(v, original.wrapping_add(100), "value for key {} should be incremented by 100", k);
            }
        }
    }

    /// Tests extract_if where predicate always returns true but mutates all values.
    #[test]
    fn extract_if_mutate_all_extract_all(entries in proptest::collection::vec((key_strategy(), value_strategy()), TEST_SIZE)) {
        let mut os_map: OSBTreeMap<i64, i64> = entries.iter().cloned().collect();
        let original_map: std::collections::HashMap<_, _> = entries.iter().cloned().collect();

        // Predicate that mutates and extracts all values
        let os_extracted: Vec<_> = os_map.extract_if(.., |_k, v| {
            *v = v.wrapping_mul(3);
            true // Always extract
        }).collect();

        // Should extract everything
        prop_assert_eq!(os_extracted.len(), original_map.len(), "should extract all elements");

        // Maps should be empty
        prop_assert!(os_map.is_empty(), "map should be empty after extracting all");

        // Verify extracted values were mutated
        for (k, v) in &os_extracted {
            if let Some(&original) = original_map.get(k) {
                prop_assert_eq!(*v, original.wrapping_mul(3), "extracted value for key {} should be tripled", k);
            }
        }
    }
}

// ─── Thread Safety Tests ──────────────────────────────────────────────────────

/// Compile-time assertions for Send/Sync bounds on iterators.
/// These tests verify that iterators have the same thread-safety guarantees as std.
mod send_sync_tests {
    use wabi_tree::OSBTreeMap;
    use wabi_tree::osbtree_map::{
        IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys, Range, RangeMut, Values, ValuesMut,
    };

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn iter_is_send_sync() {
        assert_send::<Iter<'_, i64, i64>>();
        assert_sync::<Iter<'_, i64, i64>>();
    }

    #[test]
    fn iter_mut_is_send() {
        assert_send::<IterMut<'_, i64, i64>>();
        // Note: IterMut should NOT be Sync - mutable iterators should not be shared
    }

    #[test]
    fn into_iter_is_send_sync() {
        assert_send::<IntoIter<i64, i64>>();
        assert_sync::<IntoIter<i64, i64>>();
    }

    #[test]
    fn keys_is_send_sync() {
        assert_send::<Keys<'_, i64, i64>>();
        assert_sync::<Keys<'_, i64, i64>>();
    }

    #[test]
    fn values_is_send_sync() {
        assert_send::<Values<'_, i64, i64>>();
        assert_sync::<Values<'_, i64, i64>>();
    }

    #[test]
    fn values_mut_is_send() {
        assert_send::<ValuesMut<'_, i64, i64>>();
        // Note: ValuesMut should NOT be Sync
    }

    #[test]
    fn into_keys_is_send_sync() {
        assert_send::<IntoKeys<i64, i64>>();
        assert_sync::<IntoKeys<i64, i64>>();
    }

    #[test]
    fn into_values_is_send_sync() {
        assert_send::<IntoValues<i64, i64>>();
        assert_sync::<IntoValues<i64, i64>>();
    }

    #[test]
    fn range_is_send_sync() {
        assert_send::<Range<'_, i64, i64>>();
        assert_sync::<Range<'_, i64, i64>>();
    }

    #[test]
    fn range_mut_is_send() {
        assert_send::<RangeMut<'_, i64, i64>>();
        // Note: RangeMut should NOT be Sync
    }

    #[test]
    fn map_is_send_sync() {
        assert_send::<OSBTreeMap<i64, i64>>();
        assert_sync::<OSBTreeMap<i64, i64>>();
    }
}

// ─── Drop Semantics Tests ─────────────────────────────────────────────────────

mod drop_tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use wabi_tree::OSBTreeMap;

    struct Droppable {
        drop_count: Rc<Cell<i32>>,
    }

    impl Droppable {
        fn new(_id: i64, drop_count: Rc<Cell<i32>>) -> Self {
            Self {
                drop_count,
            }
        }
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            self.drop_count.set(self.drop_count.get() + 1);
        }
    }

    #[test]
    fn values_dropped_on_remove() {
        let drop_count = Rc::new(Cell::new(0));
        let mut map: OSBTreeMap<i64, Droppable> = OSBTreeMap::new();

        for i in 0..100 {
            map.insert(i, Droppable::new(i, drop_count.clone()));
        }
        assert_eq!(drop_count.get(), 0, "no drops before removal");

        map.remove(&50);
        assert_eq!(drop_count.get(), 1, "one value dropped after remove");

        map.remove(&25);
        assert_eq!(drop_count.get(), 2, "two values dropped after two removes");
    }

    #[test]
    fn values_dropped_on_map_drop() {
        let drop_count = Rc::new(Cell::new(0));
        {
            let mut map: OSBTreeMap<i64, Droppable> = OSBTreeMap::new();
            for i in 0..100 {
                map.insert(i, Droppable::new(i, drop_count.clone()));
            }
            assert_eq!(drop_count.get(), 0, "no drops before map drop");
        }
        assert_eq!(drop_count.get(), 100, "all values dropped when map dropped");
    }

    #[test]
    fn values_dropped_on_clear() {
        let drop_count = Rc::new(Cell::new(0));
        let mut map: OSBTreeMap<i64, Droppable> = OSBTreeMap::new();

        for i in 0..100 {
            map.insert(i, Droppable::new(i, drop_count.clone()));
        }
        assert_eq!(drop_count.get(), 0, "no drops before clear");

        map.clear();
        assert_eq!(drop_count.get(), 100, "all values dropped after clear");
        assert!(map.is_empty());
    }

    #[test]
    fn old_value_dropped_on_replace() {
        let drop_count = Rc::new(Cell::new(0));
        let mut map: OSBTreeMap<i64, Droppable> = OSBTreeMap::new();

        map.insert(1, Droppable::new(1, drop_count.clone()));
        assert_eq!(drop_count.get(), 0);

        // Replace with new value - old value should be dropped
        let old = map.insert(1, Droppable::new(1, drop_count.clone()));
        assert!(old.is_some());
        // The old value is returned and then dropped when `old` goes out of scope
        drop(old);
        assert_eq!(drop_count.get(), 1, "old value dropped after replace");
    }

    #[test]
    fn values_dropped_on_pop_first_last() {
        let drop_count = Rc::new(Cell::new(0));
        let mut map: OSBTreeMap<i64, Droppable> = OSBTreeMap::new();

        for i in 0..10 {
            map.insert(i, Droppable::new(i, drop_count.clone()));
        }
        assert_eq!(drop_count.get(), 0);

        let first = map.pop_first();
        assert!(first.is_some());
        drop(first);
        assert_eq!(drop_count.get(), 1, "value dropped after pop_first");

        let last = map.pop_last();
        assert!(last.is_some());
        drop(last);
        assert_eq!(drop_count.get(), 2, "value dropped after pop_last");
    }
}

// ─── Zero-Sized Type (ZST) Tests ──────────────────────────────────────────────

mod zst_tests {
    use std::collections::BTreeMap;
    use wabi_tree::OSBTreeMap;

    #[test]
    fn map_with_zst_value() {
        let mut os_map: OSBTreeMap<i64, ()> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, ()> = BTreeMap::new();

        for i in 0..1000 {
            os_map.insert(i, ());
            bt_map.insert(i, ());
        }

        assert_eq!(os_map.len(), 1000);
        assert_eq!(os_map.len(), bt_map.len());

        let os_keys: Vec<_> = os_map.keys().copied().collect();
        let bt_keys: Vec<_> = bt_map.keys().copied().collect();
        assert_eq!(os_keys, bt_keys);

        // Test get
        assert_eq!(os_map.get(&500), Some(&()));
        assert_eq!(os_map.get(&2000), None);

        // Test remove
        assert_eq!(os_map.remove(&500), Some(()));
        assert_eq!(os_map.len(), 999);
    }

    #[test]
    fn map_with_large_key() {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
        struct LargeKey([u8; 256]);

        let mut os_map: OSBTreeMap<LargeKey, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<LargeKey, i64> = BTreeMap::new();

        for i in 0..100 {
            let mut key = [0u8; 256];
            key[0] = i as u8;
            os_map.insert(LargeKey(key), i as i64);
            bt_map.insert(LargeKey(key), i as i64);
        }

        assert_eq!(os_map.len(), bt_map.len());

        let os_items: Vec<_> = os_map.iter().map(|(k, &v)| (k.0[0], v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(k, &v)| (k.0[0], v)).collect();
        assert_eq!(os_items, bt_items);
    }

    #[test]
    fn map_with_zst_key_and_value() {
        // Edge case: both key and value are ZSTs
        // Note: This is a degenerate case but should still work
        let mut os_map: OSBTreeMap<(), ()> = OSBTreeMap::new();

        os_map.insert((), ());
        assert_eq!(os_map.len(), 1);
        assert_eq!(os_map.get(&()), Some(&()));

        os_map.insert((), ()); // Replace
        assert_eq!(os_map.len(), 1);

        os_map.remove(&());
        assert_eq!(os_map.len(), 0);
    }
}

// ─── Key Identity Tests ───────────────────────────────────────────────────────

mod key_identity_tests {
    use std::cmp::Ordering;
    use std::collections::BTreeMap;
    use wabi_tree::OSBTreeMap;

    /// A key type where Ord is based on a subset of fields.
    /// This tests that entry().key() returns the stored key, not the probe key.
    #[derive(Clone, Debug)]
    struct KeyWithPayload {
        id: i64,
        payload: String,
    }

    impl PartialEq for KeyWithPayload {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for KeyWithPayload {}

    impl PartialOrd for KeyWithPayload {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for KeyWithPayload {
        fn cmp(&self, other: &Self) -> Ordering {
            self.id.cmp(&other.id)
        }
    }

    #[test]
    fn get_key_value_returns_stored_key() {
        let mut os_map: OSBTreeMap<KeyWithPayload, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<KeyWithPayload, i64> = BTreeMap::new();

        // Insert with payload "first"
        let stored_key = KeyWithPayload {
            id: 1,
            payload: "stored".to_string(),
        };
        os_map.insert(stored_key.clone(), 100);
        bt_map.insert(stored_key.clone(), 100);

        // Lookup with different payload - should find the entry
        let probe_key = KeyWithPayload {
            id: 1,
            payload: "probe".to_string(),
        };

        // get_key_value should return the STORED key, not the probe
        let (os_k, os_v) = os_map.get_key_value(&probe_key).unwrap();
        let (bt_k, bt_v) = bt_map.get_key_value(&probe_key).unwrap();

        assert_eq!(os_k.payload, "stored", "OSBTreeMap should return stored key");
        assert_eq!(bt_k.payload, "stored", "BTreeMap should return stored key");
        assert_eq!(os_v, bt_v);
    }

    #[test]
    fn entry_occupied_key_returns_stored_key() {
        use wabi_tree::osbtree_map::Entry;

        let mut os_map: OSBTreeMap<KeyWithPayload, i64> = OSBTreeMap::new();

        // Insert with payload "stored"
        let stored_key = KeyWithPayload {
            id: 1,
            payload: "stored".to_string(),
        };
        os_map.insert(stored_key, 100);

        // Create entry with different payload
        let probe_key = KeyWithPayload {
            id: 1,
            payload: "probe".to_string(),
        };
        if let Entry::Occupied(o) = os_map.entry(probe_key) {
            // key() should return the STORED key, not the probe key
            // Note: This test documents expected behavior matching std::collections::BTreeMap
            assert_eq!(o.key().payload, "stored", "OccupiedEntry::key() should return the stored key");
        } else {
            panic!("Expected Occupied entry");
        }
    }
}

// ─── Deterministic Insertion Pattern Tests ────────────────────────────────────

/// Helper function to generate deterministic pseudo-random keys using LCG.
fn random_keys_deterministic(n: usize) -> Vec<i64> {
    let mut keys = Vec::with_capacity(n);
    let mut x: u64 = 12345; // Fixed seed for reproducibility
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        keys.push((x >> 33) as i64);
    }
    keys
}

mod insertion_pattern_tests {
    use super::*;
    use std::collections::BTreeMap;
    use wabi_tree::OSBTreeMap;

    const N: usize = 10_000;

    /// Tests ordered (ascending) inserts match BTreeMap.
    #[test]
    fn ordered_inserts_match_btreemap() {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        // Insert in ascending order
        for i in 0..N as i64 {
            os_map.insert(i, i);
            bt_map.insert(i, i);
        }

        // Verify length
        assert_eq!(os_map.len(), N);
        assert_eq!(os_map.len(), bt_map.len());

        // Verify all entries match
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(os_items, bt_items, "ordered inserts content mismatch");

        // Verify first/last
        assert_eq!(os_map.first_key_value(), bt_map.first_key_value());
        assert_eq!(os_map.last_key_value(), bt_map.last_key_value());
    }

    /// Tests reverse-ordered (descending) inserts match BTreeMap.
    #[test]
    fn reverse_ordered_inserts_match_btreemap() {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        // Insert in descending order
        for i in (0..N as i64).rev() {
            os_map.insert(i, i);
            bt_map.insert(i, i);
        }

        // Verify length
        assert_eq!(os_map.len(), N);
        assert_eq!(os_map.len(), bt_map.len());

        // Verify all entries match
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(os_items, bt_items, "reverse ordered inserts content mismatch");

        // Verify first/last
        assert_eq!(os_map.first_key_value(), bt_map.first_key_value());
        assert_eq!(os_map.last_key_value(), bt_map.last_key_value());
    }

    /// Tests random inserts match BTreeMap.
    #[test]
    fn random_inserts_match_btreemap() {
        let keys = random_keys_deterministic(N);
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        // Insert in random order
        for &k in &keys {
            os_map.insert(k, k);
            bt_map.insert(k, k);
        }

        // Verify length matches (accounting for duplicates in random keys)
        assert_eq!(os_map.len(), bt_map.len());

        // Verify all entries match
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(os_items, bt_items, "random inserts content mismatch");

        // Verify first/last
        assert_eq!(os_map.first_key_value(), bt_map.first_key_value());
        assert_eq!(os_map.last_key_value(), bt_map.last_key_value());
    }

    /// Tests ordered get operations match BTreeMap.
    #[test]
    fn ordered_gets_match_btreemap() {
        let os_map: OSBTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();
        let bt_map: BTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();

        // Get in ascending order
        for i in 0..N as i64 {
            assert_eq!(os_map.get(&i), bt_map.get(&i), "ordered get({}) mismatch", i);
        }

        // Get some non-existent keys
        for i in [N as i64, N as i64 + 1, -1, -100] {
            assert_eq!(os_map.get(&i), bt_map.get(&i), "ordered get({}) for missing key mismatch", i);
        }
    }

    /// Tests reverse-ordered get operations match BTreeMap.
    #[test]
    fn reverse_ordered_gets_match_btreemap() {
        let os_map: OSBTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();
        let bt_map: BTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();

        // Get in descending order
        for i in (0..N as i64).rev() {
            assert_eq!(os_map.get(&i), bt_map.get(&i), "reverse get({}) mismatch", i);
        }
    }

    /// Tests random get operations match BTreeMap.
    #[test]
    fn random_gets_match_btreemap() {
        let keys = random_keys_deterministic(N);
        let os_map: OSBTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();
        let bt_map: BTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();

        // Get in random order (same as insertion order)
        for &k in &keys {
            assert_eq!(os_map.get(&k), bt_map.get(&k), "random get({}) mismatch", k);
        }
    }

    /// Tests ordered remove operations match BTreeMap.
    #[test]
    fn ordered_removes_match_btreemap() {
        let mut os_map: OSBTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();
        let mut bt_map: BTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();

        // Remove in ascending order
        for i in 0..N as i64 {
            let os_result = os_map.remove(&i);
            let bt_result = bt_map.remove(&i);
            assert_eq!(os_result, bt_result, "ordered remove({}) mismatch", i);
        }

        assert!(os_map.is_empty());
        assert_eq!(os_map.len(), bt_map.len());
    }

    /// Tests reverse-ordered remove operations match BTreeMap.
    #[test]
    fn reverse_ordered_removes_match_btreemap() {
        let mut os_map: OSBTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();
        let mut bt_map: BTreeMap<i64, i64> = (0..N as i64).map(|i| (i, i)).collect();

        // Remove in descending order
        for i in (0..N as i64).rev() {
            let os_result = os_map.remove(&i);
            let bt_result = bt_map.remove(&i);
            assert_eq!(os_result, bt_result, "reverse remove({}) mismatch", i);
        }

        assert!(os_map.is_empty());
        assert_eq!(os_map.len(), bt_map.len());
    }

    /// Tests random remove operations match BTreeMap.
    #[test]
    fn random_removes_match_btreemap() {
        let keys = random_keys_deterministic(N);
        let mut os_map: OSBTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();
        let mut bt_map: BTreeMap<i64, i64> = keys.iter().map(|&k| (k, k)).collect();

        // Remove in random order (same as insertion order)
        for &k in &keys {
            let os_result = os_map.remove(&k);
            let bt_result = bt_map.remove(&k);
            assert_eq!(os_result, bt_result, "random remove({}) mismatch", k);
        }

        assert!(os_map.is_empty());
        assert_eq!(os_map.len(), bt_map.len());
    }

    /// Tests full CRUD cycle with ordered inserts then removes.
    #[test]
    fn ordered_insert_then_ordered_remove() {
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        // Insert in ascending order
        for i in 0..N as i64 {
            os_map.insert(i, i * 2);
            bt_map.insert(i, i * 2);
        }

        // Verify iteration after inserts
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(os_items, bt_items);

        // Remove in ascending order, checking iteration periodically
        for i in 0..N as i64 {
            os_map.remove(&i);
            bt_map.remove(&i);

            if i % 1000 == 999 {
                let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
                let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(os_items, bt_items, "iteration mismatch after removing {}", i);
            }
        }

        assert!(os_map.is_empty());
    }

    /// Tests full CRUD cycle with random inserts then removes.
    #[test]
    fn random_insert_then_random_remove() {
        let keys = random_keys_deterministic(N);
        let mut os_map: OSBTreeMap<i64, i64> = OSBTreeMap::new();
        let mut bt_map: BTreeMap<i64, i64> = BTreeMap::new();

        // Insert in random order
        for &k in &keys {
            os_map.insert(k, k * 2);
            bt_map.insert(k, k * 2);
        }

        // Verify iteration after inserts
        let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
        let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(os_items, bt_items);

        // Remove in random order, checking iteration periodically
        for (i, &k) in keys.iter().enumerate() {
            os_map.remove(&k);
            bt_map.remove(&k);

            if i % 1000 == 999 {
                let os_items: Vec<_> = os_map.iter().map(|(&k, &v)| (k, v)).collect();
                let bt_items: Vec<_> = bt_map.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(os_items, bt_items, "iteration mismatch after {} removals", i + 1);
            }
        }

        assert!(os_map.is_empty());
    }
}

// ─── Coverage-focused top-down tests ────────────────────────────────────────

#[test]
fn capacity_default_from_array_and_extend_refs() {
    let map: OSBTreeMap<i32, i32> = OSBTreeMap::with_capacity(8);
    assert!(map.is_empty());
    assert_eq!(map.capacity(), 8);

    let default_map: OSBTreeMap<i32, i32> = Default::default();
    assert!(default_map.is_empty());
    let _ = format!("{:?}", default_map);

    let from_arr = OSBTreeMap::from([(2, 20), (1, 10)]);
    let items: Vec<_> = from_arr.iter().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(items, vec![(1, 10), (2, 20)]);

    let data = [(3, 30), (4, 40)];
    let mut extend_map = OSBTreeMap::new();
    extend_map.extend(data.iter().map(|(k, v)| (k, v)));
    assert_eq!(extend_map.get(&3), Some(&30));
    assert_eq!(extend_map.get(&4), Some(&40));
}

#[test]
fn append_fast_paths() {
    let mut target = OSBTreeMap::new();
    target.insert(1, 10);
    let mut empty_source: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    target.append(&mut empty_source);
    assert_eq!(target.len(), 1);
    assert!(empty_source.is_empty());

    let mut empty_target: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    let mut source = OSBTreeMap::from([(2, 20), (3, 30)]);
    empty_target.append(&mut source);
    assert!(source.is_empty());
    let items: Vec<_> = empty_target.iter().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(items, vec![(2, 20), (3, 30)]);
}

#[test]
fn entry_key_remove_and_debug() {
    let mut map = OSBTreeMap::new();

    {
        let entry = map.entry(7);
        assert_eq!(entry.key(), &7);
        let _ = format!("{:?}", entry);
    }

    map.entry(7).or_insert(70);

    {
        let entry = map.entry(7);
        assert_eq!(entry.key(), &7);
        let _ = format!("{:?}", entry);
    }

    let removed = match map.entry(7) {
        osbtree_map::Entry::Occupied(occupied) => occupied.remove(),
        osbtree_map::Entry::Vacant(_) => unreachable!("entry should be occupied"),
    };
    assert_eq!(removed, 70);
    assert!(map.is_empty());
}

#[test]
fn range_edge_cases() {
    let empty: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    assert_eq!(empty.range(..).next(), None);
    assert_eq!(empty.range(1..).next(), None);

    let map = OSBTreeMap::from([(10, 1), (20, 2)]);
    assert_eq!(map.range(..=5).next(), None);
    assert_eq!(map.range(..5).next(), None);
    assert_eq!(map.range(25..).next(), None);
    {
        use core::ops::Bound::{Excluded, Unbounded};
        let mut excluded_start = map.range((Excluded(25), Unbounded));
        assert_eq!(excluded_start.next(), None);
    }

    let sparse = OSBTreeMap::from([(10, 1), (20, 2)]);
    let mut range = sparse.range(15..=15);
    assert_eq!(range.next(), None);

    let mut range_back = sparse.range(15..=15);
    assert_eq!(range_back.next_back(), None);
}

#[test]
fn range_mut_edge_cases() {
    let mut empty: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    assert_eq!(empty.range_mut(..).next(), None);

    let mut map = OSBTreeMap::from([(10, 1), (20, 2)]);
    assert_eq!(map.range_mut(..=5).next(), None);
    assert_eq!(map.range_mut(25..).next(), None);
    {
        use core::ops::Bound::{Excluded, Unbounded};
        assert_eq!(map.range_mut((Excluded(25), Unbounded)).next(), None);
        assert_eq!(map.range_mut((Unbounded, Excluded(5))).next(), None);
    }

    let mut map_excluded = OSBTreeMap::from([(10, 1), (20, 2), (30, 3)]);
    {
        use core::ops::Bound::{Excluded, Unbounded};
        let mut range_mut = map_excluded.range_mut((Excluded(10), Unbounded));
        let first = range_mut.next().map(|(k, v)| (*k, *v));
        assert_eq!(first, Some((20, 2)));
    }
    {
        use core::ops::Bound::{Excluded, Unbounded};
        let mut range_mut = map_excluded.range_mut((Unbounded, Excluded(30)));
        let last = range_mut.next_back().map(|(k, v)| (*k, *v));
        assert_eq!(last, Some((20, 2)));
    }

    let mut sparse = OSBTreeMap::from([(10, 1), (20, 2)]);
    {
        let mut range_mut = sparse.range_mut(15..=15);
        assert_eq!(range_mut.size_hint(), (0, Some(0)));
        assert_eq!(range_mut.next(), None);
    }
    {
        let mut range_mut = sparse.range_mut(15..=15);
        assert_eq!(range_mut.next_back(), None);
    }
}

#[test]
#[allow(clippy::double_ended_iterator_last)]
fn iterator_trait_impls() {
    let mut map = OSBTreeMap::from([(1, 10), (2, 20), (3, 30)]);

    for (_, value) in &mut map {
        *value += 1;
    }
    assert_eq!(map.get(&1), Some(&11));
    assert_eq!(map.get(&3), Some(&31));

    {
        let iter = map.iter();
        assert_eq!(iter.len(), 3);
        let iter_clone = iter.clone();
        let _ = format!("{:?}", iter_clone);

        let keys = map.keys();
        assert_eq!(keys.len(), 3);
        let _ = format!("{:?}", keys.clone());

        let values = map.values();
        assert_eq!(values.len(), 3);
        assert_eq!(map.values().last(), Some(&31));
        let _ = format!("{:?}", values.clone());

        let mut values_mut = map.values_mut();
        assert_eq!(values_mut.size_hint(), (3, Some(3)));
        let back_value = values_mut.next_back().map(|v| *v);
        assert_eq!(back_value, Some(31));
        let last_value = map.values_mut().last().map(|v| *v);
        assert_eq!(last_value, Some(31));

        let range = map.range(1..=2);
        assert_eq!(range.len(), 2);
        let _ = format!("{:?}", range.clone());
    }

    {
        let iter_mut = map.iter_mut();
        assert_eq!(iter_mut.len(), 3);
        let _ = format!("{:?}", iter_mut);
    }

    {
        let range_mut = map.range_mut(1..=2);
        assert_eq!(range_mut.len(), 2);
        let _ = format!("{:?}", range_mut);
    }

    let into_iter = map.clone().into_iter();
    let _ = format!("{:?}", into_iter);
    let into_keys = map.clone().into_keys();
    assert_eq!(into_keys.len(), 3);
    let _ = format!("{:?}", into_keys);
    let into_values = map.clone().into_values();
    assert_eq!(into_values.len(), 3);
    let _ = format!("{:?}", into_values);

    let empty_iter: osbtree_map::Iter<'_, i32, i32> = Default::default();
    assert_eq!(empty_iter.len(), 0);
    let _ = format!("{:?}", empty_iter.clone());

    let empty_iter_mut: osbtree_map::IterMut<'_, i32, i32> = Default::default();
    assert_eq!(empty_iter_mut.len(), 0);
    let _ = format!("{:?}", empty_iter_mut);

    let empty_into_iter: osbtree_map::IntoIter<i32, i32> = Default::default();
    let _ = format!("{:?}", empty_into_iter);

    let empty_keys: osbtree_map::Keys<'_, i32, i32> = Default::default();
    assert_eq!(empty_keys.len(), 0);
    let _ = format!("{:?}", empty_keys);

    let empty_values: osbtree_map::Values<'_, i32, i32> = Default::default();
    assert_eq!(empty_values.len(), 0);
    let _ = format!("{:?}", empty_values);

    let empty_values_mut: osbtree_map::ValuesMut<'_, i32, i32> = Default::default();
    assert_eq!(empty_values_mut.len(), 0);
    let _ = format!("{:?}", empty_values_mut);

    let empty_into_keys: osbtree_map::IntoKeys<i32, i32> = Default::default();
    let _ = format!("{:?}", empty_into_keys);

    let empty_into_values: osbtree_map::IntoValues<i32, i32> = Default::default();
    let _ = format!("{:?}", empty_into_values);

    let empty_range: osbtree_map::Range<'_, i32, i32> = Default::default();
    assert_eq!(empty_range.len(), 0);
    let _ = format!("{:?}", empty_range);

    let empty_range_mut: osbtree_map::RangeMut<'_, i32, i32> = Default::default();
    assert_eq!(empty_range_mut.len(), 0);
    let _ = format!("{:?}", empty_range_mut);

    let mut extract_map = map.clone();
    let extractor = extract_map.extract_if(.., |_, _| false);
    let _ = format!("{:?}", extractor);
}

#[test]
fn empty_clone_and_into_iter_variants() {
    let empty: OSBTreeMap<i32, i32> = OSBTreeMap::new();
    let cloned = empty.clone();
    assert!(cloned.is_empty());

    let mut into_iter = OSBTreeMap::<i32, i32>::new().into_iter();
    assert_eq!(into_iter.next(), None);

    let mut into_keys = OSBTreeMap::<i32, i32>::new().into_keys();
    assert_eq!(into_keys.next(), None);

    let mut into_values = OSBTreeMap::<i32, i32>::new().into_values();
    assert_eq!(into_values.next(), None);
}

#[test]
fn boundary_stress_around_leaf_edges() {
    use core::ops::Bound::{Excluded, Unbounded};

    // Use many even keys to guarantee gaps between adjacent keys.
    let mut map: OSBTreeMap<i32, i32> = (0..4000).map(|i| (i * 2, i)).collect();
    assert!(map.len() > 512);

    // Stress start/end bounds around many adjacent key gaps.
    for rank in 0..(map.len() - 1) {
        let k1 = *map.get_by_rank(rank).expect("rank in bounds").0;
        let k2 = *map.get_by_rank(rank + 1).expect("rank+1 in bounds").0;
        if k2 - k1 <= 1 {
            continue;
        }
        let mid = k1 + 1;

        // Lower-bound style: start at a non-existent key between two keys.
        let _ = map.range(mid..).next();

        // Upper-bound style: exclude an existing key.
        let _ = map.range((Excluded(k1), Unbounded)).next();

        // RangeMut variants exercise the same raw bound helpers.
        {
            let _ = map.range_mut(mid..).next();
        }
        {
            let _ = map.range_mut((Excluded(k1), Unbounded)).next();
        }
    }
}

#[test]
fn empty_iterators_and_ranges_are_well_formed() {
    let mut map: OSBTreeMap<i32, i32> = OSBTreeMap::new();

    {
        let iter = map.iter();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }
    {
        let iter_mut = map.iter_mut();
        assert_eq!(iter_mut.size_hint(), (0, Some(0)));
    }

    assert_eq!(map.range(..).next(), None);
    assert_eq!(map.range_mut(..).next(), None);
}

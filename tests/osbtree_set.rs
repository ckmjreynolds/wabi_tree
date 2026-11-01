use std::collections::BTreeSet;

use proptest::prelude::*;
use wabi_tree::osbtree_set;
use wabi_tree::{OSBTreeSet, Rank};

/// The number of operations to perform in each proptest case.
const TEST_SIZE: usize = 10_000;

/// Generates a vector of random values in a range that ensures collisions.
fn value_strategy() -> impl Strategy<Value = i64> {
    -20_000i64..20_000i64
}

// ─── Operations enum for driving randomized tests ────────────────────────────

#[derive(Debug, Clone)]
enum SetOp {
    Insert(i64),
    Remove(i64),
    Contains(i64),
    First,
    Last,
    PopFirst,
    PopLast,
}

fn set_op_strategy() -> impl Strategy<Value = SetOp> {
    prop_oneof![
        5 => value_strategy().prop_map(SetOp::Insert),
        3 => value_strategy().prop_map(SetOp::Remove),
        2 => value_strategy().prop_map(SetOp::Contains),
        1 => Just(SetOp::First),
        1 => Just(SetOp::Last),
        1 => Just(SetOp::PopFirst),
        1 => Just(SetOp::PopLast),
    ]
}

// ─── Core CRUD operations ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Replays a random sequence of insert/remove/contains operations on both
    /// OSBTreeSet and BTreeSet and asserts identical results at every step.
    #[test]
    fn set_ops_match_btreeset(ops in proptest::collection::vec(set_op_strategy(), TEST_SIZE)) {
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        for op in &ops {
            match op {
                SetOp::Insert(v) => {
                    let os_result = os_set.insert(*v);
                    let bt_result = bt_set.insert(*v);
                    prop_assert_eq!(os_result, bt_result, "insert({})", v);
                }
                SetOp::Remove(v) => {
                    let os_result = os_set.remove(v);
                    let bt_result = bt_set.remove(v);
                    prop_assert_eq!(os_result, bt_result, "remove({})", v);
                }
                SetOp::Contains(v) => {
                    let os_result = os_set.contains(v);
                    let bt_result = bt_set.contains(v);
                    prop_assert_eq!(os_result, bt_result, "contains({})", v);
                }
                SetOp::First => {
                    let os_result = os_set.first();
                    let bt_result = bt_set.first();
                    prop_assert_eq!(os_result, bt_result, "first()");
                }
                SetOp::Last => {
                    let os_result = os_set.last();
                    let bt_result = bt_set.last();
                    prop_assert_eq!(os_result, bt_result, "last()");
                }
                SetOp::PopFirst => {
                    let os_result = os_set.pop_first();
                    let bt_result = bt_set.pop_first();
                    prop_assert_eq!(os_result, bt_result, "pop_first()");
                }
                SetOp::PopLast => {
                    let os_result = os_set.pop_last();
                    let bt_result = bt_set.pop_last();
                    prop_assert_eq!(os_result, bt_result, "pop_last()");
                }
            }
            prop_assert_eq!(os_set.len(), bt_set.len(), "len mismatch after {:?}", op);
            prop_assert_eq!(os_set.is_empty(), bt_set.is_empty(), "is_empty mismatch after {:?}", op);
        }
    }

    /// Tests that iteration order matches BTreeSet after random insertions.
    #[test]
    fn iter_matches_btreeset(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        // Forward iteration
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        prop_assert_eq!(&os_items, &bt_items, "iter() mismatch");

        // Reverse iteration
        let os_rev: Vec<_> = os_set.iter().rev().copied().collect();
        let bt_rev: Vec<_> = bt_set.iter().rev().copied().collect();
        prop_assert_eq!(&os_rev, &bt_rev, "iter().rev() mismatch");

        // into_iter
        let os_into: Vec<_> = os_set.clone().into_iter().collect();
        let bt_into: Vec<_> = bt_set.clone().into_iter().collect();
        prop_assert_eq!(&os_into, &bt_into, "into_iter() mismatch");
    }

    /// Tests ExactSizeIterator and DoubleEndedIterator behavior.
    #[test]
    fn iter_size_and_double_ended(values in proptest::collection::vec(value_strategy(), 1..TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();

        let iter = os_set.iter();
        prop_assert_eq!(iter.len(), os_set.len(), "ExactSizeIterator len mismatch");

        // Alternating front/back
        let mut from_front = Vec::new();
        let mut from_back = Vec::new();
        let mut iter = os_set.iter();
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
        prop_assert_eq!(from_front.len() + from_back.len(), os_set.len());
    }

    /// Tests range queries match BTreeSet.
    #[test]
    fn range_matches_btreeset(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Inclusive range
        let os_range: Vec<_> = os_set.range(lo..=hi).copied().collect();
        let bt_range: Vec<_> = bt_set.range(lo..=hi).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..={}) mismatch", lo, hi);

        // Exclusive end
        let os_range: Vec<_> = os_set.range(lo..hi).copied().collect();
        let bt_range: Vec<_> = bt_set.range(lo..hi).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..{}) mismatch", lo, hi);

        // From start
        let os_range: Vec<_> = os_set.range(lo..).copied().collect();
        let bt_range: Vec<_> = bt_set.range(lo..).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..) mismatch", lo);

        // Up to end
        let os_range: Vec<_> = os_set.range(..=hi).copied().collect();
        let bt_range: Vec<_> = bt_set.range(..=hi).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range(..={}) mismatch", hi);

        // Unbounded
        let os_range: Vec<_> = os_set.range::<i64, _>(..).copied().collect();
        let bt_range: Vec<_> = bt_set.range::<i64, _>(..).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range(..) mismatch");

        // Reverse
        let os_rev: Vec<_> = os_set.range(lo..=hi).rev().copied().collect();
        let bt_rev: Vec<_> = bt_set.range(lo..=hi).rev().copied().collect();
        prop_assert_eq!(&os_rev, &bt_rev, "range({}..={}).rev() mismatch", lo, hi);
    }

    /// Tests retain matches BTreeSet.
    #[test]
    fn retain_matches_btreeset(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let mut bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        os_set.retain(|v| v % 3 != 0);
        bt_set.retain(|v| v % 3 != 0);

        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        prop_assert_eq!(&os_items, &bt_items, "retain mismatch");
    }

    /// Tests append matches BTreeSet.
    #[test]
    fn append_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let mut os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let mut os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let mut bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let mut bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        os_a.append(&mut os_b);
        bt_a.append(&mut bt_b);

        prop_assert_eq!(os_b.len(), 0, "append did not empty source");
        prop_assert_eq!(os_a.len(), bt_a.len(), "append len mismatch");

        let os_items: Vec<_> = os_a.iter().copied().collect();
        let bt_items: Vec<_> = bt_a.iter().copied().collect();
        prop_assert_eq!(&os_items, &bt_items, "append content mismatch");
    }

    /// Tests split_off matches BTreeSet.
    #[test]
    fn split_off_matches_btreeset(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        split_val in value_strategy(),
    ) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let mut bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let os_right = os_set.split_off(&split_val);
        let bt_right = bt_set.split_off(&split_val);

        let os_left: Vec<_> = os_set.iter().copied().collect();
        let bt_left: Vec<_> = bt_set.iter().copied().collect();
        prop_assert_eq!(&os_left, &bt_left, "split_off left mismatch");

        let os_right_items: Vec<_> = os_right.iter().copied().collect();
        let bt_right_items: Vec<_> = bt_right.iter().copied().collect();
        prop_assert_eq!(&os_right_items, &bt_right_items, "split_off right mismatch");
    }

    /// Tests clear empties the set.
    #[test]
    fn clear_empties_set(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        os_set.clear();
        prop_assert!(os_set.is_empty());
        prop_assert_eq!(os_set.len(), 0);
        prop_assert_eq!(os_set.iter().count(), 0);
    }

    /// Tests get matches BTreeSet behavior.
    #[test]
    fn get_matches_btreeset(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        probes in proptest::collection::vec(value_strategy(), 1000),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        for p in &probes {
            let os_result = os_set.get(p);
            let bt_result = bt_set.get(p);
            prop_assert_eq!(os_result, bt_result, "get({})", p);
        }
    }

    /// Tests take matches expected behavior.
    #[test]
    fn take_matches_expected(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        to_take in proptest::collection::vec(value_strategy(), TEST_SIZE / 5),
    ) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let mut bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        for v in &to_take {
            let os_result = os_set.take(v);
            let bt_result = bt_set.take(v);
            prop_assert_eq!(os_result, bt_result, "take({})", v);
        }

        prop_assert_eq!(os_set.len(), bt_set.len());
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        prop_assert_eq!(&os_items, &bt_items, "take residual mismatch");
    }

    /// Tests replace behavior.
    #[test]
    fn replace_matches_expected(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();

        for v in &values {
            let was_present = os_set.contains(v);
            let old = os_set.replace(*v);
            if was_present {
                prop_assert_eq!(old, Some(*v), "replace({}) should return old value", v);
            } else {
                prop_assert_eq!(old, None, "replace({}) should return None for new", v);
            }
        }
    }
}

// ─── Set operations ──────────────────────────────────────────────────────────

/// Checks is_subset / is_superset in both directions against BTreeSet.
fn check_subset_relations(bt_a: &BTreeSet<i64>, bt_b: &BTreeSet<i64>) -> Result<(), TestCaseError> {
    let os_a: OSBTreeSet<i64> = bt_a.iter().cloned().collect();
    let os_b: OSBTreeSet<i64> = bt_b.iter().cloned().collect();
    prop_assert_eq!(os_a.is_subset(&os_b), bt_a.is_subset(bt_b), "is_subset mismatch");
    prop_assert_eq!(os_b.is_subset(&os_a), bt_b.is_subset(bt_a), "reverse is_subset mismatch");
    prop_assert_eq!(os_a.is_superset(&os_b), bt_a.is_superset(bt_b), "is_superset mismatch");
    prop_assert_eq!(os_b.is_superset(&os_a), bt_b.is_superset(bt_a), "reverse is_superset mismatch");
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests difference matches BTreeSet.
    #[test]
    fn difference_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        let os_diff: Vec<_> = os_a.difference(&os_b).copied().collect();
        let bt_diff: Vec<_> = bt_a.difference(&bt_b).copied().collect();
        prop_assert_eq!(&os_diff, &bt_diff, "difference mismatch");

        // Also test Sub operator
        let os_sub: Vec<_> = (&os_a - &os_b).iter().copied().collect();
        prop_assert_eq!(&os_sub, &bt_diff, "Sub operator mismatch");
    }

    /// Tests difference size_hint bounds are valid when other is much larger or a superset.
    /// The lower bound must not exceed the actual difference size.
    #[test]
    fn difference_size_hint_bounds_valid(
        values_a in proptest::collection::vec(value_strategy(), 1..500),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        let os_diff_iter = os_a.difference(&os_b);
        let bt_diff_iter = bt_a.difference(&bt_b);

        let (os_lo, os_hi) = os_diff_iter.size_hint();
        let (bt_lo, _bt_hi) = bt_diff_iter.size_hint();

        // Count actual elements
        let os_actual: Vec<_> = os_a.difference(&os_b).copied().collect();
        let bt_actual: Vec<_> = bt_a.difference(&bt_b).copied().collect();

        prop_assert_eq!(os_actual.len(), bt_actual.len(), "difference count mismatch");

        // Lower bound must not exceed actual count
        prop_assert!(
            os_lo <= os_actual.len(),
            "OSBTreeSet difference size_hint lower bound {} exceeds actual count {}",
            os_lo, os_actual.len()
        );

        // Upper bound must be >= actual count (if Some)
        if let Some(hi) = os_hi {
            prop_assert!(
                hi >= os_actual.len(),
                "OSBTreeSet difference size_hint upper bound {} is less than actual count {}",
                hi, os_actual.len()
            );
        }

        // Compare bounds with BTreeSet (should be similar or more conservative)
        prop_assert!(
            os_lo <= bt_lo || os_lo <= os_actual.len(),
            "OSBTreeSet difference size_hint lower bound {} is less conservative than BTreeSet {}",
            os_lo, bt_lo
        );
    }

    /// Tests difference size_hint when other is a superset of self.
    /// In this case, the difference is empty, so lower bound should be 0.
    #[test]
    fn difference_size_hint_superset(
        values in proptest::collection::vec(value_strategy(), 1..500),
    ) {
        // Create a set and its superset
        let os_a: OSBTreeSet<i64> = values.iter().cloned().collect();
        let mut os_b = os_a.clone();

        // Add extra elements to make b a strict superset
        for i in 100_000..100_100 {
            os_b.insert(i);
        }

        let diff_iter = os_a.difference(&os_b);
        let (lo, hi) = diff_iter.size_hint();

        // Actual difference is empty since b is a superset of a
        let actual_count = os_a.difference(&os_b).count();
        prop_assert_eq!(actual_count, 0, "superset difference should be empty");

        // Lower bound must be 0 since actual is 0
        prop_assert!(
            lo <= actual_count,
            "difference size_hint lower bound {} exceeds actual count {} for superset case",
            lo, actual_count
        );

        // Upper bound should be >= 0
        if let Some(h) = hi {
            prop_assert!(h >= actual_count, "upper bound {} < actual {}", h, actual_count);
        }
    }

    /// Tests symmetric_difference matches BTreeSet.
    #[test]
    fn symmetric_difference_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        let os_sym: Vec<_> = os_a.symmetric_difference(&os_b).copied().collect();
        let bt_sym: Vec<_> = bt_a.symmetric_difference(&bt_b).copied().collect();
        prop_assert_eq!(&os_sym, &bt_sym, "symmetric_difference mismatch");

        // Also test BitXor operator
        let os_xor: Vec<_> = (&os_a ^ &os_b).iter().copied().collect();
        prop_assert_eq!(&os_xor, &bt_sym, "BitXor operator mismatch");
    }

    /// Tests intersection matches BTreeSet.
    #[test]
    fn intersection_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        let os_inter: Vec<_> = os_a.intersection(&os_b).copied().collect();
        let bt_inter: Vec<_> = bt_a.intersection(&bt_b).copied().collect();
        prop_assert_eq!(&os_inter, &bt_inter, "intersection mismatch");

        // Also test BitAnd operator
        let os_and: Vec<_> = (&os_a & &os_b).iter().copied().collect();
        prop_assert_eq!(&os_and, &bt_inter, "BitAnd operator mismatch");
    }

    /// Tests union matches BTreeSet.
    #[test]
    fn union_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        let os_union: Vec<_> = os_a.union(&os_b).copied().collect();
        let bt_union: Vec<_> = bt_a.union(&bt_b).copied().collect();
        prop_assert_eq!(&os_union, &bt_union, "union mismatch");

        // Also test BitOr operator
        let os_or: Vec<_> = (&os_a | &os_b).iter().copied().collect();
        prop_assert_eq!(&os_or, &bt_union, "BitOr operator mismatch");
    }

    /// Tests is_disjoint matches BTreeSet.
    #[test]
    fn is_disjoint_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        prop_assert_eq!(os_a.is_disjoint(&os_b), bt_a.is_disjoint(&bt_b), "is_disjoint mismatch");
    }

    /// Tests is_subset / is_superset matches BTreeSet.
    #[test]
    fn subset_superset_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();
        check_subset_relations(&bt_a, &bt_b)?;
    }

    /// Tests is_subset / is_superset when `a` is sampled from `b`, so the cheap early exits
    /// don't apply. Small samples take the search path; large ones take the stitch path.
    /// Optionally adds a value that isn't in `b` (a near-miss inside or outside `b`'s span),
    /// and optionally drops `b`'s min and max from `a` so the spans differ.
    #[test]
    fn subset_superset_sampled_matches_btreeset(
        values_b in proptest::collection::vec(value_strategy(), 1..TEST_SIZE / 2),
        keep_one_in in prop_oneof![Just(1usize), Just(2usize), Just(40usize), Just(1000usize)],
        offset in 0usize..1000,
        extra in proptest::option::of(value_strategy()),
        trim_ends in any::<bool>(),
    ) {
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();
        let mut bt_a: BTreeSet<i64> = bt_b
            .iter()
            .enumerate()
            .filter(|(i, _)| (i + offset) % keep_one_in == 0)
            .map(|(_, v)| *v)
            .collect();
        if trim_ends {
            bt_a.remove(bt_b.first().unwrap());
            bt_a.remove(bt_b.last().unwrap());
        }
        if let Some(x) = extra {
            bt_a.insert(x);
        }
        check_subset_relations(&bt_a, &bt_b)?;
    }

    /// Tests is_subset / is_superset for sets sharing min and max but differing in between,
    /// and for sets whose spans are disjoint.
    #[test]
    fn subset_superset_spans_match_btreeset(
        values_a in proptest::collection::vec(-200i64..200, 0..300),
        values_b in proptest::collection::vec(-200i64..200, 0..300),
        shift in prop_oneof![Just(0i64), Just(1000i64)],
    ) {
        let mut bt_a: BTreeSet<i64> = values_a.iter().map(|v| v + shift).collect();
        let mut bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();
        if shift == 0 {
            for set in [&mut bt_a, &mut bt_b] {
                set.insert(-200);
                set.insert(200);
            }
        }
        check_subset_relations(&bt_a, &bt_b)?;
    }

    /// Tests extract_if matches expected behavior.
    #[test]
    fn extract_if_matches_expected(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let os_extracted: Vec<_> = os_set.extract_if(.., |v| v % 2 == 0).collect();
        let bt_extracted: Vec<_> = bt_set.iter().filter(|v| *v % 2 == 0).copied().collect();

        prop_assert_eq!(&os_extracted, &bt_extracted, "extract_if extracted mismatch");

        let os_remaining: Vec<_> = os_set.iter().copied().collect();
        let bt_remaining: Vec<_> = bt_set.iter().filter(|v| *v % 2 != 0).copied().collect();
        prop_assert_eq!(&os_remaining, &bt_remaining, "extract_if remaining mismatch");
    }
}

// ─── Order-statistic operations (compared against Vec) ───────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests get_by_rank against a sorted Vec oracle.
    #[test]
    fn get_by_rank_matches_vec(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let sorted: Vec<i64> = BTreeSet::from_iter(values.iter().cloned())
            .into_iter()
            .collect();

        prop_assert_eq!(os_set.len(), sorted.len());

        for (rank, expected_val) in sorted.iter().enumerate() {
            let os_result = os_set.get_by_rank(rank);
            let expected = Some(expected_val);
            prop_assert_eq!(
                os_result, expected,
                "get_by_rank({}) mismatch: got {:?}, expected {:?}", rank, os_result, expected
            );
        }

        // Out of bounds
        prop_assert_eq!(os_set.get_by_rank(sorted.len()), None);
        prop_assert_eq!(os_set.get_by_rank(sorted.len() + 100), None);
    }

    /// Tests rank_of against a sorted Vec oracle.
    #[test]
    fn rank_of_matches_vec(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let sorted: Vec<i64> = BTreeSet::from_iter(values.iter().cloned())
            .into_iter()
            .collect();

        for (expected_rank, v) in sorted.iter().enumerate() {
            let rank = os_set.rank_of(v);
            prop_assert_eq!(rank, Some(expected_rank), "rank_of({})", v);
        }

        // Values not in the set should return None
        for probe in [i64::MIN, i64::MAX, 99999, -99999] {
            if !os_set.contains(&probe) {
                prop_assert_eq!(os_set.rank_of(&probe), None, "rank_of({}) should be None", probe);
            }
        }
    }

    /// Tests Index<Rank>.
    #[test]
    fn index_by_rank_matches_vec(values in proptest::collection::vec(value_strategy(), 1..TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let sorted: Vec<i64> = BTreeSet::from_iter(values.iter().cloned())
            .into_iter()
            .collect();

        for (rank, expected_val) in sorted.iter().enumerate() {
            prop_assert_eq!(os_set[Rank(rank)], *expected_val, "Index[Rank({})]", rank);
        }
    }

    /// Tests that rank_of and get_by_rank are consistent.
    #[test]
    fn rank_of_get_by_rank_roundtrip(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();

        for rank in 0..os_set.len() {
            let v = os_set.get_by_rank(rank).unwrap();
            let recovered_rank = os_set.rank_of(v).unwrap();
            prop_assert_eq!(recovered_rank, rank, "roundtrip rank mismatch at rank {}", rank);
        }
    }

    /// Tests order-statistic operations after a mix of inserts and removes.
    #[test]
    fn order_stats_after_mutations(ops in proptest::collection::vec(set_op_strategy(), TEST_SIZE)) {
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        for op in &ops {
            match op {
                SetOp::Insert(v) => {
                    os_set.insert(*v);
                    bt_set.insert(*v);
                }
                SetOp::Remove(v) => {
                    os_set.remove(v);
                    bt_set.remove(v);
                }
                _ => {}
            }
        }

        let sorted: Vec<i64> = bt_set.into_iter().collect();
        prop_assert_eq!(os_set.len(), sorted.len());

        // Spot-check ranks at various positions
        let check_positions = [0, 1, sorted.len() / 4, sorted.len() / 2, sorted.len() * 3 / 4, sorted.len().saturating_sub(1)];
        for &pos in &check_positions {
            if pos < sorted.len() {
                let os_result = os_set.get_by_rank(pos);
                prop_assert_eq!(os_result, Some(&sorted[pos]), "get_by_rank({}) after mutations", pos);

                let rank = os_set.rank_of(&sorted[pos]);
                prop_assert_eq!(rank, Some(pos), "rank_of after mutations at pos {}", pos);
            }
        }
    }
}

// ─── Trait implementations ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests FromIterator and Extend match BTreeSet.
    #[test]
    fn from_iter_matches_btreeset(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        prop_assert_eq!(&os_items, &bt_items, "FromIterator mismatch");
    }

    /// Tests Extend matches BTreeSet.
    #[test]
    fn extend_matches_btreeset(
        initial in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        extra in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let mut os_set: OSBTreeSet<i64> = initial.iter().cloned().collect();
        let mut bt_set: BTreeSet<i64> = initial.iter().cloned().collect();

        os_set.extend(extra.iter().cloned());
        bt_set.extend(extra.iter().cloned());

        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        prop_assert_eq!(&os_items, &bt_items, "extend mismatch");
    }

    /// Tests Clone produces an equal set.
    #[test]
    fn clone_produces_equal_set(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let cloned = os_set.clone();

        prop_assert_eq!(os_set.len(), cloned.len());
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let cl_items: Vec<_> = cloned.iter().copied().collect();
        prop_assert_eq!(&os_items, &cl_items, "clone content mismatch");
    }

    /// Tests PartialEq / Eq.
    #[test]
    fn eq_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        prop_assert_eq!(os_a == os_b, bt_a == bt_b, "equality mismatch");
    }

    /// Tests Ord / PartialOrd.
    #[test]
    fn ord_matches_btreeset(
        values_a in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
        values_b in proptest::collection::vec(value_strategy(), TEST_SIZE / 2),
    ) {
        let os_a: OSBTreeSet<i64> = values_a.iter().cloned().collect();
        let os_b: OSBTreeSet<i64> = values_b.iter().cloned().collect();
        let bt_a: BTreeSet<i64> = values_a.iter().cloned().collect();
        let bt_b: BTreeSet<i64> = values_b.iter().cloned().collect();

        prop_assert_eq!(os_a.cmp(&os_b), bt_a.cmp(&bt_b), "Ord mismatch");
        prop_assert_eq!(os_a.partial_cmp(&os_b), bt_a.partial_cmp(&bt_b), "PartialOrd mismatch");
    }

    /// Tests Hash consistency for equal sets.
    #[test]
    fn hash_consistent_for_equal_sets(values in proptest::collection::vec(value_strategy(), TEST_SIZE)) {
        use std::hash::{DefaultHasher, Hash, Hasher};

        let os_set1: OSBTreeSet<i64> = values.iter().cloned().collect();
        let os_set2: OSBTreeSet<i64> = values.iter().cloned().collect();

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        os_set1.hash(&mut h1);
        os_set2.hash(&mut h2);

        prop_assert_eq!(h1.finish(), h2.finish(), "equal sets should have equal hashes");
    }
}

// ─── Range edge cases (empty ranges, leaf boundaries, tuple bounds) ──────────

use core::ops::Bound;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests range with tuple bounds using Excluded/Included combinations matches BTreeSet.
    #[test]
    fn range_tuple_bounds_match_btreeset(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // (Included, Included)
        let os_range: Vec<_> = os_set.range((Bound::Included(lo), Bound::Included(hi))).copied().collect();
        let bt_range: Vec<_> = bt_set.range((Bound::Included(lo), Bound::Included(hi))).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Included({}))) mismatch", lo, hi);

        // (Included, Excluded)
        let os_range: Vec<_> = os_set.range((Bound::Included(lo), Bound::Excluded(hi))).copied().collect();
        let bt_range: Vec<_> = bt_set.range((Bound::Included(lo), Bound::Excluded(hi))).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Excluded({}))) mismatch", lo, hi);

        // (Excluded, Included)
        let os_range: Vec<_> = os_set.range((Bound::Excluded(lo), Bound::Included(hi))).copied().collect();
        let bt_range: Vec<_> = bt_set.range((Bound::Excluded(lo), Bound::Included(hi))).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Excluded({}), Included({}))) mismatch", lo, hi);

        // (Excluded, Excluded) - only valid if lo < hi
        if lo < hi {
            let os_range: Vec<_> = os_set.range((Bound::Excluded(lo), Bound::Excluded(hi))).copied().collect();
            let bt_range: Vec<_> = bt_set.range((Bound::Excluded(lo), Bound::Excluded(hi))).copied().collect();
            prop_assert_eq!(&os_range, &bt_range, "range((Excluded({}), Excluded({}))) mismatch", lo, hi);
        }

        // (Unbounded, Included)
        let os_range: Vec<_> = os_set.range((Bound::<i64>::Unbounded, Bound::Included(hi))).copied().collect();
        let bt_range: Vec<_> = bt_set.range((Bound::<i64>::Unbounded, Bound::Included(hi))).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Unbounded, Included({}))) mismatch", hi);

        // (Included, Unbounded)
        let os_range: Vec<_> = os_set.range((Bound::Included(lo), Bound::<i64>::Unbounded)).copied().collect();
        let bt_range: Vec<_> = bt_set.range((Bound::Included(lo), Bound::<i64>::Unbounded)).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Unbounded)) mismatch", lo);
    }

    /// Tests range(k..k) produces empty range (empty range at any key).
    #[test]
    fn range_empty_at_key_matches_btreeset(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        key in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        // range(k..k) should always be empty
        let os_range: Vec<_> = os_set.range(key..key).copied().collect();
        let bt_range: Vec<_> = bt_set.range(key..key).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..{}) should be empty", key, key);
        prop_assert!(os_range.is_empty(), "range(k..k) must be empty");

        // Also test with explicit bounds
        let os_range: Vec<_> = os_set.range((Bound::Included(key), Bound::Excluded(key))).copied().collect();
        let bt_range: Vec<_> = bt_set.range((Bound::Included(key), Bound::Excluded(key))).copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range((Included({}), Excluded({}))) should be empty", key, key);
    }

    /// Tests range next_back doesn't escape bounds.
    #[test]
    fn range_next_back_respects_bounds(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Collect using next_back only
        let os_range: Vec<_> = os_set.range(lo..=hi).rev().copied().collect();
        let bt_range: Vec<_> = bt_set.range(lo..=hi).rev().copied().collect();
        prop_assert_eq!(&os_range, &bt_range, "range({}..={}).rev() mismatch", lo, hi);

        // Verify all collected values are in bounds
        for v in &os_range {
            prop_assert!(*v >= lo && *v <= hi, "value {} is outside range {}..={}", v, lo, hi);
        }
    }

    /// Tests extract_if with bounded range only removes values in range.
    #[test]
    fn extract_if_bounded_only_removes_in_range(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let original_set: BTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Extract all even values in range
        let os_extracted: Vec<_> = os_set.extract_if(lo..=hi, |v| v % 2 == 0).collect();

        // Verify extracted values are in bounds, even, and in sorted order
        let mut prev = None;
        for v in &os_extracted {
            prop_assert!(*v >= lo && *v <= hi, "extracted value {} is outside range {}..={}", v, lo, hi);
            prop_assert!(v % 2 == 0, "extracted value {} should be even", v);
            prop_assert!(original_set.contains(v), "extracted value {} should be in original set", v);
            if let Some(p) = prev {
                prop_assert!(v > &p, "extracted values should be in sorted order");
            }
            prev = Some(*v);
        }

        // Verify remaining values are either outside range or odd
        for v in os_set.iter() {
            let in_range = *v >= lo && *v <= hi;
            if in_range {
                prop_assert!(v % 2 != 0, "remaining value {} in range should be odd", v);
            }
            prop_assert!(original_set.contains(v), "remaining value {} should be in original set", v);
        }

        // Verify total count
        prop_assert_eq!(os_extracted.len() + os_set.len(), original_set.len(), "total count mismatch");
    }

    /// Tests extract_if with early drop (iterator not exhausted) retains unvisited values.
    #[test]
    fn extract_if_early_drop_retains_unvisited(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
    ) {
        let mut os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let original_len = os_set.len();

        // Take only the first 10 items from extract_if, then drop
        let os_extracted: Vec<_> = os_set.extract_if(.., |_| true).take(10).collect();

        // Extracted count should be at most 10
        prop_assert!(os_extracted.len() <= 10, "should extract at most 10 items");

        // Remaining items should be original_len - extracted_len
        prop_assert_eq!(os_set.len(), original_len - os_extracted.len(), "remaining count mismatch");

        // Extracted items should be the first N items in sorted order
        let sorted_values: Vec<_> = {
            let set: BTreeSet<_> = values.iter().cloned().collect();
            set.into_iter().collect()
        };
        for (i, v) in os_extracted.iter().enumerate() {
            prop_assert_eq!(*v, sorted_values[i], "extracted value at position {} should match sorted order", i);
        }
    }

    /// Tests interleaved next/next_back for Range iterator matches BTreeSet behavior.
    /// This specifically tests crossing detection across leaf boundaries.
    #[test]
    fn range_interleaved_next_next_back(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        // Collect using alternating next/next_back
        let mut os_from_front = Vec::new();
        let mut os_from_back = Vec::new();
        let mut bt_from_front = Vec::new();
        let mut bt_from_back = Vec::new();

        let mut os_iter = os_set.range(lo..=hi);
        let mut bt_iter = bt_set.range(lo..=hi);

        let mut toggle = true;
        loop {
            if toggle {
                match (os_iter.next(), bt_iter.next()) {
                    (Some(os_item), Some(bt_item)) => {
                        prop_assert_eq!(os_item, bt_item, "interleaved range next() mismatch");
                        os_from_front.push(*os_item);
                        bt_from_front.push(*bt_item);
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
                        os_from_back.push(*os_item);
                        bt_from_back.push(*bt_item);
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

        // Verify no duplicates
        let mut os_all: Vec<_> = os_from_front.iter().chain(os_from_back.iter()).copied().collect();
        os_all.sort();
        let os_dedup_len = os_all.len();
        os_all.dedup();
        prop_assert_eq!(os_all.len(), os_dedup_len, "range iterator yielded duplicate values");
    }

    /// Tests Range iterator is properly fused (once None, always None).
    #[test]
    fn range_fused_iterator(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        let mut iter = os_set.range(lo..=hi);

        // Exhaust the iterator
        while iter.next().is_some() {}

        // Verify FusedIterator: once None, always None
        for _ in 0..10 {
            prop_assert_eq!(iter.next(), None, "FusedIterator violation: next() returned Some after None");
            prop_assert_eq!(iter.next_back(), None, "FusedIterator violation: next_back() returned Some after None");
        }
    }

    /// Tests Range iterator with heavy back-to-front consumption pattern.
    #[test]
    fn range_heavy_next_back_pattern(
        values in proptest::collection::vec(value_strategy(), TEST_SIZE),
        lo in value_strategy(),
        hi in value_strategy(),
    ) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

        let mut os_iter = os_set.range(lo..=hi);
        let mut bt_iter = bt_set.range(lo..=hi);

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
                    os_items.push(*os);
                    bt_items.push(*bt);
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

/// Tests that range with start > end panics just like BTreeSet.
#[test]
#[should_panic]
fn range_start_greater_than_end_panics() {
    let set: OSBTreeSet<i32> = [1, 2, 3].into_iter().collect();
    // This should panic because 5 > 3
    // Use tuple bounds to avoid clippy::reversed_empty_ranges lint
    let _: Vec<_> = set.range((Bound::Included(5), Bound::Included(3))).collect();
}

/// Tests that range with (Excluded(x), Excluded(x)) for same x panics.
#[test]
#[should_panic]
fn range_excluded_excluded_same_bound_panics() {
    let set: OSBTreeSet<i32> = [1, 2, 3].into_iter().collect();
    // (Excluded(2), Excluded(2)) is an invalid range
    let _: Vec<_> = set.range((Bound::Excluded(2), Bound::Excluded(2))).collect();
}

/// Tests that range with (Excluded(x), Included(y)) where x > y panics.
#[test]
#[should_panic]
fn range_excluded_included_inverted_panics() {
    let set: OSBTreeSet<i32> = [1, 2, 3].into_iter().collect();
    // (Excluded(5), Included(3)) is an invalid range because 5 > 3
    let _: Vec<_> = set.range((Bound::Excluded(5), Bound::Included(3))).collect();
}

// ─── Out-of-bounds Rank indexing panic tests ──────────────────────────────────

/// Tests that Index<Rank> panics for out-of-bounds rank on non-empty set.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_rank_out_of_bounds_panics() {
    let set: OSBTreeSet<i32> = [1, 2, 3].into_iter().collect();
    // Set has 3 elements, so Rank(3) is out of bounds
    let _ = set[Rank(3)];
}

/// Tests that Index<Rank> panics on empty set.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_rank_empty_set_panics() {
    let set: OSBTreeSet<i32> = OSBTreeSet::new();
    let _ = set[Rank(0)];
}

/// Tests that Index<Rank> panics for very large out-of-bounds rank.
#[test]
#[should_panic(expected = "index out of bounds")]
fn index_rank_large_out_of_bounds_panics() {
    let set: OSBTreeSet<i32> = [1, 2].into_iter().collect();
    let _ = set[Rank(1000)];
}

// ─── Consuming iterator interleaved tests ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Tests into_iter with interleaved next/next_back matches BTreeSet.
    #[test]
    fn into_iter_interleaved_next_next_back(values in proptest::collection::vec(value_strategy(), 1..TEST_SIZE)) {
        let os_set: OSBTreeSet<i64> = values.iter().cloned().collect();
        let bt_set: BTreeSet<i64> = values.iter().cloned().collect();

        let mut os_iter = os_set.into_iter();
        let mut bt_iter = bt_set.into_iter();

        let mut os_items = Vec::new();
        let mut bt_items = Vec::new();

        let mut toggle = true;
        loop {
            if toggle {
                match (os_iter.next(), bt_iter.next()) {
                    (Some(os_item), Some(bt_item)) => {
                        prop_assert_eq!(os_item, bt_item, "into_iter interleaved next() mismatch");
                        os_items.push(os_item);
                        bt_items.push(bt_item);
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
                        os_items.push(os_item);
                        bt_items.push(bt_item);
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
        prop_assert_eq!(os_items_sorted.len(), dedup_len, "into_iter yielded duplicate values");
    }
}

// ─── Deterministic Insertion Pattern Tests ────────────────────────────────────

/// Helper function to generate deterministic pseudo-random values using LCG.
fn random_values_deterministic(n: usize) -> Vec<i64> {
    let mut values = Vec::with_capacity(n);
    let mut x: u64 = 12345; // Fixed seed for reproducibility
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        values.push((x >> 33) as i64);
    }
    values
}

mod insertion_pattern_tests {
    use super::*;
    use std::collections::BTreeSet;
    use wabi_tree::OSBTreeSet;

    const N: usize = 10_000;

    /// Tests ordered (ascending) inserts match BTreeSet.
    #[test]
    fn ordered_inserts_match_btreeset() {
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        // Insert in ascending order
        for i in 0..N as i64 {
            os_set.insert(i);
            bt_set.insert(i);
        }

        // Verify length
        assert_eq!(os_set.len(), N);
        assert_eq!(os_set.len(), bt_set.len());

        // Verify all values match
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        assert_eq!(os_items, bt_items, "ordered inserts content mismatch");

        // Verify first/last
        assert_eq!(os_set.first(), bt_set.first());
        assert_eq!(os_set.last(), bt_set.last());
    }

    /// Tests reverse-ordered (descending) inserts match BTreeSet.
    #[test]
    fn reverse_ordered_inserts_match_btreeset() {
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        // Insert in descending order
        for i in (0..N as i64).rev() {
            os_set.insert(i);
            bt_set.insert(i);
        }

        // Verify length
        assert_eq!(os_set.len(), N);
        assert_eq!(os_set.len(), bt_set.len());

        // Verify all values match
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        assert_eq!(os_items, bt_items, "reverse ordered inserts content mismatch");

        // Verify first/last
        assert_eq!(os_set.first(), bt_set.first());
        assert_eq!(os_set.last(), bt_set.last());
    }

    /// Tests random inserts match BTreeSet.
    #[test]
    fn random_inserts_match_btreeset() {
        let values = random_values_deterministic(N);
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        // Insert in random order
        for &v in &values {
            os_set.insert(v);
            bt_set.insert(v);
        }

        // Verify length matches (accounting for duplicates in random values)
        assert_eq!(os_set.len(), bt_set.len());

        // Verify all values match
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        assert_eq!(os_items, bt_items, "random inserts content mismatch");

        // Verify first/last
        assert_eq!(os_set.first(), bt_set.first());
        assert_eq!(os_set.last(), bt_set.last());
    }

    /// Tests ordered contains operations match BTreeSet.
    #[test]
    fn ordered_contains_match_btreeset() {
        let os_set: OSBTreeSet<i64> = (0..N as i64).collect();
        let bt_set: BTreeSet<i64> = (0..N as i64).collect();

        // Contains in ascending order
        for i in 0..N as i64 {
            assert_eq!(os_set.contains(&i), bt_set.contains(&i), "ordered contains({}) mismatch", i);
        }

        // Contains some non-existent values
        for i in [N as i64, N as i64 + 1, -1, -100] {
            assert_eq!(os_set.contains(&i), bt_set.contains(&i), "ordered contains({}) for missing value mismatch", i);
        }
    }

    /// Tests reverse-ordered contains operations match BTreeSet.
    #[test]
    fn reverse_ordered_contains_match_btreeset() {
        let os_set: OSBTreeSet<i64> = (0..N as i64).collect();
        let bt_set: BTreeSet<i64> = (0..N as i64).collect();

        // Contains in descending order
        for i in (0..N as i64).rev() {
            assert_eq!(os_set.contains(&i), bt_set.contains(&i), "reverse contains({}) mismatch", i);
        }
    }

    /// Tests random contains operations match BTreeSet.
    #[test]
    fn random_contains_match_btreeset() {
        let values = random_values_deterministic(N);
        let os_set: OSBTreeSet<i64> = values.iter().copied().collect();
        let bt_set: BTreeSet<i64> = values.iter().copied().collect();

        // Contains in random order (same as insertion order)
        for &v in &values {
            assert_eq!(os_set.contains(&v), bt_set.contains(&v), "random contains({}) mismatch", v);
        }
    }

    /// Tests ordered remove operations match BTreeSet.
    #[test]
    fn ordered_removes_match_btreeset() {
        let mut os_set: OSBTreeSet<i64> = (0..N as i64).collect();
        let mut bt_set: BTreeSet<i64> = (0..N as i64).collect();

        // Remove in ascending order
        for i in 0..N as i64 {
            let os_result = os_set.remove(&i);
            let bt_result = bt_set.remove(&i);
            assert_eq!(os_result, bt_result, "ordered remove({}) mismatch", i);
        }

        assert!(os_set.is_empty());
        assert_eq!(os_set.len(), bt_set.len());
    }

    /// Tests reverse-ordered remove operations match BTreeSet.
    #[test]
    fn reverse_ordered_removes_match_btreeset() {
        let mut os_set: OSBTreeSet<i64> = (0..N as i64).collect();
        let mut bt_set: BTreeSet<i64> = (0..N as i64).collect();

        // Remove in descending order
        for i in (0..N as i64).rev() {
            let os_result = os_set.remove(&i);
            let bt_result = bt_set.remove(&i);
            assert_eq!(os_result, bt_result, "reverse remove({}) mismatch", i);
        }

        assert!(os_set.is_empty());
        assert_eq!(os_set.len(), bt_set.len());
    }

    /// Tests random remove operations match BTreeSet.
    #[test]
    fn random_removes_match_btreeset() {
        let values = random_values_deterministic(N);
        let mut os_set: OSBTreeSet<i64> = values.iter().copied().collect();
        let mut bt_set: BTreeSet<i64> = values.iter().copied().collect();

        // Remove in random order (same as insertion order)
        for &v in &values {
            let os_result = os_set.remove(&v);
            let bt_result = bt_set.remove(&v);
            assert_eq!(os_result, bt_result, "random remove({}) mismatch", v);
        }

        assert!(os_set.is_empty());
        assert_eq!(os_set.len(), bt_set.len());
    }

    /// Tests full CRUD cycle with ordered inserts then removes.
    #[test]
    fn ordered_insert_then_ordered_remove() {
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        // Insert in ascending order
        for i in 0..N as i64 {
            os_set.insert(i);
            bt_set.insert(i);
        }

        // Verify iteration after inserts
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        assert_eq!(os_items, bt_items);

        // Remove in ascending order, checking iteration periodically
        for i in 0..N as i64 {
            os_set.remove(&i);
            bt_set.remove(&i);

            if i % 1000 == 999 {
                let os_items: Vec<_> = os_set.iter().copied().collect();
                let bt_items: Vec<_> = bt_set.iter().copied().collect();
                assert_eq!(os_items, bt_items, "iteration mismatch after removing {}", i);
            }
        }

        assert!(os_set.is_empty());
    }

    /// Tests full CRUD cycle with random inserts then removes.
    #[test]
    fn random_insert_then_random_remove() {
        let values = random_values_deterministic(N);
        let mut os_set: OSBTreeSet<i64> = OSBTreeSet::new();
        let mut bt_set: BTreeSet<i64> = BTreeSet::new();

        // Insert in random order
        for &v in &values {
            os_set.insert(v);
            bt_set.insert(v);
        }

        // Verify iteration after inserts
        let os_items: Vec<_> = os_set.iter().copied().collect();
        let bt_items: Vec<_> = bt_set.iter().copied().collect();
        assert_eq!(os_items, bt_items);

        // Remove in random order, checking iteration periodically
        for (i, &v) in values.iter().enumerate() {
            os_set.remove(&v);
            bt_set.remove(&v);

            if i % 1000 == 999 {
                let os_items: Vec<_> = os_set.iter().copied().collect();
                let bt_items: Vec<_> = bt_set.iter().copied().collect();
                assert_eq!(os_items, bt_items, "iteration mismatch after {} removals", i + 1);
            }
        }

        assert!(os_set.is_empty());
    }
}

// ─── Coverage-focused top-down tests ────────────────────────────────────────

#[test]
#[allow(clippy::double_ended_iterator_last)]
fn capacity_default_from_array_extend_refs_and_iter_traits() {
    let set: OSBTreeSet<i32> = OSBTreeSet::with_capacity(16);
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 16);
    assert_eq!(set.clone().capacity(), 16);

    let default_set: OSBTreeSet<i32> = Default::default();
    assert!(default_set.is_empty());
    let _ = format!("{:?}", default_set);

    let from_arr = OSBTreeSet::from([3, 1, 2]);
    let items: Vec<_> = from_arr.iter().copied().collect();
    assert_eq!(items, vec![1, 2, 3]);

    let data = [4, 5, 6];
    let mut extend_set = OSBTreeSet::new();
    extend_set.extend(data.iter());
    assert!(extend_set.contains(&4));
    assert!(extend_set.contains(&6));

    {
        let iter = extend_set.iter();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.clone().last(), Some(&6));
        let _ = format!("{:?}", iter.clone());
        let collected: Vec<_> = (&extend_set).into_iter().copied().collect();
        assert_eq!(collected, vec![4, 5, 6]);
    }

    let empty_iter: osbtree_set::Iter<'_, i32> = Default::default();
    assert_eq!(empty_iter.len(), 0);
    let _ = format!("{:?}", empty_iter.clone());

    let empty_into_iter: osbtree_set::IntoIter<i32> = Default::default();
    let _ = format!("{:?}", empty_into_iter);

    let into_iter_len = OSBTreeSet::from([7, 8, 9]).into_iter();
    assert_eq!(into_iter_len.len(), 3);

    {
        let range = extend_set.range(4..=5);
        assert_eq!(range.clone().count(), 2);
        assert_eq!(range.clone().last(), Some(&5));
        let _ = format!("{:?}", range.clone());
    }

    let empty_range: osbtree_set::Range<'_, i32> = Default::default();
    assert_eq!(empty_range.clone().count(), 0);
    let _ = format!("{:?}", empty_range);
}

#[test]
fn set_operations_algorithm_paths_and_traits() {
    let empty: OSBTreeSet<i32> = OSBTreeSet::new();
    let disjoint_a = OSBTreeSet::from([1, 2]);
    let disjoint_b = OSBTreeSet::from([10, 20]);

    let mut diff_empty = empty.difference(&disjoint_a);
    assert_eq!(diff_empty.next(), None);
    assert_eq!(diff_empty.size_hint(), (0, Some(0)));
    let _ = format!("{:?}", diff_empty.clone());

    let mut diff_disjoint = disjoint_a.difference(&disjoint_b);
    assert_eq!(diff_disjoint.size_hint().1, Some(2));
    assert_eq!(diff_disjoint.next(), Some(&1));
    let _ = format!("{:?}", diff_disjoint.clone());

    let other_empty: OSBTreeSet<i32> = OSBTreeSet::new();
    let diff_other_empty = disjoint_a.difference(&other_empty);
    assert_eq!(diff_other_empty.size_hint(), (2, Some(2)));
    let _ = format!("{:?}", diff_other_empty.clone());

    let stitch_left = OSBTreeSet::from([1, 2, 3, 4]);
    let stitch_right = OSBTreeSet::from([3, 4, 5, 6]);
    let stitch_items: Vec<_> = stitch_left.difference(&stitch_right).copied().collect();
    assert_eq!(stitch_items, vec![1, 2]);
    let _ = format!("{:?}", stitch_left.difference(&stitch_right).clone());

    let small_mixed = OSBTreeSet::from([1, 2]);
    let large_evens: OSBTreeSet<i32> = (0..2000).step_by(2).collect();
    let diff_search = small_mixed.difference(&large_evens);
    assert_eq!(diff_search.size_hint(), (0, Some(2)));
    let _ = format!("{:?}", diff_search.clone());
    let diff_search_items: Vec<_> = small_mixed.difference(&large_evens).copied().collect();
    assert_eq!(diff_search_items, vec![1]);
    assert_eq!(small_mixed.difference(&large_evens).min(), Some(&1));

    let empty_intersection = empty.intersection(&disjoint_a);
    assert_eq!(empty_intersection.clone().next(), None);
    assert_eq!(empty_intersection.size_hint(), (0, Some(0)));
    let _ = format!("{:?}", empty_intersection);

    let other_empty_intersection = disjoint_a.intersection(&other_empty);
    assert_eq!(other_empty_intersection.clone().next(), None);
    assert_eq!(other_empty_intersection.size_hint(), (0, Some(0)));
    let _ = format!("{:?}", other_empty_intersection);

    let disjoint_intersection = disjoint_a.intersection(&disjoint_b);
    assert_eq!(disjoint_intersection.clone().next(), None);
    assert_eq!(disjoint_intersection.size_hint(), (0, Some(0)));
    let _ = format!("{:?}", disjoint_intersection);

    let small_overlap = OSBTreeSet::from([500]);
    let large_overlap: OSBTreeSet<i32> = (0..2000).collect();
    let mut intersection_search_a = small_overlap.intersection(&large_overlap);
    assert_eq!(intersection_search_a.next(), Some(&500));
    let _ = format!("{:?}", intersection_search_a.clone());

    let mut intersection_search_b = large_overlap.intersection(&small_overlap);
    assert_eq!(intersection_search_b.next(), Some(&500));
    let _ = format!("{:?}", intersection_search_b.clone());

    let stitch_intersection = stitch_left.intersection(&stitch_right);
    let _ = format!("{:?}", stitch_intersection.clone());
    let stitch_intersection_items: Vec<_> = stitch_left.intersection(&stitch_right).copied().collect();
    assert_eq!(stitch_intersection_items, vec![3, 4]);
    assert_eq!(stitch_left.intersection(&stitch_right).min(), Some(&3));

    let small_odds = OSBTreeSet::from([1, 3, 5, 7, 9]);
    let intersection_search = small_odds.intersection(&large_evens);
    assert_eq!(intersection_search.size_hint(), (0, Some(5)));
    let _ = format!("{:?}", intersection_search.clone());
    assert_eq!(small_odds.intersection(&large_evens).next(), None);
    assert_eq!(small_odds.intersection(&large_evens).min(), None);

    let union_left = OSBTreeSet::from([1, 3, 5]);
    let union_right = OSBTreeSet::from([2, 3, 4]);
    let union = union_left.union(&union_right);
    assert_eq!(union.size_hint().0, 3);
    let _ = format!("{:?}", union.clone());
    assert_eq!(union_left.union(&union_right).min(), Some(&1));

    let symmetric = union_left.symmetric_difference(&union_right);
    assert_eq!(symmetric.size_hint().0, 0);
    let _ = format!("{:?}", symmetric.clone());
    assert_eq!(union_left.symmetric_difference(&union_right).min(), Some(&1));

    let mut extract_set = OSBTreeSet::from([1, 2, 3]);
    let extractor = extract_set.extract_if(.., |_| false);
    let _ = format!("{:?}", extractor);
}

// ─── Rank jumps: nth / nth_back ─────────────────────────────────────────────

/// `step_by` drives `nth` (and `nth_back` when reversed) on the set iterators.
#[test]
fn set_iterators_step_by_match_vec() {
    let set: OSBTreeSet<i64> = (0..50_000).filter(|k| k % 7 != 3).collect();
    let all: Vec<i64> = set.iter().copied().collect();
    let in_range: Vec<i64> = all.iter().copied().filter(|k| (1_000..40_000).contains(k)).collect();
    for step in [1, 2, 63, 64, 65, 127, 128, 129, 1_000, 30_000] {
        let expected: Vec<i64> = all.iter().copied().step_by(step).collect();
        assert_eq!(set.iter().step_by(step).copied().collect::<Vec<_>>(), expected, "step {step}");
        assert_eq!(set.clone().into_iter().step_by(step).collect::<Vec<_>>(), expected, "step {step}");
        let expected_back: Vec<i64> = all.iter().copied().rev().step_by(step).collect();
        assert_eq!(set.iter().rev().step_by(step).copied().collect::<Vec<_>>(), expected_back, "step {step}");
        let expected_range: Vec<i64> = in_range.iter().copied().step_by(step).collect();
        assert_eq!(set.range(1_000..40_000).step_by(step).copied().collect::<Vec<_>>(), expected_range);
        let expected_range_back: Vec<i64> = in_range.iter().copied().rev().step_by(step).collect();
        assert_eq!(set.range(1_000..40_000).rev().step_by(step).copied().collect::<Vec<_>>(), expected_range_back);
    }
}

// ─── O(1) terminals: last / count / min / max ───────────────────────────────

/// Builds a fresh iterator with `make`, consumes `front` elements from the front and `back`
/// from the back, then checks `len`/`last`/`count`/`min`/`max` against the remaining slice.
fn check_terminals<I, F>(make: F, key: impl Fn(I::Item) -> i64, expected: &[i64])
where
    F: Fn() -> I,
    I: DoubleEndedIterator<Item: Ord> + ExactSizeIterator,
{
    for (front, back) in [(0, 0), (1, 0), (0, 1), (63, 64), (1_000, 0), (0, 5_000), (6_000, 6_000)] {
        let lo = front.min(expected.len());
        let rest = &expected[lo..expected.len().saturating_sub(back).max(lo)];
        let fresh = || {
            let mut it = make();
            if front > 0 {
                it.nth(front - 1);
            }
            if back > 0 {
                it.nth_back(back - 1);
            }
            it
        };
        let ctx = format!("front {front}, back {back}");
        assert_eq!(fresh().len(), rest.len(), "len: {ctx}");
        assert_eq!(fresh().count(), rest.len(), "count: {ctx}");
        assert_eq!(fresh().last().map(&key), rest.last().copied(), "last: {ctx}");
        assert_eq!(fresh().min().map(&key), rest.first().copied(), "min: {ctx}");
        assert_eq!(fresh().max().map(&key), rest.last().copied(), "max: {ctx}");
    }
}

/// `last`/`count`/`min`/`max` on the set iterators match a `Vec` after partial consumption.
#[test]
fn set_iterator_terminals_match_vec() {
    let set: OSBTreeSet<i64> = (0..20_000).filter(|k| k % 7 != 3).collect();
    let all: Vec<i64> = set.iter().copied().collect();
    let in_range: Vec<i64> = all.iter().copied().filter(|k| (500..15_000).contains(k)).collect();
    check_terminals(|| set.iter(), |k| *k, &all);
    check_terminals(|| set.clone().into_iter(), |k| k, &all);
    check_terminals(|| set.range(500..15_000), |k| *k, &in_range);
    check_terminals(|| set.range(-5..0), |k| *k, &[]);
}

#[test]
fn set_range_len_tracks_both_ends() {
    let set: OSBTreeSet<u32> = (0..1_000).collect();
    let mut range = set.range(100..200);
    assert_eq!(range.len(), 100);
    range.next();
    range.next_back();
    assert_eq!(range.len(), 98);
    assert_eq!(range.nth(97), Some(&198));
    assert_eq!(range.len(), 0);
    assert_eq!(set.range(5..5).len(), 0);
}

#[test]
fn is_subset_edge_cases() {
    let empty: OSBTreeSet<i32> = OSBTreeSet::new();
    let one: OSBTreeSet<i32> = [5].into();
    let other_one: OSBTreeSet<i32> = [6].into();
    let evens: OSBTreeSet<i32> = (0..100).map(|v| v * 2).collect();

    // Empty sets.
    assert!(empty.is_subset(&empty));
    assert!(empty.is_subset(&one));
    assert!(!one.is_subset(&empty));
    assert!(one.is_superset(&empty));

    // Single elements and equal sets.
    assert!(one.is_subset(&one));
    assert!(!one.is_subset(&other_one));
    assert!(evens.is_subset(&evens));
    assert!(evens.is_superset(&evens));

    // Same span and same count in range, but different elements.
    let a: OSBTreeSet<i32> = [0, 3, 10].into();
    let b: OSBTreeSet<i32> = [0, 4, 10].into();
    assert!(!a.is_subset(&b));

    // Same span, enough elements in range overall, but one missing in the middle.
    let a: OSBTreeSet<i32> = [0, 2, 3, 198].into();
    assert!(!a.is_subset(&evens));
    let a: OSBTreeSet<i32> = [0, 2, 4, 198].into();
    assert!(a.is_subset(&evens));

    // Similar sizes (stitch path): one element swapped for a near neighbour.
    let mut a = evens.clone();
    a.remove(&100);
    a.insert(101);
    assert!(!a.is_subset(&evens));
    a.remove(&101);
    assert!(a.is_subset(&evens));
    assert!(evens.is_superset(&a));

    // A value below `other`'s minimum or above its maximum.
    let a: OSBTreeSet<i32> = [-2, 2].into();
    assert!(!a.is_subset(&evens));
    let a: OSBTreeSet<i32> = [2, 200].into();
    assert!(!a.is_subset(&evens));
}

/// An `Ord` that reverses once `FLIP` is set, to make `first() > last()` after construction.
mod flippable {
    use std::cell::Cell;
    use std::cmp::Ordering;

    thread_local! {
        pub static FLIP: Cell<bool> = const { Cell::new(false) };
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Flippable(pub i32);

    impl PartialOrd for Flippable {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Flippable {
        fn cmp(&self, other: &Self) -> Ordering {
            let ord = self.0.cmp(&other.0);
            if FLIP.with(Cell::get) {
                ord.reverse()
            } else {
                ord
            }
        }
    }
}

/// An inconsistent `Ord` may give unspecified results, but `is_subset` must not panic.
#[test]
fn is_subset_inconsistent_ord_does_not_panic() {
    use flippable::{FLIP, Flippable};

    let a: OSBTreeSet<Flippable> = (0..10).map(Flippable).collect();
    let b = a.clone();
    FLIP.with(|f| f.set(true));
    let _ = a.is_subset(&b);
    let _ = a.is_superset(&b);
    FLIP.with(|f| f.set(false));
}

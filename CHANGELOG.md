# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Performance
- `OSBTreeSet::is_subset` and `is_superset` now return early when the smaller set's bounds fall
  outside the larger set's, or when an O(log n) range count shows the larger set has too few
  elements in that span. Otherwise they choose by size between walking both sets side by side
  (visiting only the larger set's elements within the smaller set's span) and looking up each
  element (#22).

## [0.1.4] - 2026-11-01

### Added
- `osbtree_set::Range` now implements `ExactSizeIterator`, matching `osbtree_map::Range` (#19).

### Performance
- `nth` and `nth_back` on the map and set iterators (including ranges) jump by rank instead of
  stepping one element at a time, so `iter().skip(n)`, `rev().skip(n)` and `step_by` cost
  O(log n) per jump rather than O(n) (#18).
- `last` and `count` answer in O(1) instead of walking every remaining element on the map's
  `iter`, `iter_mut`, `into_iter`, `keys`, `values`, `values_mut`, `into_keys`, `into_values`,
  `range` and `range_mut` and the set's `iter`, `into_iter` and `range`, e.g. `map.keys().last()`
  or `set.range(a..b).count()` (#19). `min` and `max` do too on those whose items are sorted:
  all of them except `values`, `values_mut` and `into_values`. The set-operation iterators
  (`difference`, `intersection`, etc.) and `extract_if` are unchanged.

## [0.1.3] - 2026-10-18

### Fixed
- Undefined behavior (under the Stacked Borrows model) when holding several `&mut V` references
  from `iter_mut`, `values_mut`, or `range_mut` at once, e.g. after `.collect()`.
- If a key's `Clone` panicked while an insert or remove was rebalancing the tree, entries could
  be silently lost and `len()` would disagree with iteration. The tree now stays consistent.
- `get_by_rank`, `get_by_rank_mut`, and `Index<Rank>` could hang in release builds if the tree's
  internal sizes were inconsistent. They now panic instead.

### Changed
- Minimum supported Rust version is now 1.88.
- Dependencies are no longer pinned to exact versions, so `wabi_tree` resolves alongside other
  crates that use `smallvec` without version conflicts.
- Removed the `static_assertions` dependency.
- Updated `smallvec` to 1.16.3.

### Performance
- Removing elements (`remove`, `pop_first`, `pop_last`, and the `OSBTreeSet` equivalents) is
  about 8–20% faster.
- `entry(key).or_insert(..)` and related methods no longer clone the key or search the tree twice
  when the key is missing.
- Creating a `range`/`range_mut` iterator does half as many tree lookups.
- `OSBTreeSet::is_disjoint` returns immediately when the two sets' ranges don't overlap.

### Internal
- Simplified and deduplicated internal tree code and removed dead code.
- Updated dev-dependencies and GitHub Actions, and fixed new Clippy lints.

### CI
- New Miri job checks the `unsafe` code paths under both Stacked Borrows and Tree Borrows.
- The test suite also runs in release mode.
- Coverage and Miri use a pinned nightly toolchain, so a nightly regression can't break CI.

## [0.1.2] - 2026-02-10

### Changed
- Bump `criterion` dev-dependency from 0.8.1 to 0.8.2.
- Bump `proptest` dev-dependency from 1.9.0 to 1.10.0.

## [0.1.1] - 2026-01-27

### Changed
- Improve test coverage, including doctests in coverage runs.

### CI
- Update GitHub Actions dependencies (actions/checkout v6, codecov-action v5).
- General CI and repository housekeeping.

## [0.1.0] - 2026-01-25

### Added
- Initial public release of `OSBTreeMap` and `OSBTreeSet` with rank operations.

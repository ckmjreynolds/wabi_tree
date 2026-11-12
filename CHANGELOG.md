# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-11-12

### Added
- `remove_by_rank` on `OSBTreeMap` and `OSBTreeSet` removes the element at a sorted position,
  returning `None` if the rank is out of bounds (#31).
- `_with_rank` variants that return the element's rank from the same walk down the tree, where
  `rank_of` plus the plain method walks it twice (#31):
  - `OSBTreeMap`: `get_with_rank`, `get_with_rank_mut`, `insert_with_rank` (rank after inserting)
    and `remove_with_rank` (rank before removing).
  - `OSBTreeSet`: `get_with_rank`, `insert_with_rank`, `remove_with_rank` and `take_with_rank`.

### Performance
- `OSBTreeMap::entry` now records its path down the tree, so `VacantEntry::insert` (and with it
  `or_insert`, `or_insert_with` and `or_default`) and `OccupiedEntry::remove` no longer walk down
  from the root a second time. Entry inserts and removes now cost the same as `insert` and
  `remove`, where they were 10–35% slower before. `VacantEntry::insert_entry` also no longer
  searches the tree again after inserting. The map grows by one pointer (#32).
- `extract_if` on the map and set no longer collects and clones every key in the range before it
  starts. It now passes over rejected elements along the leaves and walks down from the root only
  to remove an element. On 1,000,000 `u64` keys it is 11× faster when nothing is extracted, 3.8×
  when a tenth is, and 2.5× when everything is (#36).
- `retain` on the map and set no longer clones the key of each element it removes, and removes
  through the same walk as `extract_if`: about the same speed when a tenth of 1,000,000 `u64`
  keys are removed, and 2.3× faster when all of them are (#36).
- `OccupiedEntry::remove` and `remove_entry` on an entry from `first_entry` or `last_entry` no
  longer clone the key and search for it: 2.3× faster (#36).
- Lookups and removals by rank skip over a node's children sixteen at a time rather than one at
  a time (#36).
- Removing an element clones fewer keys: none when an internal node merges with or borrows from
  its sibling, and none when the last element of the tree is removed, as `pop_last` does (#36).
- `OSBTreeSet::replace` swaps the stored element in place rather than removing it and inserting
  the new one (#36).

### Changed
- `retain` now removes each element as the predicate rejects it, as `BTreeMap` and `BTreeSet`
  do. If the predicate panics, the elements it has already rejected are gone; before, nothing
  was removed until the predicate had seen every element (#36).
- The `Debug` output of `osbtree_map::ExtractIf` shows `rank` in place of `index` (#36).
- Minimum `smallvec` version is now 1.16.5.

## [0.1.5] - 2026-11-01

### Changed
- When `collect`, `from_iter` or `From<[_; N]>` meets equal keys, it now keeps the last one, key
  and value both, as `BTreeMap` and `BTreeSet` do. Previously the map kept the first key with the
  last value, and the set kept the first element. `extend` and `insert` are unchanged (#20).

### Performance
- `collect`, `from_iter` and `From<[_; N]>` on the map and set now sort the input, then build the
  tree in O(n) instead of inserting one element at a time: 5–14× faster on 10,000
  elements, on par with `BTreeMap`/`BTreeSet`. Leaves built from sorted input are now nearly
  full rather than half full, so the tree uses about half as many nodes (#20).
- `OSBTreeSet::is_subset` and `is_superset` now return early when the smaller set's bounds fall
  outside the larger set's, or when an O(log n) range count shows the larger set has too few
  elements in that span. Otherwise they choose by size between walking both sets side by side
  (visiting only the larger set's elements within the smaller set's span) and looking up each
  element (#22).
- `split_off` on the map and set no longer clones every key it moves and removes and inserts
  them one at a time. It now moves whichever side of the split is smaller out of the tree in
  bulk, rebuilds it in O(n), and repairs the cut edge of the other side, for
  O(min(m, n − m) + log n): 6–900× faster on 10,000 elements, depending on the split point (#21).
- `append` on the map and set now grafts the smaller tree onto the larger when their key ranges
  don't overlap, in O(min(n, m) + log(n + m)), and merges and rebuilds in O(n + m) when they
  overlap and are of similar size. Inserting one element at a time is kept for overlapping
  ranges when `other` is much smaller. On 10,000 elements, appending disjoint halves is 4.9×
  faster and interleaved halves 3.9× faster, more than twice as fast as `BTreeMap` (#21).
  If `K::clone` panics, `split_off` and a disjoint `append` leave both trees as they were. A
  merging `append` drops every entry and leaves both maps empty; it used to keep what it had
  inserted before the panic.

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

[Unreleased]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ckmjreynolds/wabi_tree/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ckmjreynolds/wabi_tree/releases/tag/v0.1.0

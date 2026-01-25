# wabi_tree

[![Crates.io](https://img.shields.io/crates/v/wabi_tree.svg)](https://crates.io/crates/wabi_tree)
[![Documentation](https://docs.rs/wabi_tree/badge.svg)](https://docs.rs/wabi_tree)
[![CI](https://github.com/ckmjreynolds/wabi_tree/actions/workflows/ci.yml/badge.svg)](https://github.com/ckmjreynolds/wabi_tree/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ckmjreynolds/wabi_tree/graph/badge.svg)](https://codecov.io/gh/ckmjreynolds/wabi_tree)
[![License](https://img.shields.io/crates/l/wabi_tree.svg)](https://github.com/ckmjreynolds/wabi_tree#license)

Order-statistic B-tree collections for Rust.

`wabi_tree` provides `OSBTreeMap` and `OSBTreeSet`, which are drop-in replacements for the standard library's `BTreeMap` and `BTreeSet` with additional **O(log n) order-statistic operations**:

- **`get_by_rank(rank)`** - Get the element at a given sorted position
- **`rank_of(key)`** - Get the sorted position of a key
- **Indexing by `Rank`** - e.g., `map[Rank(0)]` for the first element

## Features

- **`no_std` compatible** - Only requires `alloc`, no standard library dependency
- **Drop-in replacement** - API mirrors `std::collections::BTreeMap`/`BTreeSet`
- **O(log n) rank operations** - Efficient order-statistic queries via subtree size augmentation
- **Cache-efficient B+tree** - Contiguous node storage with linked leaves for fast iteration

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
wabi_tree = "0.1"
```

## Quick Start

```rust
use wabi_tree::{OSBTreeMap, Rank};

let mut scores = OSBTreeMap::new();
scores.insert("Alice", 100);
scores.insert("Bob", 85);
scores.insert("Carol", 92);

// Standard BTreeMap operations work as expected
assert_eq!(scores.get(&"Bob"), Some(&85));
assert_eq!(scores.len(), 3);

// Order-statistic operations (O(log n))
// Get the median (rank 1 = second element in sorted order by key)
let (name, score) = scores.get_by_rank(1).unwrap();
assert_eq!(*name, "Bob"); // Keys are sorted alphabetically

// Find the rank of a key
assert_eq!(scores.rank_of(&"Carol"), Some(2)); // Carol is third alphabetically

// Index by rank
assert_eq!(scores[Rank(0)], 100); // Alice's score (first alphabetically)
```

## API Overview

### OSBTreeMap

All standard `BTreeMap` methods are supported, plus:

| Method | Description | Complexity |
|--------|-------------|------------|
| `get_by_rank(rank)` | Get key-value pair at sorted position | O(log n) |
| `get_by_rank_mut(rank)` | Get key and mutable value at sorted position | O(log n) |
| `rank_of(key)` | Get the sorted position of a key | O(log n) |
| `map[Rank(i)]` | Index by rank (panics if out of bounds) | O(log n) |

### OSBTreeSet

All standard `BTreeSet` methods are supported, plus:

| Method | Description | Complexity |
|--------|-------------|------------|
| `get_by_rank(rank)` | Get element at sorted position | O(log n) |
| `rank_of(value)` | Get the sorted position of a value | O(log n) |
| `set[Rank(i)]` | Index by rank (panics if out of bounds) | O(log n) |

### Core Operations Complexity

| Operation | Complexity |
|-----------|------------|
| `get`, `contains_key` | O(log n) |
| `insert`, `remove` | O(log n) |
| `first_key_value`, `last_key_value` | O(1) |
| `pop_first`, `pop_last` | O(log n) |
| `len`, `is_empty` | O(1) |
| `iter`, `keys`, `values` | O(1) to create, O(1) amortized per step |
| `range` | O(log n) to create, O(1) amortized per step |

## Use Cases

Order-statistic trees are useful when you need both:
1. Fast key-based lookups (like a regular map/set)
2. Fast positional access (like an array)

Example applications:
- **Leaderboards** - Find a player's rank, get the top N players
- **Percentile calculations** - Find the median or any percentile in O(log n)
- **Sliding window statistics** - Maintain sorted data with efficient rank queries
- **Database indexing** - Support both key lookups and OFFSET/LIMIT queries

## Implementation

`wabi_tree` uses a B+tree variant where:
- All keys and values are stored in leaf nodes
- Internal nodes contain only separator keys and child pointers
- Leaves are linked for efficient sequential iteration
- Each internal node stores subtree sizes, enabling O(log n) rank operations

This design provides:
- Better cache locality than traditional binary search trees
- Efficient iteration via the leaf chain
- O(log n) complexity for all rank-based operations

## Comparison with std::collections::BTreeMap

| Feature | `BTreeMap` | `OSBTreeMap` |
|---------|------------|--------------|
| Key-based lookup | O(log n) | O(log n) |
| Insertion/removal | O(log n) | O(log n) |
| Get by rank | O(n) | **O(log n)** |
| Find rank of key | O(n) | **O(log n)** |
| Memory overhead | Lower | Higher (stores subtree sizes) |
| `no_std` support | No | **Yes** |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

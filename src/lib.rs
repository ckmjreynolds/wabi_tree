//! Order-statistic B-tree collections for Rust.
//!
//! This crate provides [`OSBTreeMap`] and [`OSBTreeSet`], which are drop-in replacements
//! for the standard library's `BTreeMap` and `BTreeSet` with additional O(log n)
//! order-statistic operations:
//!
//! - [`get_by_rank`](OSBTreeMap::get_by_rank) - Get the element at a given sorted position
//! - [`rank_of`](OSBTreeMap::rank_of) - Get the sorted position of a key
//! - Indexing by [`Rank`] - e.g., `map[Rank(0)]` for the first element
//!
//! # Example
//!
//! ```
//! use wabi_tree::{OSBTreeMap, Rank};
//!
//! let mut scores = OSBTreeMap::new();
//! scores.insert("Alice", 100);
//! scores.insert("Bob", 85);
//! scores.insert("Carol", 92);
//!
//! // Standard BTreeMap operations work as expected
//! assert_eq!(scores.get(&"Bob"), Some(&85));
//! assert_eq!(scores.len(), 3);
//!
//! // Order-statistic operations (O(log n))
//! // Get the median (rank 1 = second element in sorted order)
//! let (name, score) = scores.get_by_rank(1).unwrap();
//! assert_eq!(*name, "Bob"); // Keys are sorted alphabetically
//!
//! // Find the rank of a key
//! assert_eq!(scores.rank_of(&"Carol"), Some(2)); // Carol is third alphabetically
//!
//! // Index by rank
//! assert_eq!(scores[Rank(0)], 100); // Alice's score (first alphabetically)
//! ```
//!
//! # Features
//!
//! - **`no_std` compatible** - Only requires `alloc`, no standard library dependency
//! - **Drop-in replacement** - API mirrors `std::collections::BTreeMap`/`BTreeSet`
//! - **O(log n) rank operations** - Efficient order-statistic queries via subtree size augmentation
//! - **Cache-efficient** - B+tree structure with contiguous node storage
//!
//! # Implementation
//!
//! The collections are implemented as B+trees (all data in leaves, linked leaf chain)
//! with subtree size augmentation. Each internal node tracks the sizes of its subtrees,
//! enabling O(log n) rank-based access without full traversal.

#![no_std]
// These forbid rules and lint groups are meant to be very restrictive.
// NOTE: We have to allow unsafe code in order to performantly match BTreeMap and BTreeSet's functionality.
// #![forbid(unsafe_code)]
#![forbid(keyword_idents)]
#![forbid(non_ascii_idents)]
#![forbid(unreachable_pub)]
#![warn(clippy::all)]
#![warn(clippy::cargo)]
#![warn(clippy::pedantic)]
// Enable coverage attributes for nightly builds.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
// Allow unused code during initial development.
#![allow(unused)]
// Allow todo!() placeholders during API scaffolding.
#![allow(clippy::todo)]

extern crate alloc;

mod order_statistic;
mod raw;

pub mod osbtree_map;
pub mod osbtree_set;

pub use order_statistic::Rank;
pub use osbtree_map::OSBTreeMap;
pub use osbtree_set::OSBTreeSet;

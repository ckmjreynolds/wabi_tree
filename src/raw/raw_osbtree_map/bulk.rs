//! Structural bulk operations: building a packed subtree, cutting a tree at a rank, and
//! joining a subtree onto one edge of a tree. `split_off` and `append` are built from these.
//!
//! # Panic safety
//!
//! Separators are clones of keys, and `K::clone` may panic. Every operation here that could
//! leave a tree half-changed takes the separators it needs already cloned, so it never clones
//! itself: callers clone everything first (see [`RawOSBTreeMap::split_off_at_rank`] and
//! [`RawOSBTreeMap::append`]), and a panic while cloning leaves both trees untouched.

use alloc::vec::Vec;
use core::cmp::Ordering;

use super::{Path, PathElement, RawOSBTreeMap, child_for_rank};
use crate::raw::arena::Arena;
use crate::raw::handle::Handle;
use crate::raw::node::{InternalNode, LeafNode, MAX_CHILDREN, MAX_KEYS, MIN_CHILDREN, MIN_LEAF_KEYS, Node};
use crate::raw::size::Size;

/// Which edge of a tree an operation works on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Side {
    Left,
    Right,
}

impl Side {
    const fn flip(self) -> Self {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }
}

/// A packed subtree built by [`RawOSBTreeMap::build_packed`]: its nodes and values live in the
/// host tree's arenas, but nothing in the host refers to it yet.
#[derive(Clone, Copy)]
pub(super) struct Subtree {
    pub(super) root: Handle,
    pub(super) height: usize,
    pub(super) len: usize,
    pub(super) first_leaf: Handle,
    pub(super) last_leaf: Handle,
}

/// Splits `total` items into `parts` counts that differ by at most one.
fn even_split(total: usize, parts: usize) -> impl Iterator<Item = usize> {
    (0..parts).map(move |i| total / parts + usize::from(i < total % parts))
}

/// The number of leaves in a packed tree of `len > 0` entries.
const fn packed_leaf_count(len: usize) -> usize {
    len.div_ceil(MAX_KEYS)
}

/// The number of nodes and the height of a packed tree of `len > 0` entries.
pub(super) fn packed_shape(len: usize) -> (usize, usize) {
    let mut level = packed_leaf_count(len);
    let (mut nodes, mut height) = (level, 0);
    while level > 1 {
        level = level.div_ceil(MAX_CHILDREN);
        nodes += level;
        height += 1;
    }
    (nodes, height)
}

/// The positions, among `len > 0` sorted entries, of the keys a packed tree uses as
/// separators: the last entry of every leaf but the last. Each leaf boundary is exactly one
/// separator somewhere in the tree.
fn separator_positions(len: usize) -> impl Iterator<Item = usize> {
    let leaves = packed_leaf_count(len);
    even_split(len, leaves).take(leaves - 1).scan(0, |end, count| {
        *end += count;
        Some(*end - 1)
    })
}

impl<K: Clone + Ord, V> RawOSBTreeMap<K, V> {
    /// Builds `entries` (strictly increasing keys) into this tree's arenas as a packed,
    /// detached subtree, in O(n). Each level is split into as few nodes as fit, with entries
    /// (or children) spread evenly, so every node but the subtree's root is at least half full.
    ///
    /// `separator(nodes, leaf, i)` supplies the separator for leaf boundary `i`, whose left
    /// side ends with `leaf`. The subtree's leaves are linked to each other only.
    pub(super) fn build_packed<I>(
        &mut self,
        entries: I,
        mut separator: impl FnMut(&Arena<Node<K>>, Handle, usize) -> K,
    ) -> Subtree
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut entries = entries.into_iter();
        let len = entries.len();
        debug_assert!(len > 0, "`build_packed()` - no entries");
        let leaf_count = packed_leaf_count(len);
        let (node_count, _) = packed_shape(len);
        self.nodes.reserve(node_count);
        self.values.reserve(len);

        // Each node on the level being built: (handle, subtree size, rightmost leaf, and that
        // leaf's index). The rightmost leaf ends at the boundary the node's separator stands for.
        let mut level: Vec<(Handle, usize, Handle, usize)> = Vec::with_capacity(leaf_count);
        let (mut first_leaf, mut last_leaf): (Option<Handle>, Option<Handle>) = (None, None);
        for (index, count) in even_split(len, leaf_count).enumerate() {
            let mut leaf = LeafNode::new();
            for _ in 0..count {
                let (key, value) = entries.next().expect("`build_packed()` - fewer entries than `len()`");
                debug_assert!(
                    leaf.last_key()
                        .or_else(|| last_leaf.and_then(|prev| self.nodes.get(prev).as_leaf().last_key()))
                        .is_none_or(|last| *last < key),
                    "`build_packed()` - keys must be strictly increasing"
                );
                let value_handle = self.values.alloc(value);
                leaf.push(key, value_handle);
            }
            leaf.set_prev(last_leaf);
            let handle = self.nodes.alloc(Node::Leaf(leaf));
            match last_leaf {
                Some(prev) => self.nodes.get_mut(prev).as_leaf_mut().set_next(Some(handle)),
                None => first_leaf = Some(handle),
            }
            last_leaf = Some(handle);
            level.push((handle, count, handle, index));
        }
        debug_assert!(entries.next().is_none(), "`build_packed()` - more entries than `len()`");

        let mut height = 0;
        while level.len() > 1 {
            let parent_count = level.len().div_ceil(MAX_CHILDREN);
            let mut parents = Vec::with_capacity(parent_count);
            let mut rest = level.as_slice();
            for count in even_split(level.len(), parent_count) {
                let (group, tail) = rest.split_at(count);
                rest = tail;
                let mut node = InternalNode::new();
                let (first, first_size, _, _) = group[0];
                node.set_first_child(first, Size::from_usize(first_size));
                for pair in group.windows(2) {
                    let ((_, _, left_leaf, left_index), (child, child_size, _, _)) = (pair[0], pair[1]);
                    node.push_child(separator(&self.nodes, left_leaf, left_index), child, Size::from_usize(child_size));
                }
                let size = group.iter().map(|&(_, size, _, _)| size).sum();
                node.set_size(Size::from_usize(size));
                let (_, _, rightmost, rightmost_index) = group[count - 1];
                parents.push((self.nodes.alloc(Node::Internal(node)), size, rightmost, rightmost_index));
            }
            level = parents;
            height += 1;
        }

        Subtree {
            root: level[0].0,
            height,
            len,
            first_leaf: first_leaf.expect("built at least one leaf"),
            last_leaf: last_leaf.expect("built at least one leaf"),
        }
    }

    /// Builds `entries` into this tree's arenas like [`Self::build_packed`], taking the
    /// separators from `separators` (as returned by [`Self::clone_separators`] for the same
    /// entries) instead of cloning them.
    fn build_packed_with(&mut self, entries: Vec<(K, V)>, separators: Vec<K>) -> Subtree {
        let mut separators: Vec<Option<K>> = separators.into_iter().map(Some).collect();
        self.build_packed(entries, |_, _, index| {
            separators[index].take().expect("`build_packed_with()` - one separator per leaf boundary")
        })
    }

    /// Makes `subtree`, built into this empty tree's arenas, the whole tree.
    fn install(&mut self, subtree: Subtree) {
        debug_assert!(self.root.is_none());
        self.root = Some(subtree.root);
        self.first_leaf = Some(subtree.first_leaf);
        self.last_leaf = Some(subtree.last_leaf);
        self.len = subtree.len;
    }

    /// Builds a tree from sorted entries, cloning separators as needed.
    pub(super) fn build_from_sorted<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let entries = entries.into_iter();
        let len = entries.len();
        if len == 0 {
            return Self::new();
        }
        let (node_count, _) = packed_shape(len);
        let mut tree = Self::with_arena_capacities(node_count, len);
        let subtree = tree.build_packed(entries, |nodes, leaf, _| {
            nodes.get(leaf).as_leaf().last_key().expect("built leaves are non-empty").clone()
        });
        tree.install(subtree);
        tree
    }

    /// Builds a tree from sorted entries without cloning, given their separators.
    fn build_from_sorted_with(entries: Vec<(K, V)>, separators: Vec<K>) -> Self {
        if entries.is_empty() {
            return Self::new();
        }
        let (node_count, _) = packed_shape(entries.len());
        let mut tree = Self::with_arena_capacities(node_count, entries.len());
        let subtree = tree.build_packed_with(entries, separators);
        tree.install(subtree);
        tree
    }

    /// Clones the separators a packed tree would need for the `len` entries starting at rank
    /// `start` of this tree.
    fn clone_separators(&self, start: usize, len: usize) -> Vec<K> {
        if len == 0 {
            return Vec::new();
        }
        separator_positions(len)
            .map(|offset| self.get_by_rank(start + offset).expect("separator rank is in range").0.clone())
            .collect()
    }

    /// Returns the tree's height: 0 for a lone leaf.
    fn height(&self) -> usize {
        let mut height = 0;
        let Some(mut node) = self.root else {
            return 0;
        };
        while let Node::Internal(internal) = self.nodes.get(node) {
            node = internal.child(0);
            height += 1;
        }
        height
    }

    /// Descends to the element at `rank`, recording the path. Returns its leaf and index.
    fn descend_to_rank(&self, root: Handle, rank: usize, path: &mut Path) -> (Handle, usize) {
        let mut remaining = rank;
        let leaf = self.descend_with_path(root, path, |internal| {
            let (child, before) = child_for_rank(internal, remaining).expect("`descend_to_rank()` - rank in range");
            remaining -= before;
            child
        });
        (leaf, remaining)
    }

    // ─── Border fix-up ──────────────────────────────────────────────────────────

    /// Predicts what [`Self::fix_border`] will do at the leaf level, where the border leaf will
    /// hold `border_len` entries and its neighbour on the inside is `neighbor` (unchanged by the
    /// fix-up above it). If it will move entries from the neighbour into the border leaf, returns
    /// a clone of the key that becomes their separator; that is the only clone a fix-up needs.
    fn clone_leaf_separator(&self, border_len: usize, neighbor: Handle, side: Side) -> Option<K> {
        let neighbor = self.nodes.get(neighbor).as_leaf();
        let neighbor_len = neighbor.key_count();
        if border_len + neighbor_len <= MAX_KEYS || border_len >= MIN_LEAF_KEYS {
            return None;
        }
        let count = MIN_LEAF_KEYS - border_len;
        Some(match side {
            // The neighbour is on the left and keeps its first `neighbor_len - count` entries.
            Side::Right => neighbor.key(neighbor_len - count - 1).clone(),
            // The neighbour is on the right and hands over its first `count` entries.
            Side::Left => neighbor.key(count - 1).clone(),
        })
    }

    /// Collapses internal roots with a single child.
    fn fix_top(&mut self) {
        while let Some(root) = self.root {
            match self.nodes.get(root) {
                Node::Internal(internal) if internal.child_count() == 1 => {
                    let child = internal.child(0);
                    self.nodes.free(root);
                    self.root = Some(child);
                }
                _ => break,
            }
        }
    }

    /// Restores the fill invariants along one edge of the tree, where nodes may be underfull
    /// (but not empty). Every node off that edge must be valid.
    ///
    /// Works top-down, as `std`'s `fix_right_border` does: at each level the border child is
    /// merged with its neighbour if they fit in one node, and otherwise topped up from it, an
    /// internal node to `MIN_CHILDREN + 1` children so that a merge below can't leave it short.
    /// None of this clones, except topping up a leaf, which takes `leaf_separator` from
    /// [`Self::clone_leaf_separator`].
    fn fix_border(&mut self, side: Side, mut leaf_separator: Option<K>) {
        self.fix_top();
        let Some(mut node) = self.root else {
            return;
        };
        while let Node::Internal(parent) = self.nodes.get(node) {
            let count = parent.child_count();
            debug_assert!(count >= 2, "`fix_border()` - border parent has a single child");
            let left_index = match side {
                Side::Right => count - 2,
                Side::Left => 0,
            };
            if matches!(self.nodes.get(parent.child(left_index)), Node::Leaf(_)) {
                self.fix_leaf_pair(node, left_index, side, leaf_separator.take());
                break;
            }
            node = self.fix_internal_pair(node, left_index, side);
        }
        debug_assert!(leaf_separator.is_none(), "`fix_border()` - leaf separator was not needed");
        self.fix_top();
    }

    /// Fixes the internal children `index` and `index + 1` of `parent`, one of which is on the
    /// border. Returns the border child afterwards.
    fn fix_internal_pair(&mut self, parent: Handle, index: usize, side: Side) -> Handle {
        let parent_node = self.nodes.get(parent).as_internal();
        let (left, right) = (parent_node.child(index), parent_node.child(index + 1));
        let left_count = self.nodes.get(left).as_internal().child_count();
        let right_count = self.nodes.get(right).as_internal().child_count();

        if left_count + right_count <= MAX_CHILDREN {
            let (separator, right_handle, _) = self.nodes.get_mut(parent).as_internal_mut().remove_child(index);
            debug_assert_eq!(right_handle, right);
            let Node::Internal(right_node) = self.nodes.take(right) else {
                unreachable!("siblings are at the same height")
            };
            let left_node = self.nodes.get_mut(left).as_internal_mut();
            left_node.merge_with_right(separator, right_node);
            let merged_size = left_node.size();
            self.nodes.get_mut(parent).as_internal_mut().set_child_size(index, merged_size);
            return left;
        }

        match side {
            Side::Right if right_count <= MIN_CHILDREN => {
                let mut moved = self.nodes.get_mut(left).as_internal_mut().take_back(MIN_CHILDREN + 1 - right_count);
                let separator = moved.keys.remove(0);
                let old = self.nodes.get_mut(parent).as_internal_mut().replace_key(index, separator);
                moved.keys.push(old);
                self.nodes.get_mut(right).as_internal_mut().prepend(moved);
            }
            Side::Left if left_count <= MIN_CHILDREN => {
                let mut moved = self.nodes.get_mut(right).as_internal_mut().take_front(MIN_CHILDREN + 1 - left_count);
                let separator = moved.keys.pop().expect("moved at least one child");
                let old = self.nodes.get_mut(parent).as_internal_mut().replace_key(index, separator);
                moved.keys.insert(0, old);
                self.nodes.get_mut(left).as_internal_mut().extend(moved);
            }
            _ => {}
        }
        self.refresh_pair_sizes(parent, index);
        match side {
            Side::Right => right,
            Side::Left => left,
        }
    }

    /// Fixes the leaf children `index` and `index + 1` of `parent`, one of which is on the
    /// border.
    fn fix_leaf_pair(&mut self, parent: Handle, index: usize, side: Side, separator: Option<K>) {
        let parent_node = self.nodes.get(parent).as_internal();
        let (left, right) = (parent_node.child(index), parent_node.child(index + 1));
        let left_len = self.nodes.get(left).as_leaf().key_count();
        let right_len = self.nodes.get(right).as_leaf().key_count();

        if left_len + right_len <= MAX_KEYS {
            debug_assert!(separator.is_none(), "`fix_leaf_pair()` - merging needs no separator");
            let (_, right_handle, _) = self.nodes.get_mut(parent).as_internal_mut().remove_child(index);
            debug_assert_eq!(right_handle, right);
            let Node::Leaf(right_leaf) = self.nodes.take(right) else {
                unreachable!("siblings are at the same height")
            };
            let left_leaf = self.nodes.get_mut(left).as_leaf_mut();
            left_leaf.merge_with_right(right_leaf);
            match left_leaf.next() {
                Some(next) => self.nodes.get_mut(next).as_leaf_mut().set_prev(Some(left)),
                None => self.last_leaf = Some(left),
            }
            self.nodes.get_mut(parent).as_internal_mut().set_child_size(index, Size::from_usize(left_len + right_len));
            return;
        }

        match side {
            Side::Right if right_len < MIN_LEAF_KEYS => {
                let separator = separator.expect("`fix_leaf_pair()` - topping up needs a separator");
                let keep = left_len - (MIN_LEAF_KEYS - right_len);
                let left_leaf = self.nodes.get_mut(left).as_leaf_mut();
                debug_assert!(*left_leaf.key(keep - 1) == separator, "`fix_leaf_pair()` - wrong separator");
                let moved = left_leaf.take_tail(keep);
                self.nodes.get_mut(right).as_leaf_mut().prepend(moved);
                drop(self.nodes.get_mut(parent).as_internal_mut().replace_key(index, separator));
            }
            Side::Left if left_len < MIN_LEAF_KEYS => {
                let separator = separator.expect("`fix_leaf_pair()` - topping up needs a separator");
                let count = MIN_LEAF_KEYS - left_len;
                let right_leaf = self.nodes.get_mut(right).as_leaf_mut();
                debug_assert!(*right_leaf.key(count - 1) == separator, "`fix_leaf_pair()` - wrong separator");
                let moved = right_leaf.take_head(count);
                self.nodes.get_mut(left).as_leaf_mut().extend(moved);
                drop(self.nodes.get_mut(parent).as_internal_mut().replace_key(index, separator));
            }
            _ => debug_assert!(separator.is_none(), "`fix_leaf_pair()` - separator was not needed"),
        }
        self.refresh_pair_sizes(parent, index);
    }

    /// Re-records the sizes of children `index` and `index + 1` of `parent`. Their total, and
    /// so `parent`'s own size, is unchanged.
    fn refresh_pair_sizes(&mut self, parent: Handle, index: usize) {
        let parent_node = self.nodes.get(parent).as_internal();
        let left_size = self.nodes.get(parent_node.child(index)).subtree_size();
        let right_size = self.nodes.get(parent_node.child(index + 1)).subtree_size();
        let parent_node = self.nodes.get_mut(parent).as_internal_mut();
        parent_node.set_child_size(index, left_size);
        parent_node.set_child_size(index + 1, right_size);
    }

    // ─── Cutting ────────────────────────────────────────────────────────────────

    /// Clones what [`Self::cut`] at `rank` from `side` needs: the leaf separator for its
    /// fix-up. Requires `0 < rank < len`.
    fn clone_cut_separator(&self, rank: usize, side: Side) -> Option<K> {
        match side {
            Side::Right => {
                let (leaf, index) = self.locate_rank(rank - 1)?;
                let prev = self.nodes.get(leaf).as_leaf().prev()?;
                self.clone_leaf_separator(index + 1, prev, side)
            }
            Side::Left => {
                let (leaf, index) = self.locate_rank(rank)?;
                let leaf_node = self.nodes.get(leaf).as_leaf();
                let next = leaf_node.next()?;
                self.clone_leaf_separator(leaf_node.key_count() - index, next, side)
            }
        }
    }

    /// Removes the elements at ranks `rank..len` (`Side::Right`) or `0..rank` (`Side::Left`) and
    /// returns them in key order, in O(elements removed + log n). Requires `0 < rank < len`, and
    /// `leaf_separator` from [`Self::clone_cut_separator`]. Clones nothing.
    fn cut(&mut self, rank: usize, side: Side, leaf_separator: Option<K>) -> Vec<(K, V)> {
        debug_assert!(0 < rank && rank < self.len, "`cut()` - rank out of range");
        let root = self.root.expect("`cut()` - tree is non-empty");
        let moved_len = match side {
            Side::Right => self.len - rank,
            Side::Left => rank,
        };
        let mut out = Vec::with_capacity(moved_len);
        let mut path = Path::new();

        match side {
            Side::Right => {
                // Descend to the last element kept, so every node on the path keeps something.
                let (leaf, index) = self.descend_to_rank(root, rank - 1, &mut path);
                let leaf_node = self.nodes.get_mut(leaf).as_leaf_mut();
                let moved = leaf_node.take_tail(index + 1);
                leaf_node.set_next(None);
                self.last_leaf = Some(leaf);
                self.push_entries(moved, &mut out);
                // Bottom-up, the subtrees right of the path hold ever larger keys.
                for &PathElement {
                    node,
                    child_index,
                } in path.iter().rev()
                {
                    let count = self.nodes.get(node).as_internal().child_count();
                    for i in child_index + 1..count {
                        let child = self.nodes.get(node).as_internal().child(i);
                        self.drain_subtree(child, &mut out);
                    }
                    self.nodes.get_mut(node).as_internal_mut().truncate_children(child_index + 1);
                }
            }
            Side::Left => {
                let (leaf, index) = self.descend_to_rank(root, rank, &mut path);
                // Top-down, the subtrees left of the path hold ever larger keys.
                for &PathElement {
                    node,
                    child_index,
                } in &path
                {
                    for i in 0..child_index {
                        let child = self.nodes.get(node).as_internal().child(i);
                        self.drain_subtree(child, &mut out);
                    }
                    self.nodes.get_mut(node).as_internal_mut().drop_front_children(child_index);
                }
                let leaf_node = self.nodes.get_mut(leaf).as_leaf_mut();
                let moved = leaf_node.take_head(index);
                leaf_node.set_prev(None);
                self.first_leaf = Some(leaf);
                self.push_entries(moved, &mut out);
            }
        }
        debug_assert_eq!(out.len(), moved_len);
        self.len -= moved_len;

        // The border child of each path node shrank; re-record its size bottom-up.
        for &PathElement {
            node,
            ..
        } in path.iter().rev()
        {
            let internal = self.nodes.get(node).as_internal();
            let border = match side {
                Side::Right => internal.child_count() - 1,
                Side::Left => 0,
            };
            let border_size = self.nodes.get(internal.child(border)).subtree_size();
            let internal = self.nodes.get_mut(node).as_internal_mut();
            internal.set_child_size(border, border_size);
            internal.update_size();
        }

        self.fix_border(side, leaf_separator);
        out
    }

    /// Frees the subtree at `handle`, moving its entries to `out` in key order.
    fn drain_subtree(&mut self, handle: Handle, out: &mut Vec<(K, V)>) {
        match self.nodes.take(handle) {
            Node::Leaf(mut leaf) => {
                let entries = leaf.take_all();
                self.push_entries(entries, out);
            }
            Node::Internal(internal) => {
                for child in internal.into_children() {
                    self.drain_subtree(child, out);
                }
            }
        }
    }

    /// Takes the values for `entries` out of the arena and appends the pairs to `out`.
    fn push_entries(&mut self, (keys, values): crate::raw::node::LeafEntries<K>, out: &mut Vec<(K, V)>) {
        for (key, value) in keys.into_iter().zip(values) {
            out.push((key, self.values.take(value)));
        }
    }

    /// Splits the tree at `key`: keeps the keys less than `key` and returns the rest as a new
    /// tree. See [`Self::split_off_at_rank`].
    pub(crate) fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        K: core::borrow::Borrow<Q>,
        Q: ?Sized + Ord,
    {
        match self.split_point(key, false) {
            Some((_, _, rank)) => self.split_off_at_rank(rank),
            None => Self::new(),
        }
    }

    /// Splits the tree at `rank`: keeps ranks `0..rank` and returns the rest as a new tree, in
    /// O(min(rank, len - rank) + log n). The smaller side is the one moved; if that is the
    /// left, the two trees trade places at the end.
    ///
    /// Everything that needs cloning is cloned first, so if `K::clone` panics the tree is
    /// unchanged.
    pub(crate) fn split_off_at_rank(&mut self, rank: usize) -> Self {
        let len = self.len;
        if rank >= len {
            return Self::new();
        }
        if rank == 0 {
            return core::mem::replace(self, Self::new());
        }
        if len - rank <= rank {
            let separators = self.clone_separators(rank, len - rank);
            let leaf_separator = self.clone_cut_separator(rank, Side::Right);
            let moved = self.cut(rank, Side::Right, leaf_separator);
            Self::build_from_sorted_with(moved, separators)
        } else {
            let separators = self.clone_separators(0, rank);
            let leaf_separator = self.clone_cut_separator(rank, Side::Left);
            let moved = self.cut(rank, Side::Left, leaf_separator);
            let left = Self::build_from_sorted_with(moved, separators);
            core::mem::replace(self, left)
        }
    }

    // ─── Joining ────────────────────────────────────────────────────────────────

    /// Moves every entry of `other` into `self`, emptying `other`. On equal keys `self` keeps
    /// its key and takes the value from `other`, as `BTreeMap::append` does.
    ///
    /// - Key ranges that don't overlap: the smaller tree is built packed into the larger one's
    ///   arenas and joined onto its edge, in O(min(n, m) + log(n + m)). Every clone happens
    ///   first, so if `K::clone` panics both trees are unchanged.
    /// - Overlapping, `other` much smaller: one insert per entry, O(m log(n + m)).
    /// - Overlapping, similar sizes: both are merged and rebuilt in O(n + m). If `K::clone`
    ///   panics during the rebuild, the entries are dropped and both trees are left empty.
    pub(crate) fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            core::mem::swap(self, other);
            return;
        }

        let (self_first, _) = self.first_key_value().expect("non-empty");
        let (self_last, _) = self.last_key_value().expect("non-empty");
        let (other_first, _) = other.first_key_value().expect("non-empty");
        let (other_last, _) = other.last_key_value().expect("non-empty");
        let side = if other_first > self_last {
            Some(Side::Right)
        } else if other_last < self_first {
            Some(Side::Left)
        } else {
            None
        };

        if let Some(side) = side {
            // The smaller tree is grafted onto the larger. With no equal keys, which one ends
            // up holding the result doesn't matter, but they trade places only after every
            // clone, so a panic leaves them where they were.
            let swap = other.len > self.len;
            let (host, guest, side) = if swap {
                (&*other, &*self, side.flip())
            } else {
                (&*self, &*other, side)
            };
            let clones = host.clone_for_join(guest, side);
            if swap {
                core::mem::swap(self, other);
            }
            self.join_disjoint(other, side, clones);
        } else {
            let (n, m) = (self.len, other.len);
            let log = (usize::BITS - (n + m).leading_zeros()) as usize;
            if m.saturating_mul(log) < n + m {
                for (key, value) in other.drain_to_vec() {
                    self.insert(key, value);
                }
            } else {
                let merged = merge_sorted(self.drain_to_vec(), other.drain_to_vec());
                *self = Self::build_from_sorted(merged);
            }
        }
    }

    /// Clones everything [`Self::join_disjoint`] needs to graft `other`, no larger than `self`
    /// and wholly on `side` of it, onto that edge.
    fn clone_for_join(&self, other: &Self, side: Side) -> JoinClones<K> {
        let (n, m) = (self.len, other.len);
        debug_assert!(m <= n);
        let (_, sub_height) = packed_shape(m);
        let height = self.height();
        debug_assert!(sub_height <= height, "a packed tree is no taller than any tree as large");

        let join_separator = match side {
            Side::Right => self.last_key_value(),
            Side::Left => other.last_key_value(),
        }
        .expect("non-empty")
        .0
        .clone();
        let separators = other.clone_separators(0, m);
        let leaf_separator = match (sub_height, height) {
            // Two leaves become one, or two halves under a new root.
            (0, 0) if n + m > MAX_KEYS => {
                let (left, right, left_len) = match side {
                    Side::Right => (self, other, n),
                    Side::Left => (other, self, m),
                };
                let last_of_left_half = usize::midpoint(n, m) - 1;
                let key = if last_of_left_half < left_len {
                    left.get_by_rank(last_of_left_half)
                } else {
                    right.get_by_rank(last_of_left_half - left_len)
                };
                Some(key.expect("rank in range").0.clone())
            }
            // A single leaf joins the bottom level next to the edge leaf.
            (0, _) if height > 0 => {
                let neighbor = match side {
                    Side::Right => self.last_leaf,
                    Side::Left => self.first_leaf,
                };
                self.clone_leaf_separator(m, neighbor.expect("non-empty"), side)
            }
            _ => None,
        };
        JoinClones {
            join_separator,
            separators,
            leaf_separator,
        }
    }

    /// Grafts `other`, no larger than `self` and wholly on `side` of it, onto that edge, given
    /// what [`Self::clone_for_join`] cloned for the two. Clones nothing.
    fn join_disjoint(&mut self, other: &mut Self, side: Side, clones: JoinClones<K>) {
        let subtree = self.build_packed_with(other.drain_to_vec(), clones.separators);
        self.join(subtree, side, clones.join_separator, clones.leaf_separator);
    }

    /// Attaches `subtree` (built in this tree's arenas, holding keys wholly on `side` of this
    /// tree's, and no taller than it) at that edge. `join_separator` is the max key of
    /// whichever of the two is on the left. Clones nothing.
    fn join(&mut self, subtree: Subtree, side: Side, join_separator: K, leaf_separator: Option<K>) {
        let height = self.height();
        let root = self.root.expect("`join()` - tree is non-empty");

        // Splice the leaf chains.
        let (left_last, right_first) = match side {
            Side::Right => {
                let last = self.last_leaf.replace(subtree.last_leaf).expect("non-empty");
                (last, subtree.first_leaf)
            }
            Side::Left => {
                let first = self.first_leaf.replace(subtree.first_leaf).expect("non-empty");
                (subtree.last_leaf, first)
            }
        };
        self.nodes.get_mut(left_last).as_leaf_mut().set_next(Some(right_first));
        self.nodes.get_mut(right_first).as_leaf_mut().set_prev(Some(left_last));
        self.len += subtree.len;

        if subtree.height == height {
            let (left, right) = match side {
                Side::Right => (root, subtree.root),
                Side::Left => (subtree.root, root),
            };
            self.join_equal_height(left, right, join_separator, leaf_separator);
            return;
        }

        // Walk the edge down to the node one level above the subtree's root.
        let mut path = Path::new();
        let mut node = root;
        for _ in 0..height - subtree.height - 1 {
            let internal = self.nodes.get(node).as_internal();
            let child_index = match side {
                Side::Right => internal.child_count() - 1,
                Side::Left => 0,
            };
            path.push(PathElement {
                node,
                child_index,
            });
            node = internal.child(child_index);
        }
        self.adjust_sizes_along_path(&path, |size| size + subtree.len);

        let parent = self.nodes.get_mut(node).as_internal_mut();
        let subtree_size = Size::from_usize(subtree.len);
        match side {
            Side::Right => parent.push_child(join_separator, subtree.root, subtree_size),
            Side::Left => parent.push_child_front(join_separator, subtree.root, subtree_size),
        }
        parent.update_size();
        if parent.key_count() > MAX_KEYS {
            let (median, right) = parent.split();
            let (left_size, right_size) = (parent.size(), right.size());
            let right_handle = self.nodes.alloc(Node::Internal(right));
            self.propagate_split(&mut path, median, right_handle, left_size, right_size);
        }

        // Only the subtree's root (and, for a one-leaf subtree, that leaf) can be underfull.
        self.fix_border(side, leaf_separator);
    }

    /// Joins two trees of equal height, rooted at `left` and `right`, by merging their roots;
    /// if that overflows, the merged root is split in two under a new root. Both halves of a
    /// split hold at least half a node, and every node below the roots was already valid.
    fn join_equal_height(&mut self, left: Handle, right: Handle, separator: K, leaf_separator: Option<K>) {
        let right_node = self.nodes.take(right);
        let new_right = match (self.nodes.get_mut(left), right_node) {
            (Node::Internal(left_node), Node::Internal(right_node)) => {
                debug_assert!(leaf_separator.is_none());
                left_node.merge_with_right(separator, right_node);
                if left_node.key_count() > MAX_KEYS {
                    let (median, new_right) = left_node.split();
                    Some((median, Node::Internal(new_right)))
                } else {
                    None
                }
            }
            (Node::Leaf(left_leaf), Node::Leaf(right_leaf)) => {
                drop(separator);
                left_leaf.merge_with_right(right_leaf);
                if left_leaf.key_count() > MAX_KEYS {
                    let mut new_right = left_leaf.split_half();
                    let median = leaf_separator.expect("`join_equal_height()` - split needs a separator");
                    debug_assert!(left_leaf.last_key() == Some(&median), "`join_equal_height()` - wrong separator");
                    new_right.set_prev(Some(left));
                    Some((median, Node::Leaf(new_right)))
                } else {
                    debug_assert!(leaf_separator.is_none());
                    None
                }
            }
            _ => unreachable!("trees of equal height"),
        };
        // If the roots were leaves, they were the whole chain: `right` was the last leaf and is
        // gone, and the merged leaf ends the chain (`merge_with_right` cleared its `next`).
        if self.last_leaf == Some(right) {
            self.last_leaf = Some(left);
        }

        let Some((median, new_right)) = new_right else {
            self.root = Some(left);
            return;
        };
        let new_right_is_leaf = matches!(new_right, Node::Leaf(_));
        let right_size = new_right.subtree_size();
        let new_right = self.nodes.alloc(new_right);
        if new_right_is_leaf {
            // `split_half` left the right half's `next` unset; it is the last leaf.
            self.nodes.get_mut(left).as_leaf_mut().set_next(Some(new_right));
            self.last_leaf = Some(new_right);
        }
        let left_size = self.nodes.get(left).subtree_size();
        let mut root = InternalNode::new();
        root.set_first_child(left, left_size);
        root.push_child(median, new_right, right_size);
        root.update_size();
        self.root = Some(self.nodes.alloc(Node::Internal(root)));
    }
}

/// The clones [`RawOSBTreeMap::join_disjoint`] needs, made before either tree changes.
struct JoinClones<K> {
    /// The max key of whichever tree is on the left.
    join_separator: K,
    /// The separators for the grafted tree, built packed.
    separators: Vec<K>,
    /// The separator for a leaf-level split or top-up, if one will be needed.
    leaf_separator: Option<K>,
}

/// Merges two sorted, duplicate-free runs. On equal keys the key from `a` is kept with the
/// value from `b`, as `BTreeMap::append` does.
fn merge_sorted<K: Ord, V>(a: Vec<(K, V)>, b: Vec<(K, V)>) -> Vec<(K, V)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut a = a.into_iter().peekable();
    let mut b = b.into_iter().peekable();
    loop {
        let order = match (a.peek(), b.peek()) {
            (Some((ka, _)), Some((kb, _))) => ka.cmp(kb),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => break,
        };
        match order {
            Ordering::Less => out.extend(a.next()),
            Ordering::Greater => out.extend(b.next()),
            Ordering::Equal => {
                let ((key, _), (_, value)) = (a.next().expect("peeked"), b.next().expect("peeked"));
                out.push((key, value));
            }
        }
    }
    out
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use proptest::prelude::*;

    /// How a test tree is built: packed from sorted input, or by inserts in a scrambled order
    /// followed by some removals, which leaves nodes anywhere between half and completely full.
    #[derive(Clone, Copy, Debug)]
    enum Build {
        Packed,
        Scrambled(u64),
    }

    fn build(keys: &[u32], how: Build, offset: u32) -> RawOSBTreeMap<u32, u32> {
        match how {
            Build::Packed => RawOSBTreeMap::from_sorted_iter(keys.iter().map(|&k| (k, k + offset))),
            Build::Scrambled(seed) => {
                let mut order: Vec<u32> = keys.to_vec();
                let mut x = seed | 1;
                for i in (1..order.len()).rev() {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    order.swap(i, (x >> 33) as usize % (i + 1));
                }
                let mut tree = RawOSBTreeMap::new();
                // Insert some extra keys and remove them again, to thin out the nodes.
                for &k in &order {
                    tree.insert(k, k + offset);
                    tree.insert(k ^ 0x8000_0000, 0);
                }
                for &k in &order {
                    tree.remove(&(k ^ 0x8000_0000));
                }
                tree
            }
        }
    }

    fn contents(tree: &RawOSBTreeMap<u32, u32>) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(tree.len());
        let mut pos = tree.first_position();
        while let Some((leaf, i)) = pos {
            let (k, v) = tree.entry_at((leaf, i));
            out.push((*k, *v));
            pos = tree.position_at_or_after(leaf, i + 1);
        }
        out
    }

    /// Checks `tree` against `model`, then keeps mutating it and checks again.
    fn check(tree: &mut RawOSBTreeMap<u32, u32>, model: &BTreeMap<u32, u32>) {
        tree.validate_invariants();
        assert_eq!(contents(tree), model.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>());
        for (rank, (&k, _)) in model.iter().enumerate().step_by(7) {
            assert_eq!(tree.rank_of(&k), Some(rank));
        }
        let mut model = model.clone();
        for k in model.keys().copied().step_by(3).collect::<Vec<_>>() {
            assert_eq!(tree.remove(&k), model.remove(&k));
        }
        for k in 0..20u32 {
            assert_eq!(tree.insert(k * 7_919 + 3, k), model.insert(k * 7_919 + 3, k));
        }
        tree.validate_invariants();
        assert_eq!(tree.len(), model.len());
    }

    fn split_and_check(keys: &[u32], how: Build, rank: usize) {
        let mut tree = build(keys, how, 0);
        let mut model: BTreeMap<u32, u32> = keys.iter().map(|&k| (k, k)).collect();
        let mut right = tree.split_off_at_rank(rank);
        let model_right = match model.keys().nth(rank).copied() {
            Some(at) => model.split_off(&at),
            None => BTreeMap::new(),
        };
        check(&mut tree, &model);
        check(&mut right, &model_right);
    }

    #[test]
    fn split_off_at_every_rank_of_small_trees() {
        for n in 0..=300u32 {
            let keys: Vec<u32> = (0..n).map(|k| k * 2).collect();
            for rank in 0..=n as usize {
                split_and_check(&keys, Build::Packed, rank);
                split_and_check(&keys, Build::Scrambled(u64::from(n) * 31 + rank as u64), rank);
            }
        }
    }

    #[test]
    fn split_off_by_key_matches_btreemap() {
        let keys: Vec<u32> = (0..2_000).map(|k| k * 2).collect();
        for at in [0, 1, 2, 3, 999, 1_000, 1_001, 2_000, 3_997, 3_998, 3_999, 4_000, u32::MAX] {
            let mut tree = build(&keys, Build::Scrambled(u64::from(at)), 0);
            let mut model: BTreeMap<u32, u32> = keys.iter().map(|&k| (k, k)).collect();
            let mut right = tree.split_off(&at);
            let model_right = model.split_off(&at);
            check(&mut tree, &model);
            check(&mut right, &model_right);
        }
    }

    fn append_and_check(a: &[u32], b: &[u32], how_a: Build, how_b: Build) {
        // Values tell the two sides apart, to check which wins on equal keys.
        let mut tree_a = build(a, how_a, 0);
        let mut tree_b = build(b, how_b, 1_000_000);
        let mut model_a: BTreeMap<u32, u32> = a.iter().map(|&k| (k, k)).collect();
        let mut model_b: BTreeMap<u32, u32> = b.iter().map(|&k| (k, k + 1_000_000)).collect();
        tree_a.append(&mut tree_b);
        model_a.append(&mut model_b);
        assert!(tree_b.is_empty());
        tree_b.validate_invariants();
        check(&mut tree_a, &model_a);
    }

    #[test]
    fn append_disjoint_at_every_size_pair() {
        let sizes = [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 30, 31, 64, 120, 128, 129, 200, 255, 256, 257, 600, 2_100];
        for &n in &sizes {
            for &m in &sizes {
                let low: Vec<u32> = (0..n as u32).collect();
                let high: Vec<u32> = (0..m as u32).map(|k| k + 10_000).collect();
                for (how_a, how_b) in [
                    (Build::Packed, Build::Packed),
                    (Build::Scrambled(n as u64), Build::Packed),
                    (Build::Packed, Build::Scrambled(m as u64)),
                    (Build::Scrambled(3), Build::Scrambled(5)),
                ] {
                    // `other` after `self`, and `other` before `self`.
                    append_and_check(&low, &high, how_a, how_b);
                    append_and_check(&high, &low, how_a, how_b);
                }
            }
        }
    }

    #[test]
    fn append_overlapping_both_strategies() {
        let big: Vec<u32> = (0..3_000).map(|k| k * 2).collect();
        // Few keys into many: inserts. Interleaved halves: merge and rebuild.
        let few: Vec<u32> = (0..40).map(|k| k * 101).collect();
        let interleaved: Vec<u32> = (0..3_000).map(|k| k * 2 + 1).collect();
        let same: Vec<u32> = big.clone();
        for other in [&few, &interleaved, &same] {
            append_and_check(&big, other, Build::Scrambled(1), Build::Packed);
            append_and_check(other, &big, Build::Packed, Build::Scrambled(2));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn split_off_matches_btreemap(
            keys in proptest::collection::btree_set(any::<u16>(), 0..3_000),
            seed in any::<u64>(),
            packed in any::<bool>(),
            at in any::<prop::sample::Index>(),
        ) {
            let keys: Vec<u32> = keys.into_iter().map(u32::from).collect();
            let how = if packed { Build::Packed } else { Build::Scrambled(seed) };
            split_and_check(&keys, how, at.index(keys.len() + 1));
        }

        #[test]
        fn append_matches_btreemap(
            a in proptest::collection::btree_set(any::<u16>(), 0..2_000),
            b in proptest::collection::btree_set(any::<u16>(), 0..2_000),
            shift in prop_oneof![Just(0u32), Just(70_000u32)],
            b_first in any::<bool>(),
            seeds in (any::<u64>(), any::<u64>()),
            packed in (any::<bool>(), any::<bool>()),
        ) {
            // A shift makes the key ranges disjoint; `b_first` puts `b` below `a`.
            let a: Vec<u32> = a.into_iter().map(|k| u32::from(k) + if b_first { shift } else { 0 }).collect();
            let b: Vec<u32> = b.into_iter().map(|k| u32::from(k) + if b_first { 0 } else { shift }).collect();
            let how = |packed, seed| if packed { Build::Packed } else { Build::Scrambled(seed) };
            append_and_check(&a, &b, how(packed.0, seeds.0), how(packed.1, seeds.1));
        }
    }
}

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
use core::ops::RangeBounds;

use smallvec::SmallVec;

use super::{Path, PathElement, RawOSBTreeMap};
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

    /// Returns `left` for the left edge and `right` for the right edge.
    fn pick<T>(self, left: T, right: T) -> T {
        match self {
            Side::Left => left,
            Side::Right => right,
        }
    }

    /// Which of a pair to treat as the border when fixing it up: the left one if its `left_len`
    /// is below `min`, otherwise the right one. If both are short they fit in one node and merge.
    const fn short(left_len: usize, min: usize) -> Self {
        if left_len < min {
            Side::Left
        } else {
            Side::Right
        }
    }

    /// The index of the edge child among `count` children.
    fn edge(self, count: usize) -> usize {
        self.pick(0, count - 1)
    }
}

/// A packed subtree built by [`RawOSBTreeMap::build_packed`]: its nodes and values live in the
/// host tree's arenas, but nothing in the host refers to it yet.
#[derive(Clone, Copy)]
pub(super) struct Subtree {
    root: Handle,
    height: usize,
    len: usize,
    first_leaf: Handle,
    last_leaf: Handle,
}

/// What [`RawOSBTreeMap::fix_border`] does with the border leaf and its neighbour.
enum LeafFix {
    /// They fit in one leaf, so they merge.
    Merge,
    /// The border leaf takes this many entries from its neighbour.
    TopUp(usize),
    /// Both are full enough.
    Keep,
}

impl LeafFix {
    /// The fix for a border leaf of `border_len` entries next to one of `neighbor_len`. Both the
    /// fix-up and the prediction of the separator it needs call this, so they can't disagree.
    const fn new(border_len: usize, neighbor_len: usize) -> Self {
        if border_len + neighbor_len <= MAX_KEYS {
            LeafFix::Merge
        } else if border_len < MIN_LEAF_KEYS {
            LeafFix::TopUp(MIN_LEAF_KEYS - border_len)
        } else {
            LeafFix::Keep
        }
    }
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
fn packed_shape(len: usize) -> (usize, usize) {
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

/// A separator source for [`RawOSBTreeMap::build_packed`] that clones the last key of the leaf
/// left of each boundary.
pub(super) fn clone_last_key<K: Clone>(nodes: &Arena<Node<K>>, leaf: Handle, _: usize) -> K {
    nodes.get(leaf).as_leaf().last_key().expect("built leaves are non-empty").clone()
}

/// A separator source for [`RawOSBTreeMap::build_packed`] that hands out `separators`, as
/// returned by `clone_separators` for the same entries, without cloning.
fn hand_out<K>(separators: Vec<K>) -> impl FnMut(&Arena<Node<K>>, Handle, usize) -> K {
    // Boundaries are asked for level by level, not in order, so each is taken by index.
    let mut separators: Vec<Option<K>> = separators.into_iter().map(Some).collect();
    move |_, _, index| separators[index].take().expect("one separator per leaf boundary")
}

impl<K, V> RawOSBTreeMap<K, V> {
    /// Takes the entries of leaf `leaf` in `range` out of the tree's arenas and appends them to
    /// `out`, without copying the leaf.
    pub(super) fn move_entries(&mut self, leaf: Handle, range: impl RangeBounds<usize> + Clone, out: &mut Vec<(K, V)>) {
        let values = &mut self.values;
        for (key, value) in self.nodes.get_mut(leaf).as_leaf_mut().drain(range) {
            out.push((key, values.take(value)));
        }
    }

    /// Frees the subtree at `handle`, moving its entries to `out` in key order.
    fn drain_subtree(&mut self, handle: Handle, out: &mut Vec<(K, V)>) {
        if let Node::Internal(internal) = self.nodes.get(handle) {
            let children: SmallVec<[Handle; MAX_CHILDREN + 1]> =
                (0..internal.child_count()).map(|i| internal.child(i)).collect();
            for child in children {
                self.drain_subtree(child, out);
            }
        } else {
            self.move_entries(handle, .., out);
        }
        self.nodes.free(handle);
    }
}

impl<K: Clone + Ord, V> RawOSBTreeMap<K, V> {
    /// Builds `entries` (strictly increasing keys) into this tree's arenas as a packed,
    /// detached subtree, in O(n). Each level is split into as few nodes as fit, with entries
    /// (or children) spread evenly, so every node but the subtree's root is at least half full.
    ///
    /// `separator(nodes, leaf, i)` supplies the separator for leaf boundary `i`, whose left
    /// side ends with `leaf`. The subtree's leaves are linked to each other only.
    fn build_packed<I>(&mut self, entries: I, mut separator: impl FnMut(&Arena<Node<K>>, Handle, usize) -> K) -> Subtree
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut entries = entries.into_iter();
        let len = entries.len();
        debug_assert!(len > 0, "`build_packed()` - no entries");
        let leaf_count = packed_leaf_count(len);
        let (node_count, height) = packed_shape(len);
        self.nodes.reserve(node_count);
        self.values.reserve(len);

        // Each node on the level being built: (handle, subtree size, rightmost leaf, and that
        // leaf's index). The rightmost leaf ends at the boundary the node's separator stands for.
        let mut level: Vec<(Handle, usize, Handle, usize)> = Vec::with_capacity(leaf_count);
        let (mut first_leaf, mut last_leaf) = (None, None);
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
            let handle = Self::link_leaf(&mut self.nodes, &mut first_leaf, &mut last_leaf, leaf);
            level.push((handle, count, handle, index));
        }
        debug_assert!(entries.next().is_none(), "`build_packed()` - more entries than `len()`");

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
        }

        Subtree {
            root: level[0].0,
            height,
            len,
            first_leaf: first_leaf.expect("built at least one leaf"),
            last_leaf: last_leaf.expect("built at least one leaf"),
        }
    }

    /// Builds a whole tree from sorted `entries`, with separators from `separator` (see
    /// [`Self::build_packed`]).
    pub(super) fn build_tree<I>(entries: I, separator: impl FnMut(&Arena<Node<K>>, Handle, usize) -> K) -> Self
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
        let subtree = tree.build_packed(entries, separator);
        tree.root = Some(subtree.root);
        tree.first_leaf = Some(subtree.first_leaf);
        tree.last_leaf = Some(subtree.last_leaf);
        tree.len = subtree.len;
        tree
    }

    /// Clones the separators a packed tree would need for the `len > 0` entries starting at rank
    /// `start` of this tree, walking the leaf chain from `start` once.
    fn clone_separators(&self, start: usize, len: usize) -> Vec<K> {
        let (mut leaf, mut index) = self.locate_rank(start).expect("`clone_separators()` - start in range");
        let mut offset = 0;
        separator_positions(len)
            .map(|target| {
                let mut skip = target - offset;
                offset = target;
                loop {
                    let node = self.nodes.get(leaf).as_leaf();
                    let rest = node.key_count() - index;
                    if skip < rest {
                        index += skip;
                        return node.key(index).clone();
                    }
                    skip -= rest;
                    (leaf, index) = (node.next().expect("`clone_separators()` - range in tree"), 0);
                }
            })
            .collect()
    }

    /// The root, which must be a leaf.
    fn root_leaf(&self) -> &LeafNode<K> {
        self.nodes.get(self.root.expect("non-empty")).as_leaf()
    }

    /// Returns the height of this non-empty tree: 0 for a lone leaf.
    fn height(&self) -> usize {
        let mut height = 0;
        let mut node = self.root.expect("`height()` - tree is non-empty");
        while let Node::Internal(internal) = self.nodes.get(node) {
            node = internal.child(0);
            height += 1;
        }
        height
    }

    // ─── Border fix-up ──────────────────────────────────────────────────────────

    /// Predicts what [`Self::fix_leaf_pair`] will do to a border leaf of `border_len` entries on
    /// `side` of `neighbor` (unchanged by the fix-up above it). If it will move entries from the
    /// neighbour into the border leaf, returns a clone of the key that becomes their separator;
    /// that is the only clone a fix-up needs.
    fn clone_leaf_separator(border_len: usize, neighbor: &LeafNode<K>, side: Side) -> Option<K> {
        let neighbor_len = neighbor.key_count();
        let LeafFix::TopUp(count) = LeafFix::new(border_len, neighbor_len) else {
            return None;
        };
        // The new separator is the left leaf's last key afterwards: on the left edge the last key
        // handed over, on the right edge the last key the neighbour keeps.
        Some(neighbor.key(side.pick(count, neighbor_len - count) - 1).clone())
    }

    /// Collapses internal roots with a single child.
    fn fix_top(&mut self) {
        let mut root = self.root.expect("`fix_top()` - tree is non-empty");
        while let Node::Internal(internal) = self.nodes.get(root) {
            if internal.child_count() > 1 {
                break;
            }
            let child = internal.child(0);
            self.nodes.free(root);
            root = child;
        }
        self.root = Some(root);
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
        let mut node = self.root.expect("`fix_border()` - tree is non-empty");
        while let Node::Internal(parent) = self.nodes.get(node) {
            let count = parent.child_count();
            debug_assert!(count >= 2, "`fix_border()` - border parent has a single child");
            let left_index = side.pick(0, count - 2);
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
            let right_node = self.nodes.take(right).into_internal();
            let left_node = self.nodes.get_mut(left).as_internal_mut();
            left_node.merge_with_right(separator, right_node);
            let merged_size = left_node.size();
            self.nodes.get_mut(parent).as_internal_mut().set_child_size(index, merged_size);
            return left;
        }

        let border_count = side.pick(left_count, right_count);
        if border_count <= MIN_CHILDREN {
            // Rotate children through the parent: its separator moves down into the border child
            // and the donor's new max moves up.
            let count = MIN_CHILDREN + 1 - border_count;
            let (separator, moved) = match side {
                Side::Right => self.nodes.get_mut(left).as_internal_mut().take_back(count),
                Side::Left => self.nodes.get_mut(right).as_internal_mut().take_front(count),
            };
            let old = self.nodes.get_mut(parent).as_internal_mut().replace_key(index, separator);
            match side {
                Side::Right => self.nodes.get_mut(right).as_internal_mut().prepend(moved, old),
                Side::Left => self.nodes.get_mut(left).as_internal_mut().extend(old, moved),
            }
        }
        self.refresh_pair_sizes(parent, index);
        side.pick(left, right)
    }

    /// Fixes the leaf children `index` and `index + 1` of `parent`, one of which is on the
    /// border.
    fn fix_leaf_pair(&mut self, parent: Handle, index: usize, side: Side, separator: Option<K>) {
        let parent_node = self.nodes.get(parent).as_internal();
        let (left, right) = (parent_node.child(index), parent_node.child(index + 1));
        let left_len = self.nodes.get(left).as_leaf().key_count();
        let right_len = self.nodes.get(right).as_leaf().key_count();

        match LeafFix::new(side.pick(left_len, right_len), side.pick(right_len, left_len)) {
            LeafFix::Merge => {
                debug_assert!(separator.is_none(), "`fix_leaf_pair()` - merging needs no separator");
                let (_, right_handle, _) = self.nodes.get_mut(parent).as_internal_mut().remove_child(index);
                debug_assert_eq!(right_handle, right);
                let right_leaf = self.nodes.take(right).into_leaf();
                let left_leaf = self.nodes.get_mut(left).as_leaf_mut();
                left_leaf.merge_with_right(right_leaf);
                match left_leaf.next() {
                    Some(next) => self.nodes.get_mut(next).as_leaf_mut().set_prev(Some(left)),
                    None => self.last_leaf = Some(left),
                }
                let merged_size = Size::from_usize(left_len + right_len);
                self.nodes.get_mut(parent).as_internal_mut().set_child_size(index, merged_size);
            }
            LeafFix::TopUp(count) => {
                let separator = separator.expect("`fix_leaf_pair()` - topping up needs a separator");
                match side {
                    Side::Right => {
                        let moved = self.nodes.get_mut(left).as_leaf_mut().take_tail(left_len - count);
                        self.nodes.get_mut(right).as_leaf_mut().prepend(moved);
                    }
                    Side::Left => {
                        let moved = self.nodes.get_mut(right).as_leaf_mut().take_head(count);
                        self.nodes.get_mut(left).as_leaf_mut().extend(moved);
                    }
                }
                debug_assert!(
                    self.nodes.get(left).as_leaf().last_key() == Some(&separator),
                    "`fix_leaf_pair()` - wrong separator"
                );
                drop(self.nodes.get_mut(parent).as_internal_mut().replace_key(index, separator));
                self.refresh_pair_sizes(parent, index);
            }
            LeafFix::Keep => debug_assert!(separator.is_none(), "`fix_leaf_pair()` - separator was not needed"),
        }
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
        // The border leaf holds the last element kept (right edge) or the first (left edge).
        let (leaf, index) = self.locate_rank(side.pick(rank, rank - 1))?;
        let leaf = self.nodes.get(leaf).as_leaf();
        let (border_len, neighbor) = match side {
            Side::Right => (index + 1, leaf.prev()?),
            Side::Left => (leaf.key_count() - index, leaf.next()?),
        };
        Self::clone_leaf_separator(border_len, self.nodes.get(neighbor).as_leaf(), side)
    }

    /// Removes the elements at ranks `rank..len` (`Side::Right`) or `0..rank` (`Side::Left`) and
    /// returns them in key order, in O(elements removed + log n). Requires `0 < rank < len`, and
    /// `leaf_separator` from [`Self::clone_cut_separator`]. Clones nothing.
    fn cut(&mut self, rank: usize, side: Side, leaf_separator: Option<K>) -> Vec<(K, V)> {
        debug_assert!((1..self.len).contains(&rank), "`cut()` - rank out of range");
        let root = self.root.expect("`cut()` - tree is non-empty");
        let moved_len = side.pick(rank, self.len - rank);
        let mut out = Vec::with_capacity(moved_len);
        let mut path = Path::new();

        match side {
            Side::Right => {
                // Descend to the last element kept, so every node on the path keeps something.
                let (leaf, index) = self.descend_to_rank(root, rank - 1, &mut path);
                self.move_entries(leaf, index + 1.., &mut out);
                self.nodes.get_mut(leaf).as_leaf_mut().set_next(None);
                self.last_leaf = Some(leaf);
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
                for element in &mut path {
                    for i in 0..element.child_index {
                        let child = self.nodes.get(element.node).as_internal().child(i);
                        self.drain_subtree(child, &mut out);
                    }
                    self.nodes.get_mut(element.node).as_internal_mut().drop_front_children(element.child_index);
                    element.child_index = 0;
                }
                self.move_entries(leaf, ..index, &mut out);
                self.nodes.get_mut(leaf).as_leaf_mut().set_prev(None);
                self.first_leaf = Some(leaf);
            }
        }
        debug_assert_eq!(out.len(), moved_len);
        self.len -= moved_len;

        // The path now runs down the cut edge; each node's edge child shrank.
        self.refresh_sizes(&path);
        self.fix_border(side, leaf_separator);
        out
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
        let side = if len - rank <= rank {
            Side::Right
        } else {
            Side::Left
        };
        let separators = match side {
            Side::Right => self.clone_separators(rank, len - rank),
            Side::Left => self.clone_separators(0, rank),
        };
        let leaf_separator = self.clone_cut_separator(rank, side);
        let moved = Self::build_tree(self.cut(rank, side, leaf_separator), hand_out(separators));
        match side {
            Side::Right => moved,
            Side::Left => core::mem::replace(self, moved),
        }
    }

    // ─── Joining ────────────────────────────────────────────────────────────────

    /// Moves every entry of `other` into `self`, emptying `other`. On equal keys `self` keeps
    /// its key and takes the value from `other`, as `BTreeMap::append` does.
    ///
    /// - Key ranges that don't overlap: the smaller tree is built packed into the larger one's
    ///   arenas and joined onto its edge, in O(min(n, m) + log(n + m)) amortized (growing the
    ///   larger tree's arenas can copy them, as an insert can). Every clone happens
    ///   first, so if `K::clone` panics both trees are unchanged.
    /// - Overlapping, one much smaller: its entries are inserted into the other, one at a time,
    ///   in O(min(n, m) log(n + m)).
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
            let subtree = self.build_packed(other.drain_to_vec(), hand_out(clones.separators));
            self.join(subtree, side, clones.height, clones.join_separator, clones.leaf_separator);
        } else {
            let (n, m) = (self.len, other.len);
            let log = (usize::BITS - (n + m).leading_zeros()) as usize;
            if m.saturating_mul(log) < n + m {
                for (key, value) in other.drain_to_vec() {
                    self.insert(key, value);
                }
            } else if n.saturating_mul(log) < n + m {
                // A few entries of `self` into a large `other`: insert them into `other` instead,
                // keeping their keys but `other`'s values, and take the result.
                core::mem::swap(self, other);
                for (key, value) in other.drain_to_vec() {
                    drop(self.insert_key_keep_value(key, value));
                }
            } else {
                let merged = merge_sorted(self.drain_to_vec(), other.drain_to_vec());
                *self = Self::build_tree(merged, clone_last_key);
            }
        }
    }

    /// Clones everything needed to graft `other`, no larger than `self` and wholly on `side` of
    /// it, onto that edge.
    fn clone_for_join(&self, other: &Self, side: Side) -> JoinClones<K> {
        let (n, m) = (self.len, other.len);
        debug_assert!(m <= n);
        let (_, sub_height) = packed_shape(m);
        let height = self.height();
        debug_assert!(sub_height <= height, "a packed tree is no taller than any tree as large");

        let join_separator = side.pick(other, self).last_key_value().expect("non-empty").0.clone();
        let separators = other.clone_separators(0, m);
        let leaf_separator = match (sub_height, height) {
            // Two root leaves go under a new root and are fixed up as a pair; either may be short.
            (0, 0) => {
                let (left, right) = side.pick((other, self), (self, other));
                let (left, right) = (left.root_leaf(), right.root_leaf());
                let short = Side::short(left.key_count(), MIN_LEAF_KEYS);
                let (border, neighbor) = short.pick((left, right), (right, left));
                Self::clone_leaf_separator(border.key_count(), neighbor, short)
            }
            // A single leaf joins the bottom level next to the edge leaf.
            (0, _) => {
                let neighbor = side.pick(self.first_leaf, self.last_leaf).expect("non-empty");
                Self::clone_leaf_separator(m, self.nodes.get(neighbor).as_leaf(), side)
            }
            _ => None,
        };
        JoinClones {
            height,
            join_separator,
            separators,
            leaf_separator,
        }
    }

    /// Attaches `subtree` (built in this tree's arenas, holding keys wholly on `side` of this
    /// tree's, and no taller than it) at that edge. `height` is this tree's height, and
    /// `join_separator` the max key of whichever of the two is on the left. Clones nothing.
    fn join(&mut self, subtree: Subtree, side: Side, height: usize, join_separator: K, leaf_separator: Option<K>) {
        let root = self.root.expect("`join()` - tree is non-empty");

        // Splice the leaf chains.
        let (left_last, right_first) = match side {
            Side::Right => (self.last_leaf.replace(subtree.last_leaf).expect("non-empty"), subtree.first_leaf),
            Side::Left => (subtree.last_leaf, self.first_leaf.replace(subtree.first_leaf).expect("non-empty")),
        };
        self.nodes.get_mut(left_last).as_leaf_mut().set_next(Some(right_first));
        self.nodes.get_mut(right_first).as_leaf_mut().set_prev(Some(left_last));
        self.len += subtree.len;

        if subtree.height == height {
            let (left, right) = side.pick((subtree.root, root), (root, subtree.root));
            self.join_roots(left, right, join_separator, leaf_separator);
            return;
        }

        // Walk the edge down to the node one level above the subtree's root.
        let mut path = Path::new();
        let mut node = root;
        for _ in 0..height - subtree.height - 1 {
            let internal = self.nodes.get(node).as_internal();
            let child_index = side.edge(internal.child_count());
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

    /// Joins two trees of equal height rooted at `left` and `right` (already in this tree's
    /// arenas, leaf chains spliced). If the roots fit in one node they merge into it; otherwise
    /// a new root goes over both, and the pair is fixed up, topping up whichever old root is
    /// short. Neither way can a node outgrow its inline buffers.
    fn join_roots(&mut self, left: Handle, right: Handle, separator: K, leaf_separator: Option<K>) {
        self.root = Some(left);
        let leaves = matches!(self.nodes.get(left), Node::Leaf(_));
        let (left_fill, right_fill) = (self.nodes.get(left).fill(), self.nodes.get(right).fill());
        let (max, min) = if leaves {
            (MAX_KEYS, MIN_LEAF_KEYS)
        } else {
            (MAX_CHILDREN, MIN_CHILDREN + 1)
        };

        if left_fill + right_fill <= max {
            debug_assert!(leaf_separator.is_none(), "`join_roots()` - merging needs no separator");
            let right_node = self.nodes.take(right);
            if leaves {
                drop(separator);
                // The two leaves were the whole chain; the merged one ends it.
                self.nodes.get_mut(left).as_leaf_mut().merge_with_right(right_node.into_leaf());
                self.last_leaf = Some(left);
            } else {
                self.nodes.get_mut(left).as_internal_mut().merge_with_right(separator, right_node.into_internal());
            }
            return;
        }

        let (left_size, right_size) = (self.nodes.get(left).subtree_size(), self.nodes.get(right).subtree_size());
        // With an empty path, this puts a new root over `left` and `right`.
        self.propagate_split(&mut Path::new(), separator, right, left_size, right_size);
        let new_root = self.root.expect("just made");
        let short = Side::short(left_fill, min);
        if leaves {
            self.fix_leaf_pair(new_root, 0, short, leaf_separator);
        } else {
            debug_assert!(leaf_separator.is_none());
            self.fix_internal_pair(new_root, 0, short);
        }
    }
}

/// The clones grafting one tree onto another needs, made before either tree changes.
struct JoinClones<K> {
    /// The host tree's height.
    height: usize,
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
        assert!(!tree.nodes.iter().any(Node::spilled), "a node's buffers moved to the heap");
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
    fn split_off_of_an_empty_tree_is_empty() {
        let mut tree: RawOSBTreeMap<u32, u32> = RawOSBTreeMap::new();
        let right = tree.split_off(&5);
        assert!(tree.is_empty() && right.is_empty());
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

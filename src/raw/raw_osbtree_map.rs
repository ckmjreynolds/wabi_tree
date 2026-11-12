use core::borrow::Borrow;
use core::ops::{Bound, RangeBounds};

use alloc::boxed::Box;
use smallvec::SmallVec;

use super::arena::Arena;
use super::handle::Handle;
use super::node::{InternalNode, LeafNode, MAX_KEYS, MIN_LEAF_KEYS, Node, SearchResult};
use super::size::Size;

mod bulk;

/// How many sibling leaves [`RawOSBTreeMap::seek`] walks before it climbs the finger instead.
/// Every leaf except a root leaf holds at least `MIN_LEAF_KEYS` keys, so a target within
/// `HOP_LEAVES * MIN_LEAF_KEYS` positions of the current leaf is at most this many hops away.
const HOP_LEAVES: usize = 2;

/// The core B+Tree implementation backing `OSBTreeMap`.
pub(crate) struct RawOSBTreeMap<K, V> {
    /// Arena storing all tree nodes.
    nodes: Arena<Node<K>>,
    /// Arena storing all values (separate from nodes for cache efficiency).
    values: Arena<V>,
    /// Handle to the root node, if the tree is non-empty.
    root: Option<Handle>,
    /// Total number of key-value pairs in the tree.
    len: usize,
    /// Handle to the first (leftmost) leaf, for forward iteration.
    first_leaf: Option<Handle>,
    /// Handle to the last (rightmost) leaf, for backward iteration.
    last_leaf: Option<Handle>,
    /// The descent path recorded by the last [`Self::search_recording_path`], so an entry can
    /// insert or remove without walking down from the root again. Boxed so the tree grows by a
    /// pointer, not a whole `Path`; the box is allocated on first use and then reused.
    ///
    /// Only an entry that recorded the path may consume it, and only while it holds the tree's
    /// unique borrow. No entry method changes the tree's shape before consuming the entry, so
    /// the path can't go stale. Nothing else reads it, so it is never cloned or kept in sync.
    entry_path: Option<Box<Path>>,
}

/// Positions of the first and last elements of a range, plus its length.
/// Both positions are `None` (and `len` is 0) when the range is empty.
#[derive(Clone, Copy, Default)]
pub(crate) struct RangeCursor {
    pub(crate) front_leaf: Option<Handle>,
    pub(crate) front_index: usize,
    pub(crate) back_leaf: Option<Handle>,
    pub(crate) back_index: usize,
    pub(crate) len: usize,
    /// Rank of the first element of the range.
    pub(crate) start_rank: usize,
}

/// One level of a [`Finger`]: a node and the ranks `start_rank..end_rank` held in its subtree.
#[derive(Clone, Copy)]
struct FingerLevel {
    node: Handle,
    start_rank: usize,
    end_rank: usize,
}

/// The path from the root to the node an iterator's last long jump reached, with each level's
/// rank range. [`RawOSBTreeMap::seek`] climbs it only as far as a jump needs instead of
/// restarting at the root. A [`Span`] allocates its fingers on its first long jump, so plain
/// iteration never pays for them. Inner nodes have at least `MIN_CHILDREN` (64) children, so
/// eight levels cover any tree that fits in memory and the levels live inside that one
/// allocation.
///
/// A finger records where the last long jump landed, not the iterator's current position:
/// `next`, `next_back` and short jumps move the iterator without touching it, so it can be left
/// behind, even under a different parent. That is harmless. Each level's rank range describes
/// the tree, which cannot change while the iterator exists, so the levels stay a true path from
/// the root with correct ranges. `seek` climbs by rank range rather than from the current leaf,
/// so a finger left behind only makes the next long jump climb higher, and that cost was
/// already paid by the steps that left it behind.
type Finger = SmallVec<[FingerLevel; 8]>;

/// The two fingers of a [`Span`], one per end. `Span` holds them as `Option<Box<Fingers>>`, so a
/// span that never makes a long jump pays only for a null pointer.
#[derive(Default)]
struct Fingers {
    front: Finger,
    back: Finger,
}

/// The state shared by the borrowing iterators (`Iter`, `IterMut`, and the `Range` types that
/// wrap them): the positions of the front and back elements, how many elements are left, and
/// the rank bookkeeping that lets `nth` and `nth_back` jump by rank.
///
/// `remaining` is the only test for exhaustion: the positions are meaningful only while it is
/// non-zero, so the iterators never compare positions to find the end.
pub(crate) struct Span {
    front_leaf: Option<Handle>,
    front_index: usize,
    back_leaf: Option<Handle>,
    back_index: usize,
    remaining: usize,
    /// Rank of the front element. The back element's rank is `front_rank + remaining - 1`, so
    /// it needs no field of its own.
    front_rank: usize,
    /// Allocated on the first long jump; see [`RawOSBTreeMap::seek`].
    fingers: Option<Box<Fingers>>,
}

/// The fingers are only a cache that [`RawOSBTreeMap::seek`] rebuilds on demand, so a clone
/// starts without them rather than allocating copies.
impl Clone for Span {
    fn clone(&self) -> Self {
        Span {
            fingers: None,
            ..*self
        }
    }
}

impl Span {
    /// Creates a span over the elements `cursor` resolved.
    pub(crate) fn new(cursor: RangeCursor) -> Self {
        Span {
            front_leaf: cursor.front_leaf,
            front_index: cursor.front_index,
            back_leaf: cursor.back_leaf,
            back_index: cursor.back_index,
            remaining: cursor.len,
            front_rank: cursor.start_rank,
            fingers: None,
        }
    }

    /// Returns the number of elements left.
    pub(crate) const fn len(&self) -> usize {
        self.remaining
    }

    /// Returns the position of the front element and moves the front past it, or returns
    /// `None` if no element is left.
    ///
    /// # Safety
    /// If elements remain, `tree` must be valid and unmodified since the span was created.
    pub(crate) unsafe fn pop_front<K, V>(&mut self, tree: *const RawOSBTreeMap<K, V>) -> Option<(Handle, usize)> {
        if self.remaining == 0 {
            return None;
        }
        let pos = (self.front_leaf.expect("Span::pop_front: elements remain but the front is empty"), self.front_index);
        // SAFETY: elements remain, so the caller guarantees `tree` is valid.
        let leaf = unsafe { RawOSBTreeMap::node_ptr(tree, pos.0) }.as_leaf();
        self.remaining -= 1;
        self.front_rank += 1;
        self.front_index += 1;
        if self.front_index >= leaf.key_count() {
            self.front_leaf = leaf.next();
            self.front_index = 0;
        }
        Some(pos)
    }

    /// Returns the position of the back element and moves the back before it, or returns
    /// `None` if no element is left.
    ///
    /// # Safety
    /// As for [`Self::pop_front`].
    pub(crate) unsafe fn pop_back<K, V>(&mut self, tree: *const RawOSBTreeMap<K, V>) -> Option<(Handle, usize)> {
        if self.remaining == 0 {
            return None;
        }
        let pos = (self.back_leaf.expect("Span::pop_back: elements remain but the back is empty"), self.back_index);
        self.remaining -= 1;
        if self.back_index == 0 {
            // SAFETY: elements remained, so the caller guarantees `tree` is valid.
            self.back_leaf = unsafe { RawOSBTreeMap::node_ptr(tree, pos.0) }.as_leaf().prev();
            if let Some(prev) = self.back_leaf {
                // SAFETY: as above.
                let count = unsafe { RawOSBTreeMap::node_ptr(tree, prev) }.as_leaf().key_count();
                self.back_index = count.saturating_sub(1);
            }
        } else {
            self.back_index -= 1;
        }
        Some(pos)
    }

    /// Skips `n` elements at the front, then pops the front as [`Self::pop_front`] does.
    /// Returns `None`, leaving the span empty, if fewer than `n + 1` elements are left.
    ///
    /// # Safety
    /// As for [`Self::pop_front`].
    pub(crate) unsafe fn nth_front<K, V>(
        &mut self,
        tree: *const RawOSBTreeMap<K, V>,
        n: usize,
    ) -> Option<(Handle, usize)> {
        if n >= self.remaining {
            self.remaining = 0;
            return None;
        }
        if n > 0 {
            let current = self.front_leaf.expect("Span::nth_front: elements remain but the front is empty");
            let target = self.front_rank + n;
            let (index, rank, fingers) = (self.front_index, self.front_rank, &mut self.fingers);
            // SAFETY: n < remaining, so the caller guarantees `tree` is valid.
            let (leaf, index) =
                unsafe { RawOSBTreeMap::seek(tree, current, index, rank, target, fingers, |f| &mut f.front) };
            (self.front_leaf, self.front_index, self.front_rank) = (Some(leaf), index, target);
            self.remaining -= n;
        }
        // SAFETY: forwarded from the caller.
        unsafe { self.pop_front(tree) }
    }

    /// Skips `n` elements at the back, then pops the back, as [`Self::nth_front`] does at the
    /// front.
    ///
    /// # Safety
    /// As for [`Self::pop_front`].
    pub(crate) unsafe fn nth_back<K, V>(
        &mut self,
        tree: *const RawOSBTreeMap<K, V>,
        n: usize,
    ) -> Option<(Handle, usize)> {
        if n >= self.remaining {
            self.remaining = 0;
            return None;
        }
        if n > 0 {
            let current = self.back_leaf.expect("Span::nth_back: elements remain but the back is empty");
            let rank = self.front_rank + self.remaining - 1;
            let (index, fingers) = (self.back_index, &mut self.fingers);
            // SAFETY: n < remaining, so the caller guarantees `tree` is valid.
            let (leaf, index) =
                unsafe { RawOSBTreeMap::seek(tree, current, index, rank, rank - n, fingers, |f| &mut f.back) };
            (self.back_leaf, self.back_index) = (Some(leaf), index);
            self.remaining -= n;
        }
        // SAFETY: forwarded from the caller.
        unsafe { self.pop_back(tree) }
    }
}

/// Path element for tracking traversal during mutations.
struct PathElement {
    /// Handle to the node at this level.
    node: Handle,
    /// Index of the child we descended into.
    child_index: usize,
}

/// Type alias for a path through the tree (stack of path elements).
type Path = SmallVec<[PathElement; 16]>;

impl<K, V> RawOSBTreeMap<K, V> {
    /// Creates a new, empty tree.
    pub(crate) const fn new() -> Self {
        Self {
            nodes: Arena::new(),
            values: Arena::new(),
            root: None,
            len: 0,
            first_leaf: None,
            last_leaf: None,
            entry_path: None,
        }
    }

    /// Creates a new tree with the specified capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::with_arena_capacities(capacity.div_ceil(MAX_KEYS), capacity)
    }

    /// Creates an empty tree whose arenas have room for `nodes` nodes and `values` values.
    fn with_arena_capacities(nodes: usize, values: usize) -> Self {
        Self {
            nodes: Arena::with_capacity(nodes),
            values: Arena::with_capacity(values),
            root: None,
            len: 0,
            first_leaf: None,
            last_leaf: None,
            entry_path: None,
        }
    }

    /// Allocates `leaf` and links it after `last_leaf`, making it the new last leaf. Returns its
    /// handle. Used when building a tree's leaves left to right.
    fn push_leaf(&mut self, leaf: LeafNode<K>) -> Handle {
        Self::link_leaf(&mut self.nodes, &mut self.first_leaf, &mut self.last_leaf, leaf)
    }

    /// Allocates `leaf` in `nodes` and links it after `last`, updating the chain ends `first`
    /// and `last`. Returns its handle.
    fn link_leaf(
        nodes: &mut Arena<Node<K>>,
        first: &mut Option<Handle>,
        last: &mut Option<Handle>,
        mut leaf: LeafNode<K>,
    ) -> Handle {
        leaf.set_prev(*last);
        let handle = nodes.alloc(Node::Leaf(leaf));
        match *last {
            Some(prev) => nodes.get_mut(prev).as_leaf_mut().set_next(Some(handle)),
            None => *first = Some(handle),
        }
        *last = Some(handle);
        handle
    }

    /// Returns the number of key-value pairs in the tree.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the tree contains no elements.
    pub(crate) const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the tree.
    pub(crate) fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Clears all elements from the tree.
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.values.clear();
        self.root = None;
        self.len = 0;
        self.first_leaf = None;
        self.last_leaf = None;
    }

    /// Drains all key-value pairs from the tree by walking the leaf chain.
    /// This is O(n) as it avoids rebalancing, unlike repeated `pop_first`/`pop_last`.
    pub(crate) fn drain_to_vec(&mut self) -> alloc::vec::Vec<(K, V)> {
        let len = self.len;
        if len == 0 {
            return alloc::vec::Vec::new();
        }

        let mut result = alloc::vec::Vec::with_capacity(len);
        let mut current_leaf = self.first_leaf;

        while let Some(leaf_handle) = current_leaf {
            current_leaf = self.nodes.get(leaf_handle).as_leaf().next();
            self.move_entries(leaf_handle, .., &mut result);
        }

        // Every value has been taken, so the tree can be reset wholesale.
        self.clear();

        result
    }

    /// Returns the handle of the first leaf, if any.
    #[cfg(test)]
    pub(crate) fn first_leaf(&self) -> Option<Handle> {
        self.first_leaf
    }

    /// Returns the handle of the last leaf, if any.
    #[cfg(test)]
    pub(crate) fn last_leaf(&self) -> Option<Handle> {
        self.last_leaf
    }

    /// Returns the position of the first element, if any.
    pub(crate) fn first_position(&self) -> Option<(Handle, usize)> {
        let first_leaf = self.first_leaf?;
        (self.nodes.get(first_leaf).as_leaf().key_count() > 0).then_some((first_leaf, 0))
    }

    /// Returns the position of the last element, if any.
    pub(crate) fn last_position(&self) -> Option<(Handle, usize)> {
        let last_leaf = self.last_leaf?;
        let count = self.nodes.get(last_leaf).as_leaf().key_count();
        count.checked_sub(1).map(|idx| (last_leaf, idx))
    }

    /// Returns a cursor over every element: the first leaf from index 0 through the last position.
    pub(crate) fn full_cursor(&self) -> RangeCursor {
        let (back_leaf, back_index) = self.last_position().unzip();
        RangeCursor {
            front_leaf: self.first_leaf,
            front_index: 0,
            back_leaf,
            back_index: back_index.unwrap_or(0),
            len: self.len,
            start_rank: 0,
        }
    }

    /// Returns the handle of the root node, if any.
    #[cfg(test)]
    pub(crate) fn root(&self) -> Option<Handle> {
        self.root
    }

    /// Returns a reference to a node by handle.
    pub(crate) fn node(&self, handle: Handle) -> &Node<K> {
        self.nodes.get(handle)
    }

    /// Returns a reference to a node by handle from a raw pointer.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, allocated `RawOSBTreeMap<K, V>`.
    pub(crate) unsafe fn node_ptr<'a>(ptr: *const Self, handle: Handle) -> &'a Node<K> {
        // SAFETY: We only access the `nodes` field through addr_of, avoiding aliasing with
        // the `values` field.
        unsafe { Arena::get_ptr(core::ptr::addr_of!((*ptr).nodes), handle) }
    }

    /// Returns a mutable reference to a node by handle.
    #[cfg(test)]
    pub(crate) fn node_mut(&mut self, handle: Handle) -> &mut Node<K> {
        self.nodes.get_mut(handle)
    }

    /// Returns a reference to a value by handle.
    pub(crate) fn value(&self, handle: Handle) -> &V {
        self.values.get(handle)
    }

    /// Returns a mutable reference to a value by handle.
    pub(crate) fn value_mut(&mut self, handle: Handle) -> &mut V {
        self.values.get_mut(handle)
    }

    /// Returns a mutable reference to a value by handle from a raw pointer.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, allocated `RawOSBTreeMap<K, V>`.
    /// - The caller must ensure no other mutable references to the values arena exist.
    /// - The caller must have logical exclusive access to the value at `handle`.
    pub(crate) unsafe fn value_mut_ptr<'a>(ptr: *mut Self, handle: Handle) -> &'a mut V {
        // SAFETY: We only access the `values` field, avoiding aliasing with the `nodes` field,
        // and `get_mut_ptr` borrows only the one slot, so values handed out earlier stay valid.
        unsafe { Arena::get_mut_ptr(core::ptr::addr_of_mut!((*ptr).values), handle) }
    }

    /// Returns the position of the element with rank `target`, starting from the position
    /// `(leaf, index)`, whose rank is `rank`. Used by the iterators' `nth` and `nth_back`.
    ///
    /// A target in the current leaf, or within `HOP_LEAVES` leaves of it, is reached through the
    /// leaf chain. For a longer jump, `finger` is climbed to the lowest node whose rank range
    /// contains `target` (no node reads), and the search descends from there, leaving `finger`
    /// on the new path. A long jump reads O(1 + log d) nodes, where `d` is the distance from the
    /// previous long jump's target (the steps in between were each paid for by `next` or a
    /// short jump), so a whole pass over an iterator stays O(n) however it is split into jumps.
    /// The finger is not updated by those steps, so it may be behind `(leaf, index)`; see
    /// [`Finger`] for why that is safe.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, allocated `RawOSBTreeMap<K, V>`.
    /// - `(leaf, index)` must be a valid position in it, with rank `rank`.
    /// - `target` must be less than the tree's length, and the finger that `which` picks from
    ///   `fingers` must be empty or have been built by earlier calls on the same, unmodified tree.
    ///
    /// `fingers` is allocated only on a long jump, so a span that makes only short jumps never
    /// pays for it.
    unsafe fn seek(
        ptr: *const Self,
        mut leaf: Handle,
        index: usize,
        rank: usize,
        target: usize,
        fingers: &mut Option<Box<Fingers>>,
        which: fn(&mut Fingers) -> &mut Finger,
    ) -> (Handle, usize) {
        // SAFETY (all `node_ptr` calls and field reads below): the caller guarantees `ptr` is valid.
        let leaf_at = |handle| unsafe { Self::node_ptr(ptr, handle) }.as_leaf();

        // Short jump: walk the leaf chain, giving up once the target is too far for the hops left.
        if target >= rank {
            let mut start = rank - index; // rank of the first key in `leaf`
            for hops in 0..=HOP_LEAVES {
                let node = leaf_at(leaf);
                let end = start + node.key_count();
                if target < end {
                    return (leaf, target - start);
                }
                if target - end >= (HOP_LEAVES - hops) * MIN_LEAF_KEYS {
                    break;
                }
                // `end <= target < len`, so this is not the last leaf.
                let next = node.next().expect("seek: target is past the last leaf");
                (leaf, start) = (next, end);
            }
        } else {
            let mut start = rank - index; // rank of the first key in `leaf`
            for hops in 0..=HOP_LEAVES {
                if target >= start {
                    return (leaf, target - start);
                }
                // `start - target - 1` keys lie between `target` and `leaf`, mirroring `target - end` above.
                if start - target > (HOP_LEAVES - hops) * MIN_LEAF_KEYS {
                    break;
                }
                // `target < start`, so this is not the first leaf.
                let prev = leaf_at(leaf).prev().expect("seek: target is before the first leaf");
                (leaf, start) = (prev, start - leaf_at(prev).key_count());
            }
        }

        // Long jump: climb the finger to the lowest node containing `target`, then descend.
        let finger = which(fingers.get_or_insert_default());
        while finger.last().is_some_and(|level| !(level.start_rank..level.end_rank).contains(&target)) {
            finger.pop();
        }
        let mut level = if let Some(&level) = finger.last() {
            level
        } else {
            let root = unsafe { core::ptr::addr_of!((*ptr).root).read() };
            let len = unsafe { core::ptr::addr_of!((*ptr).len).read() };
            let level = FingerLevel {
                node: root.expect("seek: tree is empty"),
                start_rank: 0,
                end_rank: len,
            };
            finger.push(level);
            level
        };
        loop {
            match unsafe { Self::node_ptr(ptr, level.node) } {
                Node::Leaf(_) => return (level.node, target - level.start_rank),
                Node::Internal(internal) => {
                    let (child, before) = child_for_rank(internal, target - level.start_rank)
                        .expect("seek: tree size invariant violated");
                    let start_rank = level.start_rank + before;
                    level = FingerLevel {
                        node: internal.child(child),
                        start_rank,
                        end_rank: start_rank + internal.child_size(child).to_usize(),
                    };
                    finger.push(level);
                }
            }
        }
    }

    /// Takes a value from the arena.
    #[cfg(test)]
    pub(crate) fn take_value(&mut self, handle: Handle) -> V {
        self.values.take(handle)
    }
}

/// Clones only the live entries, so the clone is O(len) and compact even if the original has
/// many free arena slots from earlier removals. The clone keeps the original's capacity.
impl<K: Clone, V: Clone> Clone for RawOSBTreeMap<K, V> {
    fn clone(&self) -> Self {
        /// Clones the subtree at `handle` into `out`, linking each cloned leaf after
        /// `out.last_leaf` (leaves are visited left to right). Returns the new handle.
        fn clone_node<K: Clone, V: Clone>(
            src: &RawOSBTreeMap<K, V>,
            out: &mut RawOSBTreeMap<K, V>,
            handle: Handle,
        ) -> Handle {
            match src.nodes.get(handle) {
                Node::Leaf(leaf) => {
                    let mut new_leaf = LeafNode::new();
                    for i in 0..leaf.key_count() {
                        let value_handle = out.values.alloc(src.values.get(leaf.value(i)).clone());
                        new_leaf.push(leaf.key(i).clone(), value_handle);
                    }
                    out.push_leaf(new_leaf)
                }
                Node::Internal(internal) => {
                    let mut new_internal = InternalNode::new();
                    let first = clone_node(src, out, internal.child(0));
                    new_internal.set_first_child(first, internal.child_size(0));
                    for i in 0..internal.key_count() {
                        let child = clone_node(src, out, internal.child(i + 1));
                        new_internal.push_child(internal.key(i).clone(), child, internal.child_size(i + 1));
                    }
                    new_internal.set_size(internal.size());
                    out.nodes.alloc(Node::Internal(new_internal))
                }
            }
        }

        let mut out = Self::with_arena_capacities(self.nodes.capacity(), self.values.capacity());
        out.len = self.len;
        if let Some(root) = self.root {
            out.root = Some(clone_node(self, &mut out, root));
        }
        out
    }
}

impl<K: Clone + Ord, V> RawOSBTreeMap<K, V> {
    /// Builds a tree from `entries`, whose keys must be strictly increasing, in O(n).
    ///
    /// Each level is split into as few nodes as fit, with entries (or children) spread evenly
    /// across them, so every non-root node is at least half full without any fix-up pass.
    ///
    /// If `K::clone` panics while a separator is being built, nothing leaks or drops twice:
    /// every node and value lives in the arenas from the moment it is created, and dropping
    /// the partly built tree drops each of them exactly once.
    pub(crate) fn from_sorted_iter<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        I::IntoIter: ExactSizeIterator,
    {
        Self::build_tree(entries, bulk::clone_last_key)
    }

    /// Descends from `root` to a leaf and returns the leaf's handle. At each internal node,
    /// `pick` receives the node's handle and the node, and returns the child to descend into.
    #[inline]
    fn descend(&self, root: Handle, mut pick: impl FnMut(Handle, &InternalNode<K>) -> usize) -> Handle {
        let mut current = root;
        while let Node::Internal(internal) = self.nodes.get(current) {
            current = internal.child(pick(current, internal));
        }
        current
    }

    /// Descends from the root to the leaf whose key range covers `key`.
    #[inline]
    fn find_leaf<Q>(&self, key: &Q) -> Option<(Handle, &LeafNode<K>)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let leaf = self.descend(self.root?, |_, internal| internal.search_child(key));
        Some((leaf, self.nodes.get(leaf).as_leaf()))
    }

    /// Like [`Self::find_leaf`], but also returns the number of elements stored in
    /// leaves to the left of the returned leaf.
    fn find_leaf_with_rank<Q>(&self, key: &Q) -> Option<(Handle, &LeafNode<K>, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let mut rank = 0;
        let leaf = self.descend(self.root?, |_, internal| {
            let child_idx = internal.search_child(key);
            rank += size_before(internal, child_idx);
            child_idx
        });
        Some((leaf, self.nodes.get(leaf).as_leaf(), rank))
    }

    /// Descends from `root` to a leaf, recording the path taken in `path` (which the
    /// caller owns, to avoid moving the inline buffer). `pick` chooses the child to
    /// descend into at each internal node.
    fn descend_with_path(
        &self,
        root: Handle,
        path: &mut Path,
        mut pick: impl FnMut(&InternalNode<K>) -> usize,
    ) -> Handle {
        self.descend(root, |node, internal| {
            let child_index = pick(internal);
            path.push(PathElement {
                node,
                child_index,
            });
            child_index
        })
    }

    /// Returns the number of elements in leaves to the left of the leaf a descent along `path`
    /// reached.
    fn rank_along_path(&self, path: &Path) -> usize {
        path.iter().map(|elem| size_before(self.nodes.get(elem.node).as_internal(), elem.child_index)).sum()
    }

    /// Searches for a key and returns the leaf handle and index if found.
    pub(crate) fn search<Q>(&self, key: &Q) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (leaf_handle, leaf) = self.find_leaf(key)?;
        match leaf.search(key) {
            SearchResult::Found(idx) => Some((leaf_handle, idx)),
            SearchResult::NotFound(_) => None,
        }
    }

    /// Searches for `key` like [`Self::search`], recording the descent path for a following
    /// [`Self::insert_at_recorded`] or [`Self::remove_at_recorded`].
    ///
    /// Returns `Ok((leaf, index))` if the key is present, `Err(Some((leaf, index)))` with the
    /// position it would be inserted at if not, and `Err(None)` if the tree is empty.
    #[allow(clippy::type_complexity)]
    pub(crate) fn search_recording_path<Q>(&mut self, key: &Q) -> Result<(Handle, usize), Option<(Handle, usize)>>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let root = self.root.ok_or(None)?;
        let mut path = self.entry_path.take().unwrap_or_default();
        path.clear();
        let leaf_handle = self.descend_with_path(root, &mut path, |internal| internal.search_child(key));
        self.entry_path = Some(path);
        match self.nodes.get(leaf_handle).as_leaf().search(key) {
            SearchResult::Found(idx) => Ok((leaf_handle, idx)),
            SearchResult::NotFound(idx) => Err(Some((leaf_handle, idx))),
        }
    }

    /// Returns the key-value pair at a position.
    fn entry_at(&self, (leaf_handle, idx): (Handle, usize)) -> (&K, &V) {
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        (leaf.key(idx), self.values.get(leaf.value(idx)))
    }

    /// Returns a reference to the value corresponding to the key.
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.search(key).map(|pos| self.entry_at(pos).1)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    pub(crate) fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (leaf_handle, idx) = self.search(key)?;
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        let value_handle = leaf.value(idx);
        Some(self.values.get_mut(value_handle))
    }

    /// Returns the key-value pair corresponding to the key.
    pub(crate) fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.search(key).map(|pos| self.entry_at(pos))
    }

    /// Returns true if the tree contains the specified key.
    pub(crate) fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.search(key).is_some()
    }

    /// Returns the first key-value pair in the tree.
    pub(crate) fn first_key_value(&self) -> Option<(&K, &V)> {
        self.first_position().map(|pos| self.entry_at(pos))
    }

    /// Returns the last key-value pair in the tree.
    pub(crate) fn last_key_value(&self) -> Option<(&K, &V)> {
        self.last_position().map(|pos| self.entry_at(pos))
    }

    /// Inserts a key-value pair into the tree.
    /// Returns the old value if the key was already present.
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_with_handle(key, value).1
    }

    /// Inserts a key-value pair into the tree, returning the handle of the stored value
    /// and the old value if the key was already present. Value handles are stable across
    /// node splits, so the handle stays valid until the entry is removed.
    pub(crate) fn insert_with_handle(&mut self, key: K, value: V) -> (Handle, Option<V>) {
        let (value_handle, _, old_value) = self.insert_impl(key, value);
        (value_handle, old_value)
    }

    /// Inserts a key-value pair into the tree, returning the position `(leaf, index)` the key
    /// now occupies and the old value if the key was already present. The position stays
    /// valid until the tree is next modified structurally.
    pub(crate) fn insert_with_position(&mut self, key: K, value: V) -> ((Handle, usize), Option<V>) {
        let (_, position, old_value) = self.insert_impl(key, value);
        (position, old_value)
    }

    /// Inserts a key-value pair into the tree, returning the rank the key now has and the old
    /// value if the key was already present.
    ///
    /// This mirrors [`Self::insert_impl`] rather than sharing it, so plain inserts don't pay for
    /// the rank.
    pub(crate) fn insert_with_rank(&mut self, key: K, value: V) -> (usize, Option<V>) {
        let Some(root) = self.root else {
            return (0, self.insert(key, value));
        };
        let mut path = Path::new();
        let leaf_handle = self.descend_with_path(root, &mut path, |internal| internal.search_child(&key));
        let found = self.nodes.get(leaf_handle).as_leaf().search(&key);
        let (SearchResult::Found(idx) | SearchResult::NotFound(idx)) = found;
        // Taken before `insert_at`, whose split pops `path`. A split moves the key to another
        // leaf but not to another rank.
        let rank = self.rank_along_path(&path) + idx;
        match found {
            SearchResult::Found(_) => {
                let value_handle = self.nodes.get(leaf_handle).as_leaf().value(idx);
                (rank, Some(core::mem::replace(self.values.get_mut(value_handle), value)))
            }
            SearchResult::NotFound(_) => {
                self.insert_at(leaf_handle, idx, key, value, &mut path);
                (rank, None)
            }
        }
    }

    /// Inserts `key` and `value`. If an equal key is present, the stored key stays and only the
    /// value is replaced.
    ///
    /// Returns the handle of the stored value, the position `(leaf, index)` of the key, and the
    /// old value, if any.
    fn insert_impl(&mut self, key: K, value: V) -> (Handle, (Handle, usize), Option<V>) {
        let Some(root) = self.root else {
            let value_handle = self.values.alloc(value);
            let mut leaf = LeafNode::new();
            leaf.push(key, value_handle);
            let leaf_handle = self.nodes.alloc(Node::Leaf(leaf));
            self.root = Some(leaf_handle);
            self.first_leaf = Some(leaf_handle);
            self.last_leaf = Some(leaf_handle);
            self.len = 1;
            return (value_handle, (leaf_handle, 0), None);
        };

        let mut path = Path::new();
        let current = self.descend_with_path(root, &mut path, |internal| internal.search_child(&key));

        // Insert into leaf
        let leaf = self.nodes.get(current).as_leaf();
        match leaf.search(&key) {
            SearchResult::Found(idx) => {
                let value_handle = leaf.value(idx);
                // Key exists, replace value in-place to avoid alloc/free churn
                let old_value = core::mem::replace(self.values.get_mut(value_handle), value);
                (value_handle, (current, idx), Some(old_value))
            }
            SearchResult::NotFound(idx) => {
                let (value_handle, position) = self.insert_at(current, idx, key, value, &mut path);
                (value_handle, position, None)
            }
        }
    }

    /// Inserts a new key at `idx` in the leaf reached via `path`, splitting as needed. The key
    /// must be absent and belong at that position. Returns the handle of the stored value and
    /// the position `(leaf, index)` the key ends up at.
    fn insert_at(
        &mut self,
        leaf_handle: Handle,
        idx: usize,
        key: K,
        value: V,
        path: &mut Path,
    ) -> (Handle, (Handle, usize)) {
        let value_handle = self.values.alloc(value);
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        leaf.insert(idx, key, value_handle);
        self.len += 1;

        // Count the new element before splitting: a split only clones keys, so if
        // `K::clone` panics the tree is left consistent, with one oversized leaf.
        let needs_split = leaf.key_count() > MAX_KEYS;
        self.adjust_sizes_along_path(path, |size| size + 1);
        let mut position = (leaf_handle, idx);
        if needs_split {
            let (right_handle, left_count) = self.split_leaf_and_propagate(leaf_handle, path);
            if idx >= left_count {
                position = (right_handle, idx - left_count);
            }
        }
        (value_handle, position)
    }

    /// Inserts `key` at the position a [`Self::search_recording_path`] for it returned as
    /// `Err(Some((leaf_handle, idx)))`, using the path that search recorded. The tree must not
    /// have been modified since.
    pub(crate) fn insert_at_recorded(
        &mut self,
        leaf_handle: Handle,
        idx: usize,
        key: K,
        value: V,
    ) -> (Handle, (Handle, usize)) {
        let mut path = self.entry_path.take().expect("search_recording_path recorded a path");
        let inserted = self.insert_at(leaf_handle, idx, key, value, &mut path);
        self.entry_path = Some(path);
        inserted
    }

    /// Removes the element at the position a [`Self::search_recording_path`] returned as
    /// `Ok((leaf_handle, idx))`, using the path that search recorded. The tree must not have
    /// been modified since.
    pub(crate) fn remove_at_recorded(&mut self, leaf_handle: Handle, idx: usize) -> (K, V) {
        let mut path = self.entry_path.take().expect("search_recording_path recorded a path");
        let removed = self.remove_at(leaf_handle, idx, &mut path);
        self.entry_path = Some(path);
        removed
    }

    /// Removes the element at `(leaf_handle, idx)`, a position this tree returned and that no
    /// structural change has invalidated. The descent path is found again from the stored key,
    /// so the caller needs neither a recorded path nor a copy of the key.
    ///
    /// # Panics
    ///
    /// Panics if searching for the stored key doesn't lead back to its own leaf, which takes a
    /// key whose `Ord` is inconsistent. Removing along the wrong path would corrupt the sizes.
    pub(crate) fn remove_at_position(&mut self, leaf_handle: Handle, idx: usize) -> (K, V) {
        let root = self.root.expect("`remove_at_position()` - a position implies a non-empty tree");
        let mut path = Path::new();
        let key = self.nodes.get(leaf_handle).as_leaf().key(idx);
        let found = self.descend_with_path(root, &mut path, |internal| internal.search_child(key));
        assert_eq!(found, leaf_handle, "`remove_at_position()` - entry must exist");
        self.remove_at(leaf_handle, idx, &mut path)
    }

    /// Inserts `key` with `value`, unless an equal key is present: then `key` takes the stored
    /// key's place, the stored value stays, and the old key is returned.
    pub(crate) fn replace_key(&mut self, key: K, value: V) -> Option<K> {
        let Some(root) = self.root else {
            self.insert(key, value);
            return None;
        };
        let mut path = Path::new();
        let leaf_handle = self.descend_with_path(root, &mut path, |internal| internal.search_child(&key));
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        match leaf.search(&key) {
            // An equal key compares the same as the old one, so separators stay valid.
            SearchResult::Found(idx) => Some(leaf.replace_key(idx, key)),
            SearchResult::NotFound(idx) => {
                self.insert_at(leaf_handle, idx, key, value, &mut path);
                None
            }
        }
    }

    /// Splits a leaf and propagates splits up the tree as needed. Returns the handle of the new
    /// right leaf and the number of keys left in `leaf_handle`.
    fn split_leaf_and_propagate(&mut self, leaf_handle: Handle, path: &mut Path) -> (Handle, usize) {
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        let (separator, mut right_leaf) = leaf.split();

        // Get the sizes
        let left_size = Size::from_usize(leaf.key_count());
        let right_size = Size::from_usize(right_leaf.key_count());

        // Set up leaf links
        let old_next = leaf.next();
        right_leaf.set_prev(Some(leaf_handle));
        right_leaf.set_next(old_next);

        // Allocate the right leaf
        let right_handle = self.nodes.alloc(Node::Leaf(right_leaf));

        // Fix up the links
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        leaf.set_next(Some(right_handle));

        if let Some(old_next) = old_next {
            self.nodes.get_mut(old_next).as_leaf_mut().set_prev(Some(right_handle));
        }

        // Update last_leaf if needed
        if self.last_leaf == Some(leaf_handle) {
            self.last_leaf = Some(right_handle);
        }

        // Propagate the split up
        self.propagate_split(path, separator, right_handle, left_size, right_size);
        (right_handle, left_size.to_usize())
    }

    /// Propagates a split up the tree. Sizes along `path` must already count the inserted
    /// element; each node that gains a child here recomputes its sizes from its children.
    fn propagate_split(
        &mut self,
        path: &mut Path,
        mut separator: K,
        mut new_child: Handle,
        mut left_size: Size,
        mut right_size: Size,
    ) {
        while let Some(elem) = path.pop() {
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();

            // Update the size of the child we descended into
            parent.set_child_size(elem.child_index, left_size);

            // Insert the new child
            parent.insert_child(elem.child_index, separator, new_child, right_size);
            parent.update_size();

            // Check if parent needs to split
            if parent.key_count() <= MAX_KEYS {
                // No more splits needed; the remaining ancestors were already counted.
                return;
            }

            // Split this internal node
            let (median, right_internal) = parent.split();
            left_size = parent.size();
            right_size = right_internal.size();

            let right_handle = self.nodes.alloc(Node::Internal(right_internal));

            separator = median;
            new_child = right_handle;
        }

        // Need a new root
        let old_root = self.root.unwrap();
        let old_root_size = self.nodes.get(old_root).subtree_size();

        let mut new_root = InternalNode::new();
        new_root.set_first_child(old_root, old_root_size);
        new_root.push_child(separator, new_child, right_size);
        new_root.update_size();

        let new_root_handle = self.nodes.alloc(Node::Internal(new_root));
        self.root = Some(new_root_handle);
    }

    /// Applies `adjust` to the total size of every node on `path` and to the size it
    /// records for the child it descended into. Used when an insertion or removal
    /// changes the element count by one without restructuring those nodes.
    fn adjust_sizes_along_path(&mut self, path: &Path, adjust: impl Fn(usize) -> usize) {
        for elem in path {
            let node = self.nodes.get_mut(elem.node).as_internal_mut();
            node.set_size(Size::from_usize(adjust(node.size().to_usize())));
            let child_size = adjust(node.child_size(elem.child_index).to_usize());
            node.set_child_size(elem.child_index, Size::from_usize(child_size));
        }
    }

    /// Recomputes sizes bottom-up along `path`: each node's recorded size for the child
    /// it descended into, then the node's own total size.
    fn refresh_sizes(&mut self, path: &Path) {
        for elem in path.iter().rev() {
            let child = self.nodes.get(elem.node).as_internal().child(elem.child_index);
            let child_size = self.nodes.get(child).subtree_size();
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            parent.set_child_size(elem.child_index, child_size);
            parent.update_size();
        }
    }

    /// Searches for `key` like [`Self::search`], recording the descent in `path`.
    fn search_with_path<Q>(&self, key: &Q, path: &mut Path) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let leaf_handle = self.descend_with_path(self.root?, path, |internal| internal.search_child(key));
        match self.nodes.get(leaf_handle).as_leaf().search(key) {
            SearchResult::Found(idx) => Some((leaf_handle, idx)),
            SearchResult::NotFound(_) => None,
        }
    }

    /// Removes a key from the tree and returns the value.
    pub(crate) fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the tree and returns the key-value pair.
    pub(crate) fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let mut path = Path::new();
        let (leaf_handle, idx) = self.search_with_path(key, &mut path)?;
        Some(self.remove_at(leaf_handle, idx, &mut path))
    }

    /// Removes a key from the tree and returns its rank before removal and the key-value pair.
    pub(crate) fn remove_entry_with_rank<Q>(&mut self, key: &Q) -> Option<(usize, K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let mut path = Path::new();
        let (leaf_handle, idx) = self.search_with_path(key, &mut path)?;
        let rank = self.rank_along_path(&path) + idx;
        let (k, v) = self.remove_at(leaf_handle, idx, &mut path);
        Some((rank, k, v))
    }

    /// Removes and returns the key-value pair at `rank`, or `None` if `rank` is out of bounds.
    pub(crate) fn remove_by_rank(&mut self, rank: usize) -> Option<(K, V)> {
        if rank >= self.len {
            return None;
        }
        self.remove_by_rank_if(rank, |_, _| true).ok()
    }

    /// Removes and returns the key-value pair at `rank` if `pred` accepts it, in one descent.
    /// If `pred` rejects it, returns its position instead. `rank` must be in bounds.
    #[inline]
    pub(crate) fn remove_by_rank_if(
        &mut self,
        rank: usize,
        pred: impl FnOnce(&K, &mut V) -> bool,
    ) -> Result<(K, V), (Handle, usize)> {
        let root = self.root.expect("`remove_by_rank_if()` - rank in range");
        let mut path = Path::new();
        let (leaf_handle, idx) = self.descend_to_rank(root, rank, &mut path);
        let (key, value) = self.key_value_mut(leaf_handle, idx);
        if !pred(key, value) {
            return Err((leaf_handle, idx));
        }
        Ok(self.remove_at(leaf_handle, idx, &mut path))
    }

    /// Returns the key at a position and a mutable reference to its value.
    #[inline]
    pub(crate) fn key_value_mut(&mut self, leaf_handle: Handle, idx: usize) -> (&K, &mut V) {
        // `nodes` and `values` are disjoint fields, so both can be borrowed at once.
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        (leaf.key(idx), self.values.get_mut(leaf.value(idx)))
    }

    /// Returns the position of the element after the one at `(leaf_handle, idx)`, if any.
    pub(crate) fn position_after(&self, leaf_handle: Handle, idx: usize) -> Option<(Handle, usize)> {
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        if idx + 1 < leaf.key_count() {
            Some((leaf_handle, idx + 1))
        } else {
            leaf.next().map(|next| (next, 0))
        }
    }

    /// Removes and returns the first key-value pair.
    pub(crate) fn pop_first(&mut self) -> Option<(K, V)> {
        let root = self.root?;
        let mut path = Path::new();
        let leaf_handle = self.descend_with_path(root, &mut path, |_| 0);
        Some(self.remove_at(leaf_handle, 0, &mut path))
    }

    /// Removes and returns the last key-value pair.
    pub(crate) fn pop_last(&mut self) -> Option<(K, V)> {
        let root = self.root?;
        let mut path = Path::new();
        let leaf_handle = self.descend_with_path(root, &mut path, |internal| internal.child_count() - 1);
        let last_idx = self.nodes.get(leaf_handle).as_leaf().key_count() - 1;
        Some(self.remove_at(leaf_handle, last_idx, &mut path))
    }

    /// Removes the element at `idx` in the leaf reached via `path`, rebalancing as needed.
    fn remove_at(&mut self, leaf_handle: Handle, idx: usize, path: &mut Path) -> (K, V) {
        let (removed_key, value_handle) = self.nodes.get_mut(leaf_handle).as_leaf_mut().remove(idx);
        let removed_value = self.values.take(value_handle);
        self.len -= 1;

        if self.len == 0 {
            self.clear();
            return (removed_key, removed_value);
        }

        // Uncount the element before anything that clones a key, so a panicking `K::clone`
        // leaves sizes consistent. Rebalancing only moves elements between siblings under one
        // parent, so every ancestor's size is final from here on.
        self.adjust_sizes_along_path(path, |size| size - 1);
        if !self.nodes.get(leaf_handle).as_leaf().is_underfull() || path.is_empty() {
            // A stale separator is still a valid bound, so it's fine if this clone panics.
            self.update_separators_after_remove(leaf_handle, idx, path);
        } else {
            self.rebalance(leaf_handle, path);
        }

        (removed_key, removed_value)
    }

    /// Tightens the separator bounding `leaf_handle` after removing the element at
    /// `removed_idx`, which is only needed when the leaf's last key was removed. A separator
    /// may stay stale (above its left subtree's max key), so this keeps separators exact where
    /// that is cheap rather than upholding an invariant.
    fn update_separators_after_remove(&mut self, leaf_handle: Handle, removed_idx: usize, path: &Path) {
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        if removed_idx != leaf.key_count() {
            return;
        }
        let Some(last_key) = leaf.last_key() else {
            return;
        };
        // The nearest ancestor reached through a child that isn't its last holds the separator.
        // On an all-rightmost path there is none, and so nothing to clone.
        let Some(elem) =
            path.iter().rev().find(|elem| elem.child_index < self.nodes.get(elem.node).as_internal().key_count())
        else {
            return;
        };
        let last_key = last_key.clone();
        self.nodes.get_mut(elem.node).as_internal_mut().set_key(elem.child_index, last_key);
    }

    /// Rebalances a node, leaf or internal, that a removal left underfull: borrows from a
    /// sibling that can spare an entry, or else merges with one.
    fn rebalance(&mut self, node_handle: Handle, path: &mut Path) {
        let &PathElement {
            node: parent_handle,
            child_index: child_idx,
        } = path.last().unwrap();
        let parent = self.nodes.get(parent_handle).as_internal();
        let is_leaf = matches!(self.nodes.get(node_handle), Node::Leaf(_));
        let left = (child_idx > 0).then(|| parent.child(child_idx - 1));
        let right = (child_idx + 1 < parent.child_count()).then(|| parent.child(child_idx + 1));
        let can_lend = |sibling: &Handle| self.nodes.get(*sibling).can_lend();

        if let Some(left) = left.filter(can_lend) {
            if is_leaf {
                self.borrow_from_left_leaf(node_handle, left, parent_handle, child_idx);
            } else {
                self.borrow_from_left_internal(node_handle, left, parent_handle, child_idx);
            }
        } else if let Some(right) = right.filter(can_lend) {
            if is_leaf {
                self.borrow_from_right_leaf(node_handle, right, parent_handle, child_idx);
            } else {
                self.borrow_from_right_internal(node_handle, right, parent_handle, child_idx);
            }
        } else {
            // Neither sibling can lend, so merge, preferring the left one.
            let (left, right, separator_idx) = match left {
                Some(left) => (left, node_handle, child_idx - 1),
                None => (node_handle, right.expect("a non-root node has a sibling"), child_idx),
            };
            if is_leaf {
                self.merge_leaves(left, right, path, separator_idx);
            } else {
                self.merge_internals(left, right, path, separator_idx);
            }
        }
    }

    /// Borrows a key-value from the left leaf sibling.
    fn borrow_from_left_leaf(
        &mut self,
        leaf_handle: Handle,
        left_handle: Handle,
        parent_handle: Handle,
        child_idx: usize,
    ) {
        // Clone the left sibling's new last key before popping, so a panicking `K::clone`
        // leaves both leaves intact.
        let left = self.nodes.get_mut(left_handle).as_leaf_mut();
        let left_new_last_key = left.key(left.key_count() - 2).clone();

        // Pop from left sibling
        let (key, value) = left.pop().unwrap();
        let left_new_size = left.key_count();

        // Push to front of current leaf
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        leaf.push_front(key, value);
        let current_new_size = leaf.key_count();

        // Update parent separator and child sizes
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_key(child_idx - 1, left_new_last_key);
        parent.set_child_size(child_idx - 1, Size::from_usize(left_new_size));
        parent.set_child_size(child_idx, Size::from_usize(current_new_size));
    }

    /// Borrows a key-value from the right leaf sibling.
    fn borrow_from_right_leaf(
        &mut self,
        leaf_handle: Handle,
        right_handle: Handle,
        parent_handle: Handle,
        child_idx: usize,
    ) {
        // Clone the borrowed key before popping, so a panicking `K::clone` leaves both leaves
        // intact.
        let right = self.nodes.get_mut(right_handle).as_leaf_mut();
        let separator = right.key(0).clone();

        // Pop from front of right sibling
        let (key, value) = right.pop_front().unwrap();
        let right_new_size = right.key_count();

        // Push to end of current leaf
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        leaf.push(key, value);
        let current_new_size = leaf.key_count();

        // Update parent separator and child sizes.
        // The borrowed key becomes the new max key of the current node,
        // so it's the correct separator for child_idx.
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_key(child_idx, separator);
        parent.set_child_size(child_idx, Size::from_usize(current_new_size));
        parent.set_child_size(child_idx + 1, Size::from_usize(right_new_size));
    }

    /// Moves every entry of the leaf `right_handle` into its left neighbour `left_handle`,
    /// frees it and unlinks it from the leaf chain. The caller fixes up the parent.
    fn absorb_right_leaf(&mut self, left_handle: Handle, right_handle: Handle) {
        let right = self.nodes.take(right_handle).into_leaf();
        let left = self.nodes.get_mut(left_handle).as_leaf_mut();
        left.merge_with_right(right);
        match left.next() {
            Some(next) => self.nodes.get_mut(next).as_leaf_mut().set_prev(Some(left_handle)),
            None => self.last_leaf = Some(left_handle),
        }
    }

    /// Merges two leaf nodes.
    fn merge_leaves(&mut self, left_handle: Handle, right_handle: Handle, path: &mut Path, separator_idx: usize) {
        self.absorb_right_leaf(left_handle, right_handle);
        // The right leaf's node was taken by the merge, so only its slot in the parent is left.
        let parent_handle = path.last().unwrap().node;
        self.nodes.get_mut(parent_handle).as_internal_mut().remove_child(separator_idx);
        self.propagate_merge(path, separator_idx);
    }

    /// Finishes a merge whose right node and separator are already out of the parent: fixes
    /// the merged child's size there and propagates rebalancing up.
    fn propagate_merge(&mut self, path: &mut Path, separator_idx: usize) {
        let parent_elem = path.pop().unwrap();
        let parent_handle = parent_elem.node;

        // Update the merged child's size in the parent (the left child absorbed the right)
        let parent = self.nodes.get(parent_handle).as_internal();
        let merged_child_handle = parent.child(separator_idx);
        let merged_child_size = self.nodes.get(merged_child_handle).subtree_size();
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        // The parent's own total is unchanged: the two children became one.
        parent.set_child_size(separator_idx, merged_child_size);

        // Check if parent needs rebalancing
        if path.is_empty() {
            // This is the root
            if parent.child_count() == 1 {
                // Root has only one child, make that child the new root
                let new_root = parent.child(0);
                self.nodes.free(parent_handle);
                self.root = Some(new_root);
            }
            return;
        }

        if !parent.is_underfull() {
            return;
        }

        // Need to rebalance internal node
        self.rebalance(parent_handle, path);
    }

    /// Borrows from left internal sibling.
    fn borrow_from_left_internal(
        &mut self,
        node_handle: Handle,
        left_handle: Handle,
        parent_handle: Handle,
        child_idx: usize,
    ) {
        // Pop from left sibling
        let left = self.nodes.get_mut(left_handle).as_internal_mut();
        let (left_key, left_child, left_child_size) = left.pop_child().unwrap();
        left.update_size();
        let left_new_size = left.size();

        // The popped key rotates up into the parent, and the parent's separator rotates down.
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        let parent_sep = parent.replace_key(child_idx - 1, left_key);
        parent.set_child_size(child_idx - 1, left_new_size);

        // Push to front of current node with parent separator
        let node = self.nodes.get_mut(node_handle).as_internal_mut();
        node.push_child_front(parent_sep, left_child, left_child_size);
        node.update_size();
        let node_new_size = node.size();

        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_child_size(child_idx, node_new_size);
    }

    /// Borrows from right internal sibling.
    fn borrow_from_right_internal(
        &mut self,
        node_handle: Handle,
        right_handle: Handle,
        parent_handle: Handle,
        child_idx: usize,
    ) {
        // Pop from front of right sibling
        let right = self.nodes.get_mut(right_handle).as_internal_mut();
        let (right_key, right_child, right_child_size) = right.pop_child_front().unwrap();
        right.update_size();
        let right_new_size = right.size();

        // The popped key rotates up into the parent, and the parent's separator rotates down.
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        let parent_sep = parent.replace_key(child_idx, right_key);
        parent.set_child_size(child_idx + 1, right_new_size);

        // Push to end of current node with parent separator
        let node = self.nodes.get_mut(node_handle).as_internal_mut();
        node.push_child(parent_sep, right_child, right_child_size);
        node.update_size();
        let node_new_size = node.size();

        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_child_size(child_idx, node_new_size);
    }

    /// Merges two internal nodes.
    fn merge_internals(&mut self, left_handle: Handle, right_handle: Handle, path: &mut Path, separator_idx: usize) {
        let parent_handle = path.last().unwrap().node;

        // The parent's separator comes down between the two halves, so it moves, not clones.
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        let (separator, removed, _) = parent.remove_child(separator_idx);
        debug_assert_eq!(removed, right_handle);

        // Take the right node
        let right = self.nodes.take(right_handle).into_internal();

        // Merge into left
        let left = self.nodes.get_mut(left_handle).as_internal_mut();
        left.merge_with_right(separator, right);

        self.propagate_merge(path, separator_idx);
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

    /// Returns the position of the element with the given rank.
    fn locate_rank(&self, rank: usize) -> Option<(Handle, usize)> {
        if rank >= self.len {
            return None;
        }

        let mut remaining = rank;
        let leaf = self.descend(self.root?, |_, internal| {
            // A real panic: in release builds a violated invariant would otherwise loop forever.
            let Some((child, before)) = child_for_rank(internal, remaining) else {
                panic!(
                    "locate_rank: tree size invariant violated - rank {} not found in children (node size: {})",
                    rank,
                    internal.size().to_usize()
                );
            };
            remaining -= before;
            child
        });
        Some((leaf, remaining))
    }

    /// Gets an element by its rank (0-indexed position in sorted order).
    pub(crate) fn get_by_rank(&self, rank: usize) -> Option<(&K, &V)> {
        self.locate_rank(rank).map(|pos| self.entry_at(pos))
    }

    /// Gets a mutable element by its rank.
    pub(crate) fn get_by_rank_mut(&mut self, rank: usize) -> Option<(&K, &mut V)> {
        let (leaf_handle, idx) = self.locate_rank(rank)?;
        // `nodes` and `values` are disjoint fields, so both can be borrowed at once.
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        Some((leaf.key(idx), self.values.get_mut(leaf.value(idx))))
    }

    /// Returns the rank (0-indexed position) of a key.
    pub(crate) fn rank_of<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (_, leaf, rank) = self.find_leaf_with_rank(key)?;
        match leaf.search(key) {
            SearchResult::Found(idx) => Some(rank + idx),
            SearchResult::NotFound(_) => None,
        }
    }

    /// Returns the rank of a key together with the stored key-value pair.
    pub(crate) fn get_with_rank<Q>(&self, key: &Q) -> Option<(usize, &K, &V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (_, leaf, rank) = self.find_leaf_with_rank(key)?;
        let SearchResult::Found(idx) = leaf.search(key) else {
            return None;
        };
        Some((rank + idx, leaf.key(idx), self.values.get(leaf.value(idx))))
    }

    /// Returns the rank of a key together with the stored key and a mutable reference to its value.
    pub(crate) fn get_with_rank_mut<Q>(&mut self, key: &Q) -> Option<(usize, &K, &mut V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (leaf_handle, _, rank) = self.find_leaf_with_rank(key)?;
        // `nodes` and `values` are disjoint fields, so both can be borrowed at once.
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        let SearchResult::Found(idx) = leaf.search(key) else {
            return None;
        };
        Some((rank + idx, leaf.key(idx), self.values.get_mut(leaf.value(idx))))
    }

    /// Locates the split point for `key`: the leaf it falls in, how many of that leaf's
    /// keys come before the split, and how many elements in the whole tree come before
    /// it. Keys equal to `key` come before the split only if `include_equal` is true.
    fn split_point<Q>(&self, key: &Q, include_equal: bool) -> Option<(Handle, usize, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (leaf_handle, leaf, leaf_rank) = self.find_leaf_with_rank(key)?;
        let idx = match leaf.search(key) {
            SearchResult::Found(idx) if include_equal => idx + 1,
            SearchResult::Found(idx) | SearchResult::NotFound(idx) => idx,
        };
        Some((leaf_handle, idx, leaf_rank + idx))
    }

    /// Returns the first position at or after `idx` in `leaf_handle`, following the leaf chain.
    fn position_at_or_after(&self, leaf_handle: Handle, idx: usize) -> Option<(Handle, usize)> {
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        if idx < leaf.key_count() {
            return Some((leaf_handle, idx));
        }
        let next = leaf.next()?;
        (self.nodes.get(next).as_leaf().key_count() > 0).then_some((next, 0))
    }

    /// Returns the last position before `idx` in `leaf_handle`, following the leaf chain.
    fn position_before(&self, leaf_handle: Handle, idx: usize) -> Option<(Handle, usize)> {
        if idx > 0 {
            return Some((leaf_handle, idx - 1));
        }
        let prev = self.nodes.get(leaf_handle).as_leaf().prev()?;
        let count = self.nodes.get(prev).as_leaf().key_count();
        count.checked_sub(1).map(|last| (prev, last))
    }

    /// Returns the first position after the split point for `key` (see
    /// [`Self::split_point`]) and the number of elements before the split.
    fn first_position_after<Q>(&self, key: &Q, include_equal: bool) -> (Option<(Handle, usize)>, usize)
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        match self.split_point(key, include_equal) {
            Some((leaf_handle, idx, rank)) => (self.position_at_or_after(leaf_handle, idx), rank),
            None => (None, 0),
        }
    }

    /// Returns the last position before the split point for `key` (see
    /// [`Self::split_point`]) and the number of elements before the split.
    fn last_position_before<Q>(&self, key: &Q, include_equal: bool) -> (Option<(Handle, usize)>, usize)
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        match self.split_point(key, include_equal) {
            Some((leaf_handle, idx, rank)) => (self.position_before(leaf_handle, idx), rank),
            None => (None, 0),
        }
    }

    /// Resolves `range` to the positions of its first and last elements and its length.
    pub(crate) fn range_cursor<T, R>(&self, range: &R) -> RangeCursor
    where
        K: Borrow<T>,
        T: ?Sized + Ord,
        R: RangeBounds<T>,
    {
        let (front, start_rank) = match range.start_bound() {
            Bound::Unbounded => (self.first_position(), 0),
            Bound::Included(start) => self.first_position_after(start, false),
            Bound::Excluded(start) => self.first_position_after(start, true),
        };
        let (back, end_rank) = match range.end_bound() {
            Bound::Unbounded => (self.last_position(), self.len),
            Bound::Included(end) => self.last_position_before(end, true),
            Bound::Excluded(end) => self.last_position_before(end, false),
        };

        match (front, back) {
            (Some((front_leaf, front_index)), Some((back_leaf, back_index))) if end_rank > start_rank => RangeCursor {
                front_leaf: Some(front_leaf),
                front_index,
                back_leaf: Some(back_leaf),
                back_index,
                len: end_rank - start_rank,
                start_rank,
            },
            _ => RangeCursor::default(),
        }
    }
}

/// Returns the number of elements in the children of `internal` before `child_idx`.
fn size_before<K>(internal: &InternalNode<K>, child_idx: usize) -> usize {
    internal.child_sizes()[..child_idx].iter().map(|s| s.to_usize()).sum()
}

/// Returns the index of the child of `internal` holding the element `rank` positions into its
/// subtree, and the number of elements in the children before it. Returns `None` if `rank` is
/// not less than the sum of the child sizes.
fn child_for_rank<K>(internal: &InternalNode<K>, rank: usize) -> Option<(usize, usize)> {
    const CHUNK: usize = 16;
    let sizes = internal.child_sizes();
    let mut before = 0;
    let mut skipped = 0;
    // Skip whole chunks of children first: summing a chunk has no early exit, so it vectorises.
    for chunk in sizes.as_chunks::<CHUNK>().0 {
        let chunk_size: usize = chunk.iter().map(|s| s.to_usize()).sum();
        if rank - before < chunk_size {
            break;
        }
        before += chunk_size;
        skipped += CHUNK;
    }
    for (i, child_size) in sizes.iter().enumerate().skip(skipped) {
        let child_size = child_size.to_usize();
        if rank - before < child_size {
            return Some((i, before));
        }
        before += child_size;
    }
    None
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(
    clippy::collapsible_if,
    clippy::uninlined_format_args,
    clippy::stable_sort_primitive,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use crate::raw::node::MIN_CHILDREN;
    use alloc::string::String;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    impl<K: Ord + Clone, V> RawOSBTreeMap<K, V> {
        /// First position with a key >= `key`.
        fn lower_bound(&self, key: &K) -> Option<(Handle, usize)> {
            self.first_position_after(key, false).0
        }

        /// First position with a key > `key`.
        fn upper_bound(&self, key: &K) -> Option<(Handle, usize)> {
            self.first_position_after(key, true).0
        }

        /// Last position with a key <= `key`.
        fn upper_bound_inclusive(&self, key: &K) -> Option<(Handle, usize)> {
            self.last_position_before(key, true).0
        }

        /// Last position with a key < `key`.
        fn lower_bound_exclusive(&self, key: &K) -> Option<(Handle, usize)> {
            self.last_position_before(key, false).0
        }

        /// Validates all B+Tree invariants. Panics with a descriptive message if any are violated.
        /// This is intended for use in tests to catch tree corruption.
        pub(crate) fn validate_invariants(&self) {
            if self.root.is_none() {
                // Empty tree
                assert_eq!(self.len, 0, "Empty tree should have len 0");
                assert!(self.first_leaf.is_none(), "Empty tree should have no first_leaf");
                assert!(self.last_leaf.is_none(), "Empty tree should have no last_leaf");
                return;
            }

            let root = self.root.unwrap();
            let mut errors: Vec<String> = Vec::new();

            // 1. Validate tree structure and collect all leaves
            let mut all_leaves: Vec<Handle> = Vec::new();
            let mut leaf_depth: Option<usize> = None;
            self.validate_node(root, 0, &mut leaf_depth, &mut all_leaves, &mut errors);

            // 2. Validate leaf chain matches collected leaves
            self.validate_leaf_chain(&all_leaves, &mut errors);

            // 3. Validate len matches actual count
            let actual_count: usize = all_leaves.iter().map(|&h| self.nodes.get(h).as_leaf().key_count()).sum();
            if self.len != actual_count {
                errors.push(alloc::format!("len mismatch: self.len={}, actual count={}", self.len, actual_count));
            }

            // 4. Validate root size equals len (if root is internal)
            if let Node::Internal(internal) = self.nodes.get(root) {
                if internal.size().to_usize() != self.len {
                    let root_size = internal.size().to_usize();
                    let len = self.len;
                    errors.push(alloc::format!("Root size mismatch: root.size={root_size}, self.len={len}"));
                }
            }

            assert!(errors.is_empty(), "Tree invariant violations:\n{}", errors.join("\n"));
        }

        fn validate_node(
            &self,
            handle: Handle,
            depth: usize,
            leaf_depth: &mut Option<usize>,
            all_leaves: &mut Vec<Handle>,
            errors: &mut Vec<String>,
        ) -> (Option<K>, usize) {
            // Returns (max_key, subtree_size)
            let node = self.nodes.get(handle);
            match node {
                Node::Leaf(leaf) => {
                    // Check leaf depth consistency
                    match *leaf_depth {
                        None => *leaf_depth = Some(depth),
                        Some(expected) => {
                            if depth != expected {
                                errors.push(alloc::format!(
                                    "Leaf depth mismatch: expected {}, got {} at handle {:?}",
                                    expected,
                                    depth,
                                    handle
                                ));
                            }
                        }
                    }

                    // Check keys are sorted
                    for i in 1..leaf.key_count() {
                        if leaf.key(i - 1) >= leaf.key(i) {
                            errors.push(alloc::format!(
                                "Leaf keys not sorted at handle {:?}, indices {} and {}",
                                handle,
                                i - 1,
                                i
                            ));
                        }
                    }

                    // Every leaf except a root leaf is at least half full.
                    if depth > 0 && leaf.key_count() < MIN_LEAF_KEYS {
                        errors.push(alloc::format!(
                            "Underfull leaf at handle {:?}: {} keys, minimum {}",
                            handle,
                            leaf.key_count(),
                            MIN_LEAF_KEYS
                        ));
                    }

                    all_leaves.push(handle);

                    let max_key = leaf.last_key().cloned();
                    (max_key, leaf.key_count())
                }
                Node::Internal(internal) => {
                    // A non-root internal node is at least half full; an internal root has at
                    // least two children.
                    let min_children = if depth == 0 {
                        2
                    } else {
                        MIN_CHILDREN
                    };
                    if internal.child_count() < min_children {
                        errors.push(alloc::format!(
                            "Underfull internal node at handle {:?}: {} children, minimum {}",
                            handle,
                            internal.child_count(),
                            min_children
                        ));
                    }

                    // Check keys are sorted
                    for i in 1..internal.key_count() {
                        if internal.key(i - 1) >= internal.key(i) {
                            errors.push(alloc::format!(
                                "Internal keys not sorted at handle {:?}, indices {} and {}",
                                handle,
                                i - 1,
                                i
                            ));
                        }
                    }

                    // Validate children
                    // Note: child_count = key_count + 1 in B+tree
                    // keys[i] = max key of children[i] for i < key_count
                    // children[key_count] is the rightmost child with no corresponding key
                    let mut total_size = 0usize;
                    let mut last_child_max: Option<K> = None;
                    for i in 0..internal.child_count() {
                        let child = internal.child(i);
                        let (child_max, child_size) =
                            self.validate_node(child, depth + 1, leaf_depth, all_leaves, errors);

                        // Check child_size matches stored size
                        let stored_size = internal.child_size(i).to_usize();
                        if stored_size != child_size {
                            errors.push(alloc::format!(
                                "Child size mismatch at handle {:?} child {}: stored={}, actual={}",
                                handle,
                                i,
                                stored_size,
                                child_size
                            ));
                        }
                        total_size += child_size;

                        // Note: We don't strictly validate that separator[i] == max key of child[i]
                        // because after rebalancing operations (borrow/merge), separators may become
                        // "stale" but the tree still works correctly for searches. The important
                        // invariant is that keys are sorted and separators correctly partition
                        // the key space (child[i]'s keys <= separator[i] < child[i+1]'s keys).

                        last_child_max = child_max;
                    }

                    // Check internal node size equals sum of child sizes
                    if internal.size().to_usize() != total_size {
                        errors.push(alloc::format!(
                            "Internal size mismatch at handle {:?}: stored={}, computed={}",
                            handle,
                            internal.size().to_usize(),
                            total_size
                        ));
                    }

                    // The max key of this internal node is the max key of its rightmost child
                    // (last_child_max).
                    (last_child_max, total_size)
                }
            }
        }

        fn validate_leaf_chain(&self, all_leaves: &[Handle], errors: &mut Vec<String>) {
            if all_leaves.is_empty() {
                if self.first_leaf.is_some() || self.last_leaf.is_some() {
                    errors.push("first_leaf/last_leaf should be None for empty tree".into());
                }
                return;
            }

            // Check first_leaf
            if self.first_leaf != Some(all_leaves[0]) {
                errors.push(alloc::format!(
                    "first_leaf mismatch: expected {:?}, got {:?}",
                    Some(all_leaves[0]),
                    self.first_leaf
                ));
            }

            // Check last_leaf
            if self.last_leaf != Some(*all_leaves.last().unwrap()) {
                errors.push(alloc::format!(
                    "last_leaf mismatch: expected {:?}, got {:?}",
                    all_leaves.last().copied(),
                    self.last_leaf
                ));
            }

            // Check forward chain
            for i in 0..all_leaves.len() {
                let leaf = self.nodes.get(all_leaves[i]).as_leaf();
                let expected_next = if i + 1 < all_leaves.len() {
                    Some(all_leaves[i + 1])
                } else {
                    None
                };
                if leaf.next() != expected_next {
                    errors.push(alloc::format!(
                        "Leaf chain next mismatch at index {}: expected {:?}, got {:?}",
                        i,
                        expected_next,
                        leaf.next()
                    ));
                }
            }

            // Check backward chain
            for i in 0..all_leaves.len() {
                let leaf = self.nodes.get(all_leaves[i]).as_leaf();
                let expected_prev = if i > 0 {
                    Some(all_leaves[i - 1])
                } else {
                    None
                };
                if leaf.prev() != expected_prev {
                    errors.push(alloc::format!(
                        "Leaf chain prev mismatch at index {}: expected {:?}, got {:?}",
                        i,
                        expected_prev,
                        leaf.prev()
                    ));
                }
            }
        }
    }

    // Test operations enum for property testing
    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32),
        Remove(i32),
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            3 => (0i32..1000).prop_map(Op::Insert),
            1 => (0i32..1000).prop_map(Op::Remove),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn tree_invariants_maintained_after_operations(ops in prop::collection::vec(op_strategy(), 0..500)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

            for op in ops {
                match op {
                    Op::Insert(key) => {
                        tree.insert(key, key * 2);
                    }
                    Op::Remove(key) => {
                        tree.remove(&key);
                    }
                }
                tree.validate_invariants();
            }
        }

        #[test]
        fn get_by_rank_correctness(ops in prop::collection::vec((0i32..500).prop_map(Op::Insert), 1..200)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
            let mut expected: Vec<i32> = Vec::new();

            for op in ops {
                if let Op::Insert(key) = op {
                    if tree.insert(key, key * 2).is_none() {
                        expected.push(key);
                    }
                }
            }
            expected.sort();

            tree.validate_invariants();

            // Test get_by_rank for all valid ranks
            for (rank, &expected_key) in expected.iter().enumerate() {
                let result = tree.get_by_rank(rank);
                prop_assert!(result.is_some(), "get_by_rank({}) returned None", rank);
                let (key, value) = result.unwrap();
                prop_assert_eq!(*key, expected_key, "get_by_rank({}) returned wrong key", rank);
                prop_assert_eq!(*value, expected_key * 2, "get_by_rank({}) returned wrong value", rank);
            }

            // Test out of bounds
            prop_assert!(tree.get_by_rank(expected.len()).is_none());
        }

        #[test]
        fn rank_of_correctness(ops in prop::collection::vec((0i32..500).prop_map(Op::Insert), 1..200)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
            let mut expected: Vec<i32> = Vec::new();

            for op in ops {
                if let Op::Insert(key) = op {
                    if tree.insert(key, key * 2).is_none() {
                        expected.push(key);
                    }
                }
            }
            expected.sort();

            tree.validate_invariants();

            // Test rank_of for all keys
            for (rank, &key) in expected.iter().enumerate() {
                let result = tree.rank_of(&key);
                prop_assert_eq!(result, Some(rank), "rank_of({}) returned wrong rank", key);
            }

            // Test non-existent key
            let max_key = expected.iter().max().copied().unwrap_or(0);
            prop_assert!(tree.rank_of(&(max_key + 1)).is_none());
        }

        #[test]
        fn clone_after_removals_is_compact_and_valid(ops in prop::collection::vec(op_strategy(), 0..500)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
            for op in ops {
                match op {
                    Op::Insert(key) => { tree.insert(key, key * 2); }
                    Op::Remove(key) => { tree.remove(&key); }
                }
            }

            let clone = tree.clone();
            clone.validate_invariants();
            prop_assert_eq!(clone.len(), tree.len());
            // Only live entries are copied: no free slots carry over.
            prop_assert_eq!(clone.values.slot_count(), tree.len());
            prop_assert_eq!(clone.nodes.slot_count(), clone.nodes.len());
            prop_assert_eq!(clone.capacity(), tree.capacity());
            for rank in 0..tree.len() {
                prop_assert_eq!(clone.get_by_rank(rank), tree.get_by_rank(rank));
            }
        }

        #[test]
        fn rank_roundtrip(ops in prop::collection::vec((0i32..500).prop_map(Op::Insert), 1..200)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

            for op in ops {
                if let Op::Insert(key) = op {
                    tree.insert(key, key * 2);
                }
            }

            tree.validate_invariants();

            // For every key, rank_of(key) should give a rank where get_by_rank returns the same key
            let mut current = tree.first_leaf;
            while let Some(leaf_handle) = current {
                let leaf = tree.nodes.get(leaf_handle).as_leaf();
                for i in 0..leaf.key_count() {
                    let key = leaf.key(i);
                    let rank = tree.rank_of(key).expect("rank_of should succeed for existing key");
                    let (retrieved_key, _) = tree.get_by_rank(rank).expect("get_by_rank should succeed");
                    prop_assert_eq!(key, retrieved_key, "rank roundtrip failed");
                }
                current = leaf.next();
            }
        }

        #[test]
        fn boundary_rank_operations(count in 1usize..100) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

            for i in 0..count as i32 {
                tree.insert(i, i * 2);
            }

            tree.validate_invariants();

            // Test Rank(0) - first element
            let (first_key, first_value) = tree.get_by_rank(0).expect("get_by_rank(0) should succeed");
            prop_assert_eq!(*first_key, 0, "First key should be 0");
            prop_assert_eq!(*first_value, 0, "First value should be 0");

            // Test Rank(len-1) - last element
            let last_rank = count - 1;
            let (last_key, last_value) = tree.get_by_rank(last_rank).expect("get_by_rank(last) should succeed");
            prop_assert_eq!(*last_key, (count - 1) as i32, "Last key should be count-1");
            prop_assert_eq!(*last_value, ((count - 1) * 2) as i32, "Last value should be (count-1)*2");

            // Test out of bounds
            prop_assert!(tree.get_by_rank(count).is_none(), "get_by_rank(len) should be None");
            prop_assert!(tree.get_by_rank(count + 100).is_none(), "get_by_rank(len+100) should be None");
        }

        #[test]
        fn interleaved_rank_and_mutations(ops in prop::collection::vec(op_strategy(), 0..300)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
            let mut expected: alloc::collections::BTreeMap<i32, i32> = alloc::collections::BTreeMap::new();

            for op in ops {
                match op {
                    Op::Insert(key) => {
                        tree.insert(key, key * 2);
                        expected.insert(key, key * 2);
                    }
                    Op::Remove(key) => {
                        tree.remove(&key);
                        expected.remove(&key);
                    }
                }

                tree.validate_invariants();

                // After each operation, verify rank operations are consistent
                if !expected.is_empty() {
                    let expected_keys: Vec<_> = expected.keys().copied().collect();

                    // Check a few random ranks
                    for rank in [0, expected.len() / 2, expected.len() - 1] {
                        if rank < expected.len() {
                            let (key, _) = tree.get_by_rank(rank).expect("get_by_rank should succeed");
                            prop_assert_eq!(*key, expected_keys[rank], "Key at rank {} mismatch", rank);
                        }
                    }
                }
            }
        }
    }

    /// Operations that exercise the `_with_rank` methods and `remove_by_rank`.
    #[derive(Clone, Debug)]
    enum RankOp {
        Insert(i32),
        Remove(i32),
        Get(i32),
        GetMut(i32),
        RemoveByRank(usize),
    }

    fn rank_op_strategy() -> impl Strategy<Value = RankOp> {
        prop_oneof![
            4 => (0i32..1000).prop_map(RankOp::Insert),
            2 => (0i32..1000).prop_map(RankOp::Remove),
            2 => (0i32..1000).prop_map(RankOp::Get),
            1 => (0i32..1000).prop_map(RankOp::GetMut),
            // Reaches past the end, so out-of-range ranks are covered too.
            1 => (0usize..800).prop_map(RankOp::RemoveByRank),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Each `_with_rank` method agrees with the two-call form it replaces, and
        /// `remove_by_rank` agrees with `Vec::remove` on a sorted model.
        #[test]
        fn with_rank_methods_match_two_call_form(ops in prop::collection::vec(rank_op_strategy(), 0..600)) {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
            let mut model: Vec<(i32, i32)> = Vec::new();

            for op in ops {
                match op {
                    RankOp::Insert(key) => {
                        let (rank, old) = tree.insert_with_rank(key, key * 2);
                        prop_assert_eq!(Some(rank), tree.rank_of(&key));
                        match model.binary_search_by_key(&key, |&(k, _)| k) {
                            Ok(i) => {
                                prop_assert_eq!(rank, i);
                                prop_assert_eq!(old, Some(core::mem::replace(&mut model[i].1, key * 2)));
                            }
                            Err(i) => {
                                prop_assert_eq!(rank, i);
                                prop_assert_eq!(old, None);
                                model.insert(i, (key, key * 2));
                            }
                        }
                    }
                    RankOp::Remove(key) => {
                        let expected_rank = tree.rank_of(&key);
                        let removed = tree.remove_entry_with_rank(&key);
                        let expected = model
                            .binary_search_by_key(&key, |&(k, _)| k)
                            .ok()
                            .map(|i| { let (k, v) = model.remove(i); (i, k, v) });
                        prop_assert_eq!(removed.map(|(r, _, _)| r), expected_rank);
                        prop_assert_eq!(removed, expected);
                    }
                    RankOp::Get(key) => {
                        let expected = tree.rank_of(&key).map(|r| (r, tree.get_key_value(&key).unwrap()));
                        let got = tree.get_with_rank(&key).map(|(r, k, v)| (r, (k, v)));
                        prop_assert_eq!(got, expected);
                    }
                    RankOp::GetMut(key) => {
                        let expected_rank = tree.rank_of(&key);
                        let got = tree.get_with_rank_mut(&key).map(|(rank, k, v)| {
                            *v += 1;
                            (rank, *k)
                        });
                        prop_assert_eq!(got, expected_rank.map(|rank| (rank, key)));
                        if let Some(rank) = expected_rank {
                            model[rank].1 += 1;
                        }
                    }
                    RankOp::RemoveByRank(rank) => {
                        let expected = (rank < model.len()).then(|| model.remove(rank));
                        prop_assert_eq!(tree.remove_by_rank(rank), expected);
                    }
                }

                tree.validate_invariants();
                prop_assert_eq!(tree.len(), model.len());
            }

            let entries: Vec<(i32, i32)> = (0..tree.len()).map(|r| { let (k, v) = tree.get_by_rank(r).unwrap(); (*k, *v) }).collect();
            prop_assert_eq!(entries, model);
        }
    }

    /// Number of levels in `tree`, counting the leaves.
    fn tree_height<K, V>(tree: &RawOSBTreeMap<K, V>) -> usize {
        let mut height = 0;
        let mut node = tree.root;
        while let Some(handle) = node {
            height += 1;
            node = match tree.nodes.get(handle) {
                Node::Internal(internal) => Some(internal.child(0)),
                Node::Leaf(_) => None,
            };
        }
        height
    }

    /// Inserting into a full root leaf splits it, and with it the root. The rank must be right
    /// wherever the new key lands: in the left half, at the split point, or in the right half.
    #[test]
    fn insert_with_rank_across_a_root_leaf_split() {
        for position in 0..=MAX_KEYS {
            let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
            for i in 0..MAX_KEYS as i32 {
                tree.insert(i * 2, 0);
            }
            assert_eq!(tree_height(&tree), 1);

            let key = position as i32 * 2 - 1;
            assert_eq!(tree.insert_with_rank(key, 1), (position, None), "position = {position}");
            assert_eq!(tree_height(&tree), 2, "position = {position}");
            assert_eq!(tree.rank_of(&key), Some(position));
            tree.validate_invariants();
        }
    }

    /// Ascending and descending inserts split leaves and internal nodes at every level; each
    /// insert reports the rank the key ends up at.
    #[test]
    fn insert_with_rank_through_internal_splits() {
        let n = MAX_KEYS * MAX_KEYS * 3;

        let mut ascending: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        for i in 0..n {
            assert_eq!(ascending.insert_with_rank(i as i32, 0), (i, None));
        }
        ascending.validate_invariants();
        assert!(tree_height(&ascending) >= 3);

        let mut descending: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        for i in (0..n).rev() {
            assert_eq!(descending.insert_with_rank(i as i32, 0), (0, None));
        }
        descending.validate_invariants();

        // Present keys report their rank and keep their position.
        for i in [0, n / 2, n - 1] {
            assert_eq!(ascending.insert_with_rank(i as i32, 1), (i, Some(0)));
        }
    }

    #[test]
    fn with_rank_methods_on_empty_and_single_element_trees() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        assert_eq!(tree.get_with_rank(&1), None);
        assert_eq!(tree.get_with_rank_mut(&1), None);
        assert_eq!(tree.remove_entry_with_rank(&1), None);
        assert_eq!(tree.remove_by_rank(0), None);

        assert_eq!(tree.insert_with_rank(1, 10), (0, None));
        assert_eq!(tree.get_with_rank(&1), Some((0, &1, &10)));
        assert_eq!(tree.get_with_rank(&2), None);
        assert_eq!(tree.remove_by_rank(1), None);
        assert_eq!(tree.remove_entry_with_rank(&2), None);
        assert_eq!(tree.remove_entry_with_rank(&1), Some((0, 1, 10)));
        tree.validate_invariants();

        tree.insert(1, 10);
        assert_eq!(tree.remove_by_rank(0), Some((1, 10)));
        assert_eq!(tree.len(), 0);
        tree.validate_invariants();
    }

    #[test]
    fn empty_tree_rank_operations() {
        let tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        tree.validate_invariants();

        assert!(tree.get_by_rank(0).is_none());
        assert!(tree.get_by_rank(100).is_none());
        assert!(tree.rank_of(&0).is_none());
        assert!(tree.rank_of(&100).is_none());
    }

    #[test]
    fn single_element_rank_operations() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        tree.insert(42, 84);
        tree.validate_invariants();

        // get_by_rank
        let (key, value) = tree.get_by_rank(0).expect("should have rank 0");
        assert_eq!(*key, 42);
        assert_eq!(*value, 84);
        assert!(tree.get_by_rank(1).is_none());

        // rank_of
        assert_eq!(tree.rank_of(&42), Some(0));
        assert!(tree.rank_of(&0).is_none());
        assert!(tree.rank_of(&100).is_none());
    }

    #[test]
    fn get_by_rank_mut_modifies_value() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        for i in 0..10 {
            tree.insert(i, i * 2);
        }
        tree.validate_invariants();

        // Modify value at rank 5
        {
            let (key, value) = tree.get_by_rank_mut(5).expect("should have rank 5");
            assert_eq!(*key, 5);
            assert_eq!(*value, 10);
            *value = 999;
        }

        tree.validate_invariants();

        // Verify modification persisted
        let (_, value) = tree.get_by_rank(5).expect("should still have rank 5");
        assert_eq!(*value, 999);

        // Verify other values unchanged
        let (_, value) = tree.get_by_rank(4).expect("should have rank 4");
        assert_eq!(*value, 8);
    }

    #[test]
    fn ranks_stable_after_rebalancing() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

        // Insert enough elements to create multiple levels
        for i in 0..100 {
            tree.insert(i, i * 2);
            tree.validate_invariants();
        }

        // Before removals, each key's rank equals its value.
        for (expected_rank, key) in (0..100).enumerate() {
            assert_eq!(tree.rank_of(&key), Some(expected_rank));
        }

        // Remove some elements (this may trigger rebalancing)
        for i in (0..100).step_by(3) {
            tree.remove(&i);
            tree.validate_invariants();
        }

        // Verify remaining keys have correct relative ordering
        let remaining: Vec<i32> = (0..100).filter(|i| i % 3 != 0).collect();
        for (expected_rank, &key) in remaining.iter().enumerate() {
            let actual_rank = tree.rank_of(&key).expect("remaining key should exist");
            assert_eq!(actual_rank, expected_rank, "Rank of {} should be {}", key, expected_rank);
        }
    }

    #[test]
    fn accessors_and_pointer_helpers_work() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        tree.insert(1, 10);
        tree.insert(2, 20);

        let root = tree.root().expect("root exists after inserts");
        let first_leaf = tree.first_leaf().expect("first leaf exists after inserts");
        let last_leaf = tree.last_leaf().expect("last leaf exists after inserts");
        assert_eq!(first_leaf, last_leaf);

        // Access mutable node reference.
        let leaf_count = tree.node_mut(first_leaf).as_leaf_mut().key_count();
        assert!(leaf_count > 0);

        // SAFETY: `tree_ptr` points to a valid tree and `first_leaf` is a valid handle.
        unsafe {
            let tree_ptr: *const RawOSBTreeMap<i32, i32> = &raw const tree;
            let node = RawOSBTreeMap::node_ptr(tree_ptr, first_leaf);
            assert!(node.is_leaf());
        }

        // Mutate a value through the raw pointer helper.
        let value_handle = tree.node(first_leaf).as_leaf().value(0);
        // SAFETY: `tree_ptr` points to a valid tree and `value_handle` is a valid handle.
        unsafe {
            let tree_ptr: *mut RawOSBTreeMap<i32, i32> = &raw mut tree;
            let value = RawOSBTreeMap::value_mut_ptr(tree_ptr, value_handle);
            *value += 1;
        }
        assert_eq!(tree.get(&1), Some(&11));

        // Root accessor is used in several iterators; ensure it is reachable.
        assert!(tree.node(root).key_count() > 0);
        tree.validate_invariants();
    }

    /// The tree's leaves, first to last, by walking the leaf chain.
    fn leaf_chain<K, V>(tree: &RawOSBTreeMap<K, V>) -> Vec<Handle> {
        let mut leaves = Vec::new();
        let mut current = tree.first_leaf();
        while let Some(handle) = current {
            leaves.push(handle);
            current = tree.node(handle).as_leaf().next();
        }
        leaves
    }

    #[test]
    fn empty_leaf_neighbors_cause_bounds_to_return_none() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        // Use many keys to ensure multiple leaves.
        for i in 0..4000 {
            tree.insert(i * 2, i);
        }
        tree.validate_invariants();

        let leaves = leaf_chain(&tree);
        assert!(leaves.len() >= 5, "expected multiple leaves");

        let mid_index = leaves.len() / 2;
        let mid = leaves[mid_index];
        let prev = leaves[mid_index - 1];
        let next = leaves[mid_index + 1];

        let mid_key = *tree.node(mid).as_leaf().key(0);

        // Create empty neighbors around `mid` to exercise empty-leaf fallbacks.
        tree.node_mut(prev).as_leaf_mut().drain(..).for_each(drop);
        tree.node_mut(mid).as_leaf_mut().drain(..).for_each(drop);
        tree.node_mut(next).as_leaf_mut().drain(..).for_each(drop);

        // With an empty next leaf, lower/upper bounds should return None here.
        assert_eq!(tree.lower_bound(&mid_key), None);
        assert_eq!(tree.upper_bound(&mid_key), None);
        assert_eq!(tree.upper_bound_inclusive(&mid_key), None);
        assert_eq!(tree.lower_bound_exclusive(&mid_key), None);

        // Exercise the early return when the leaf is empty but the path is non-empty.
        let root = tree.root().expect("root exists");
        let mut synthetic_path: Path = SmallVec::new();
        synthetic_path.push(PathElement {
            node: root,
            child_index: 0,
        });
        tree.update_separators_after_remove(mid, 0, &synthetic_path);
    }

    #[test]
    fn refresh_sizes_covers_leaf_children() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        for i in 0..64 {
            tree.insert(i, i);
        }
        tree.validate_invariants();

        let root = tree.root().expect("root exists");
        let root_internal = match tree.node(root) {
            Node::Internal(internal) => internal,
            Node::Leaf(_) => panic!("expected internal root for this test"),
        };
        let first_child = root_internal.child(0);
        assert!(matches!(tree.node(first_child), Node::Leaf(_)));

        let mut path: Path = SmallVec::new();
        path.push(PathElement {
            node: root,
            child_index: 0,
        });
        tree.refresh_sizes(&path);

        tree.validate_invariants();
    }

    fn make_two_leaf_tree(left_keys: &[i32], right_keys: &[i32]) -> (RawOSBTreeMap<i32, i32>, Handle, Handle, Handle) {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

        let left_handle = tree.nodes.alloc(Node::new_leaf());
        let right_handle = tree.nodes.alloc(Node::new_leaf());

        for &key in left_keys {
            let value_handle = tree.values.alloc(key * 10);
            tree.nodes.get_mut(left_handle).as_leaf_mut().push(key, value_handle);
        }
        for &key in right_keys {
            let value_handle = tree.values.alloc(key * 10);
            tree.nodes.get_mut(right_handle).as_leaf_mut().push(key, value_handle);
        }

        tree.nodes.get_mut(left_handle).as_leaf_mut().set_next(Some(right_handle));
        tree.nodes.get_mut(right_handle).as_leaf_mut().set_prev(Some(left_handle));

        let root_handle = tree.nodes.alloc(Node::new_internal());
        {
            let root = tree.nodes.get_mut(root_handle).as_internal_mut();
            root.set_first_child(left_handle, Size::from_usize(left_keys.len()));
            root.push_child(
                *left_keys.last().expect("left leaf must be non-empty"),
                right_handle,
                Size::from_usize(right_keys.len()),
            );
            root.update_size();
        }

        tree.root = Some(root_handle);
        tree.first_leaf = Some(left_handle);
        tree.last_leaf = Some(right_handle);
        tree.len = left_keys.len() + right_keys.len();

        (tree, root_handle, left_handle, right_handle)
    }

    #[test]
    fn take_value_and_empty_first_last_leaf_return_none() {
        // Cover the take_value accessor using a valid value handle.
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        tree.insert(1, 10);
        let leaf = tree.first_leaf().expect("leaf exists");
        let value_handle = tree.node(leaf).as_leaf().value(0);
        let taken = tree.take_value(value_handle);
        assert_eq!(taken, 10);

        // Create multiple leaves, then empty the first and last leaves while keeping their handles.
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        for i in 0..400 {
            tree.insert(i * 2, i);
        }

        let first = tree.first_leaf().expect("first leaf exists");
        let last = tree.last_leaf().expect("last leaf exists");
        tree.node_mut(first).as_leaf_mut().drain(..).for_each(drop);
        tree.node_mut(last).as_leaf_mut().drain(..).for_each(drop);

        assert_eq!(tree.first_key_value(), None);
        assert_eq!(tree.last_key_value(), None);
    }

    #[test]
    fn bounds_fallback_to_next_leaf_with_stale_separator() {
        let (mut tree, root_handle, _left, right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);

        // Make the separator far too large so search_child can choose the left leaf incorrectly.
        tree.nodes.get_mut(root_handle).as_internal_mut().set_key(0, i32::MAX);

        assert_eq!(tree.lower_bound(&7), Some((right, 0)));
        assert_eq!(tree.upper_bound(&7), Some((right, 0)));
        assert_eq!(tree.upper_bound(&4), Some((right, 0)));
    }

    #[test]
    fn lower_bound_exclusive_uses_previous_leaf_on_boundary() {
        // Leaves of `MIN_LEAF_KEYS` keys each, so the fixture passes `validate_invariants`.
        let left_keys: Vec<i32> = (0..8).map(|i| i * 2).collect();
        let right_keys: Vec<i32> = (8..16).map(|i| i * 2).collect();
        let (tree, _root, left, _right) = make_two_leaf_tree(&left_keys, &right_keys);
        tree.validate_invariants();

        // Exact boundary: key is the first key of the right leaf.
        assert_eq!(tree.lower_bound_exclusive(&16), Some((left, 7)));
        // Between leaves: not found in right leaf at index 0, so we step to the previous leaf.
        assert_eq!(tree.lower_bound_exclusive(&15), Some((left, 7)));
    }

    #[test]
    fn upper_bound_returns_none_when_next_leaf_is_empty() {
        let (mut tree, _root, _left, right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        tree.node_mut(right).as_leaf_mut().drain(..).for_each(drop);

        // Found at the last index with an empty next leaf should return None.
        assert_eq!(tree.upper_bound(&4), None);
    }

    #[test]
    fn lower_bound_exclusive_returns_none_when_prev_leaf_is_empty() {
        let (mut tree, _root, left, _right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        tree.node_mut(left).as_leaf_mut().drain(..).for_each(drop);

        // Found at index 0 with an empty previous leaf should return None.
        assert_eq!(tree.lower_bound_exclusive(&6), None);
    }

    #[test]
    fn upper_bound_returns_none_on_last_key_without_next_leaf() {
        let (tree, _root, _left, _right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        assert_eq!(tree.upper_bound(&10), None);
    }

    #[test]
    fn lower_bound_exclusive_returns_none_on_first_key_without_prev_leaf() {
        let (tree, _root, _left, _right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        assert_eq!(tree.lower_bound_exclusive(&0), None);
    }

    #[test]
    fn remove_from_parent_updates_sizes_when_path_child_is_leaf() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

        let leaf_count = crate::raw::node::MIN_INTERNAL_KEYS + 3;
        let mut leaves: Vec<Handle> = Vec::with_capacity(leaf_count);
        for i in 0..leaf_count {
            let handle = tree.nodes.alloc(Node::new_leaf());
            let key = (i as i32) * 2;
            let value_handle = tree.values.alloc(key * 10);
            tree.nodes.get_mut(handle).as_leaf_mut().push(key, value_handle);
            leaves.push(handle);
        }

        for (i, &handle) in leaves.iter().enumerate() {
            let prev = i.checked_sub(1).and_then(|j| leaves.get(j).copied());
            let next = leaves.get(i + 1).copied();
            let leaf = tree.nodes.get_mut(handle).as_leaf_mut();
            leaf.set_prev(prev);
            leaf.set_next(next);
        }

        let bottom_handle = tree.nodes.alloc(Node::new_internal());
        {
            let bottom = tree.nodes.get_mut(bottom_handle).as_internal_mut();
            bottom.set_first_child(leaves[0], Size::from_usize(1));
            for (i, &child) in leaves.iter().enumerate().skip(1) {
                let sep_key = ((i - 1) as i32) * 2;
                bottom.push_child(sep_key, child, Size::from_usize(1));
            }
            bottom.update_size();
        }

        let root_handle = tree.nodes.alloc(Node::new_internal());
        {
            let bottom_size = tree.nodes.get(bottom_handle).as_internal().size();
            let root = tree.nodes.get_mut(root_handle).as_internal_mut();
            root.set_first_child(bottom_handle, bottom_size);
            root.update_size();
        }

        tree.root = Some(root_handle);
        tree.first_leaf = Some(leaves[0]);
        tree.last_leaf = Some(leaves[leaf_count - 1]);
        tree.len = leaf_count;

        // Use a duplicate bottom element so the remaining path includes an internal whose child is a leaf.
        let mut path: Path = SmallVec::new();
        path.push(PathElement {
            node: root_handle,
            child_index: 0,
        });
        path.push(PathElement {
            node: bottom_handle,
            child_index: 0,
        });
        path.push(PathElement {
            node: bottom_handle,
            child_index: 0,
        });

        tree.nodes.get_mut(bottom_handle).as_internal_mut().remove_child(1);
        tree.propagate_merge(&mut path, 1);

        let bottom_size = tree.nodes.get(bottom_handle).as_internal().size();
        let root_child_size = tree.nodes.get(root_handle).as_internal().child_size(0);
        assert_eq!(root_child_size, bottom_size);
    }

    #[test]
    #[should_panic(expected = "locate_rank: tree size invariant violated")]
    fn get_by_rank_panics_when_internal_sizes_are_inconsistent() {
        let (mut tree, root_handle, _left, _right) = make_two_leaf_tree(&[0, 2], &[4, 6]);
        {
            let root = tree.nodes.get_mut(root_handle).as_internal_mut();
            root.set_size(Size::from_usize(5));
        }
        tree.len = 5;

        let _ = tree.get_by_rank(4);
    }

    #[test]
    #[should_panic(expected = "locate_rank: tree size invariant violated")]
    fn get_by_rank_mut_panics_when_internal_sizes_are_inconsistent() {
        let (mut tree, root_handle, _left, _right) = make_two_leaf_tree(&[0, 2], &[4, 6]);
        {
            let root = tree.nodes.get_mut(root_handle).as_internal_mut();
            root.set_size(Size::from_usize(5));
        }
        tree.len = 5;

        let _ = tree.get_by_rank_mut(4);
    }

    // ─── Bulk construction ──────────────────────────────────────────────────

    /// `n` entries with keys `0, 2, 4, ...` and values equal to their rank.
    fn sorted_entries(n: usize) -> impl ExactSizeIterator<Item = (i32, usize)> {
        (0..n).map(|rank| (i32::try_from(rank * 2).unwrap(), rank))
    }

    fn sorted_tree(n: usize) -> RawOSBTreeMap<i32, usize> {
        RawOSBTreeMap::from_sorted_iter(sorted_entries(n))
    }

    #[test]
    fn from_sorted_iter_builds_valid_packed_trees_at_every_size() {
        // With ORDER = 16, 600 entries reach three levels; 3,840 = 16 * 16 * 15 is where a
        // fourth level starts.
        for n in (0..=600).chain([3_839, 3_840, 3_841, 4_000]) {
            let tree = sorted_tree(n);
            tree.validate_invariants();
            assert_eq!(tree.len(), n);
            // As few leaves as fit, not the half-full leaves that sorted inserts leave behind.
            assert_eq!(leaf_chain(&tree).len(), n.div_ceil(MAX_KEYS), "n = {n}");
            assert_eq!(tree.first_key_value().map(|(&k, &v)| (k, v)), (n > 0).then_some((0, 0)));
            assert_eq!(tree.last_key_value().map(|(&k, &v)| (k, v)), sorted_entries(n).last());
            for (key, rank) in sorted_entries(n) {
                assert_eq!(tree.get_by_rank(rank).map(|(&k, &v)| (k, v)), Some((key, rank)));
                assert_eq!(tree.rank_of(&key), Some(rank));
                assert_eq!(tree.get(&key), Some(&rank));
                assert_eq!(tree.get(&(key + 1)), None);
            }
        }
    }

    #[test]
    fn from_sorted_iter_allocates_no_spare_slots() {
        let tree = sorted_tree(4_000);
        assert_eq!(tree.values.slot_count(), 4_000);
        assert!(tree.values.capacity() >= 4_000);
        // 267 leaves, 17 nodes above them, 2 above those, then the root.
        assert_eq!(tree.nodes.slot_count(), 267 + 17 + 2 + 1);
        assert!(tree.nodes.capacity() >= tree.nodes.slot_count());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Built trees must be ordinary trees: the insert and remove rebalancing code has to
        /// handle them like any other.
        #[test]
        fn from_sorted_iter_trees_stay_valid_under_mutation(
            n in 0usize..700,
            ops in prop::collection::vec(op_strategy(), 0..300),
        ) {
            let mut tree = sorted_tree(n);
            let mut model: alloc::collections::BTreeMap<i32, usize> = sorted_entries(n).collect();
            for op in ops {
                match op {
                    Op::Insert(key) => {
                        let value = usize::try_from(key).unwrap();
                        prop_assert_eq!(tree.insert(key, value), model.insert(key, value));
                    }
                    Op::Remove(key) => {
                        prop_assert_eq!(tree.remove(&key), model.remove(&key));
                    }
                }
                tree.validate_invariants();
            }
            prop_assert_eq!(tree.len(), model.len());
            for (rank, (&key, &value)) in model.iter().enumerate() {
                prop_assert_eq!(tree.get_by_rank(rank).map(|(&k, &v)| (k, v)), Some((key, value)));
            }
        }
    }

    #[test]
    #[should_panic(expected = "keys must be strictly increasing")]
    #[cfg(debug_assertions)]
    fn from_sorted_iter_rejects_unsorted_keys_in_debug() {
        let _ = RawOSBTreeMap::from_sorted_iter([(1, ()), (3, ()), (2, ())]);
    }

    #[test]
    #[should_panic(expected = "keys must be strictly increasing")]
    #[cfg(debug_assertions)]
    fn from_sorted_iter_rejects_unsorted_keys_across_leaves_in_debug() {
        // The second leaf's first key is smaller than the first leaf's last key.
        let keys = (0..MAX_KEYS).chain([0]).chain(MAX_KEYS + 1..MAX_KEYS * 2);
        let _ = RawOSBTreeMap::from_sorted_iter(keys.map(|k| (k, ())).collect::<Vec<_>>());
    }

    /// `insert_with_position` must report where the key ended up, including after a leaf split
    /// that moves it into the new right leaf and after splits that grow a new root.
    #[test]
    fn insert_with_position_reports_final_position() {
        let n = MAX_KEYS * MAX_KEYS * 3;
        let orders: [Vec<usize>; 3] =
            [(0..n).collect(), (0..n).rev().collect(), (0..n).map(|i| (i * 7919) % n).collect()];
        for keys in orders {
            let mut tree = RawOSBTreeMap::new();
            for &key in &keys {
                let ((leaf, idx), old) = tree.insert_with_position(key, key * 2);
                assert_eq!(old, None);
                assert_eq!(*tree.node(leaf).as_leaf().key(idx), key);
                assert_eq!(tree.entry_at((leaf, idx)), (&key, &(key * 2)));
            }
            tree.validate_invariants();
            for &key in &keys {
                let ((leaf, idx), old) = tree.insert_with_position(key, key * 3);
                assert_eq!(old, Some(key * 2));
                assert_eq!(tree.entry_at((leaf, idx)), (&key, &(key * 3)));
            }
        }
    }
}

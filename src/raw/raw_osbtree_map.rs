use core::borrow::Borrow;
use core::ops::{Bound, RangeBounds};

use smallvec::SmallVec;

use super::arena::Arena;
use super::handle::Handle;
use super::node::{InternalNode, LeafNode, MAX_KEYS, Node, SearchResult};
use super::size::Size;

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
        }
    }

    /// Creates a new tree with the specified capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Arena::with_capacity(capacity.div_ceil(MAX_KEYS)),
            values: Arena::with_capacity(capacity),
            root: None,
            len: 0,
            first_leaf: None,
            last_leaf: None,
        }
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
            let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
            let next = leaf.next();
            let (keys, value_handles) = leaf.take_all();

            for (key, vh) in keys.into_iter().zip(value_handles) {
                let value = self.values.take(vh);
                result.push((key, value));
            }

            current_leaf = next;
        }

        // Every value has been taken, so the tree can be reset wholesale.
        self.clear();

        result
    }

    /// Returns a reference to the first leaf, if any.
    pub(crate) fn first_leaf(&self) -> Option<Handle> {
        self.first_leaf
    }

    /// Returns a reference to the last leaf, if any.
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

    /// Returns a reference to the root node, if any.
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
                    new_leaf.set_prev(out.last_leaf);
                    let new_handle = out.nodes.alloc(Node::Leaf(new_leaf));
                    match out.last_leaf {
                        Some(prev) => out.nodes.get_mut(prev).as_leaf_mut().set_next(Some(new_handle)),
                        None => out.first_leaf = Some(new_handle),
                    }
                    out.last_leaf = Some(new_handle);
                    new_handle
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

        let mut out = Self {
            nodes: Arena::with_capacity(self.nodes.capacity()),
            values: Arena::with_capacity(self.values.capacity()),
            root: None,
            len: self.len,
            first_leaf: None,
            last_leaf: None,
        };
        if let Some(root) = self.root {
            out.root = Some(clone_node(self, &mut out, root));
        }
        out
    }
}

impl<K: Clone + Ord, V> RawOSBTreeMap<K, V> {
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
            rank += internal.child_sizes()[..child_idx].iter().map(|s| s.to_usize()).sum::<usize>();
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
        let (leaf_handle, idx) = self.search(key)?;
        let value_handle = self.nodes.get(leaf_handle).as_leaf().value(idx);
        Some(self.values.get(value_handle))
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
        let Some(root) = self.root else {
            let value_handle = self.values.alloc(value);
            let mut leaf = LeafNode::new();
            leaf.push(key, value_handle);
            let leaf_handle = self.nodes.alloc(Node::Leaf(leaf));
            self.root = Some(leaf_handle);
            self.first_leaf = Some(leaf_handle);
            self.last_leaf = Some(leaf_handle);
            self.len = 1;
            return (value_handle, None);
        };

        let mut path = Path::new();
        let current = self.descend_with_path(root, &mut path, |internal| internal.search_child(&key));

        // Insert into leaf
        let leaf = self.nodes.get_mut(current).as_leaf_mut();
        match leaf.search(&key) {
            SearchResult::Found(idx) => {
                // Key exists, replace value in-place to avoid alloc/free churn
                let value_handle = leaf.value(idx);
                let old_value = core::mem::replace(self.values.get_mut(value_handle), value);
                (value_handle, Some(old_value))
            }
            SearchResult::NotFound(idx) => {
                // Insert new key-value
                let value_handle = self.values.alloc(value);
                leaf.insert(idx, key, value_handle);
                self.len += 1;

                // Count the new element before splitting: a split only clones keys, so if
                // `K::clone` panics the tree is left consistent, with one oversized leaf.
                let needs_split = leaf.key_count() > MAX_KEYS;
                self.adjust_sizes_along_path(&path, |size| size + 1);
                if needs_split {
                    self.split_leaf_and_propagate(current, &mut path);
                }

                (value_handle, None)
            }
        }
    }

    /// Splits a leaf and propagates splits up the tree as needed.
    fn split_leaf_and_propagate(&mut self, leaf_handle: Handle, path: &mut Path) {
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        let (separator, mut right_leaf) = leaf.split();

        // Get the sizes
        let left_size = Size::from_usize(leaf.key_count());
        let right_size = Size::from_usize(right_leaf.key_count());

        // Set up leaf links
        let old_next = leaf.next();
        right_leaf.set_prev(Some(leaf_handle));
        right_leaf.set_next(old_next);
        leaf.set_next(None); // Will be set after we allocate right_leaf

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
        let root = self.root?;
        let mut path = Path::new();
        let leaf_handle = self.descend_with_path(root, &mut path, |internal| internal.search_child(key));
        let SearchResult::Found(idx) = self.nodes.get(leaf_handle).as_leaf().search(key) else {
            return None;
        };
        Some(self.remove_at(leaf_handle, idx, &mut path))
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
        // leaves sizes consistent. Rebalancing recomputes the sizes of the nodes it touches.
        self.adjust_sizes_along_path(path, |size| size - 1);
        if !self.nodes.get(leaf_handle).as_leaf().is_at_minimum() || path.is_empty() {
            // A stale separator is still a valid bound, so it's fine if this clone panics.
            self.update_separators_after_remove(leaf_handle, idx, path);
        } else {
            self.rebalance_leaf(leaf_handle, path);
        }

        (removed_key, removed_value)
    }

    /// Updates the separator bounding `leaf_handle` after removing the element at
    /// `removed_idx`. Separators equal the max key of their left subtree, so this is
    /// only needed when the leaf's last key was removed.
    fn update_separators_after_remove(&mut self, leaf_handle: Handle, removed_idx: usize, path: &Path) {
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        if path.is_empty() || removed_idx != leaf.key_count() {
            return;
        }
        let Some(last_key) = leaf.last_key() else {
            return;
        };
        let last_key = last_key.clone();

        for elem in path.iter().rev() {
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            if elem.child_index < parent.key_count() {
                parent.set_key(elem.child_index, last_key);
                break;
            }
        }
    }

    /// Rebalances a leaf after a removal caused it to underflow.
    fn rebalance_leaf(&mut self, leaf_handle: Handle, path: &mut Path) {
        let parent_elem = path.last().unwrap();
        let parent_handle = parent_elem.node;
        let child_idx = parent_elem.child_index;

        let parent = self.nodes.get(parent_handle).as_internal();

        // Try to borrow from left sibling
        if child_idx > 0 {
            let left_sibling_handle = parent.child(child_idx - 1);
            let left_sibling = self.nodes.get(left_sibling_handle).as_leaf();
            if left_sibling.can_lend() {
                self.borrow_from_left_leaf(leaf_handle, left_sibling_handle, parent_handle, child_idx);
                // Update overall sizes along path (child sizes already updated in borrow)
                self.refresh_sizes(path);
                return;
            }
        }

        // Try to borrow from right sibling
        if child_idx < parent.child_count() - 1 {
            let right_sibling_handle = parent.child(child_idx + 1);
            let right_sibling = self.nodes.get(right_sibling_handle).as_leaf();
            if right_sibling.can_lend() {
                self.borrow_from_right_leaf(leaf_handle, right_sibling_handle, parent_handle, child_idx);
                // Update overall sizes along path (child sizes already updated in borrow)
                self.refresh_sizes(path);
                return;
            }
        }

        // Must merge
        if child_idx > 0 {
            // Merge with left sibling
            let left_sibling_handle = parent.child(child_idx - 1);
            self.merge_leaves(left_sibling_handle, leaf_handle, path, child_idx - 1);
        } else {
            // Merge with right sibling
            let right_sibling_handle = parent.child(child_idx + 1);
            self.merge_leaves(leaf_handle, right_sibling_handle, path, child_idx);
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

    /// Merges two leaf nodes.
    fn merge_leaves(&mut self, left_handle: Handle, right_handle: Handle, path: &mut Path, separator_idx: usize) {
        // Take the right leaf
        let right = match self.nodes.take(right_handle) {
            Node::Leaf(leaf) => leaf,
            Node::Internal(_) => panic!("expected leaf"),
        };

        // Merge into left
        let left = self.nodes.get_mut(left_handle).as_leaf_mut();
        left.merge_with_right(right);

        // Update next pointer of node after right (if any)
        if let Some(next_handle) = left.next() {
            self.nodes.get_mut(next_handle).as_leaf_mut().set_prev(Some(left_handle));
        }

        // Update last_leaf if needed
        if self.last_leaf == Some(right_handle) {
            self.last_leaf = Some(left_handle);
        }

        // Update first_leaf if needed
        if self.first_leaf == Some(right_handle) {
            self.first_leaf = Some(left_handle);
        }

        // Remove separator from parent and propagate
        self.remove_from_parent_and_propagate(path, separator_idx);
    }

    /// Removes a separator from a parent node and propagates rebalancing up.
    fn remove_from_parent_and_propagate(&mut self, path: &mut Path, separator_idx: usize) {
        let parent_elem = path.pop().unwrap();
        let parent_handle = parent_elem.node;

        // The removed child's node was already taken by the merge, so it isn't freed here.
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.remove_child(separator_idx);

        // Update the merged child's size in the parent (the left child absorbed the right)
        let merged_child_handle = parent.child(separator_idx);
        let merged_child_size = self.nodes.get(merged_child_handle).subtree_size();
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_child_size(separator_idx, merged_child_size);
        parent.update_size();

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

        if !parent.is_at_minimum() {
            self.refresh_sizes(path);
            return;
        }

        // Need to rebalance internal node
        self.rebalance_internal(parent_handle, path);
    }

    /// Rebalances an internal node after a child was removed.
    fn rebalance_internal(&mut self, node_handle: Handle, path: &mut Path) {
        let parent_elem = path.last().unwrap();
        let parent_handle = parent_elem.node;
        let child_idx = parent_elem.child_index;

        let parent = self.nodes.get(parent_handle).as_internal();

        // Try to borrow from left sibling
        if child_idx > 0 {
            let left_sibling_handle = parent.child(child_idx - 1);
            let left_sibling = self.nodes.get(left_sibling_handle).as_internal();
            if left_sibling.can_lend() {
                self.borrow_from_left_internal(node_handle, left_sibling_handle, parent_handle, child_idx);
                self.refresh_sizes(path);
                return;
            }
        }

        // Try to borrow from right sibling
        if child_idx < parent.child_count() - 1 {
            let right_sibling_handle = parent.child(child_idx + 1);
            let right_sibling = self.nodes.get(right_sibling_handle).as_internal();
            if right_sibling.can_lend() {
                self.borrow_from_right_internal(node_handle, right_sibling_handle, parent_handle, child_idx);
                self.refresh_sizes(path);
                return;
            }
        }

        // Must merge
        if child_idx > 0 {
            // Merge with left sibling
            let left_sibling_handle = parent.child(child_idx - 1);
            self.merge_internals(left_sibling_handle, node_handle, path, child_idx - 1);
        } else {
            // Merge with right sibling
            let right_sibling_handle = parent.child(child_idx + 1);
            self.merge_internals(node_handle, right_sibling_handle, path, child_idx);
        }
    }

    /// Borrows from left internal sibling.
    fn borrow_from_left_internal(
        &mut self,
        node_handle: Handle,
        left_handle: Handle,
        parent_handle: Handle,
        child_idx: usize,
    ) {
        // Get parent separator
        let parent = self.nodes.get(parent_handle).as_internal();
        let parent_sep = parent.key(child_idx - 1).clone();

        // Pop from left sibling
        let left = self.nodes.get_mut(left_handle).as_internal_mut();
        let (left_key, left_child, left_child_size) = left.pop_child().unwrap();
        left.update_size();
        let left_new_size = left.size();

        // Push to front of current node with parent separator
        let node = self.nodes.get_mut(node_handle).as_internal_mut();
        node.push_child_front(parent_sep, node.child(0), node.child_size(0));
        node.set_first_child(left_child, left_child_size);
        node.update_size();
        let node_new_size = node.size();

        // Update parent separator and child sizes
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_key(child_idx - 1, left_key);
        parent.set_child_size(child_idx - 1, left_new_size);
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
        // Get parent separator
        let parent = self.nodes.get(parent_handle).as_internal();
        let parent_sep = parent.key(child_idx).clone();

        // Pop from front of right sibling
        let right = self.nodes.get_mut(right_handle).as_internal_mut();
        let (right_key, right_child, right_child_size) = right.pop_child_front().unwrap();
        right.update_size();
        let right_new_size = right.size();

        // Push to end of current node with parent separator
        let node = self.nodes.get_mut(node_handle).as_internal_mut();
        node.push_child(parent_sep, right_child, right_child_size);
        node.update_size();
        let node_new_size = node.size();

        // Update parent separator and child sizes
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_key(child_idx, right_key);
        parent.set_child_size(child_idx, node_new_size);
        parent.set_child_size(child_idx + 1, right_new_size);
    }

    /// Merges two internal nodes.
    fn merge_internals(&mut self, left_handle: Handle, right_handle: Handle, path: &mut Path, separator_idx: usize) {
        let parent_elem = path.last().unwrap();
        let parent_handle = parent_elem.node;

        // Get separator from parent
        let parent = self.nodes.get(parent_handle).as_internal();
        let separator = parent.key(separator_idx).clone();

        // Take the right node
        let right = match self.nodes.take(right_handle) {
            Node::Internal(internal) => internal,
            Node::Leaf(_) => panic!("expected internal"),
        };

        // Merge into left
        let left = self.nodes.get_mut(left_handle).as_internal_mut();
        left.merge_with_right(separator, right);

        // Remove separator from parent and propagate
        self.remove_from_parent_and_propagate(path, separator_idx);
    }

    /// Returns the position of the element with the given rank.
    fn locate_rank(&self, rank: usize) -> Option<(Handle, usize)> {
        if rank >= self.len {
            return None;
        }

        let mut current = self.root?;
        let mut remaining = rank;

        while let Node::Internal(internal) = self.nodes.get(current) {
            // Find which child contains the rank
            let mut found = false;
            for i in 0..internal.child_count() {
                let child_size = internal.child_size(i).to_usize();
                if remaining < child_size {
                    current = internal.child(i);
                    found = true;
                    break;
                }
                remaining -= child_size;
            }
            // A real assert: in release builds a violated invariant would otherwise loop forever.
            assert!(
                found,
                "locate_rank: tree size invariant violated - rank {} not found in children (node size: {})",
                rank,
                internal.size().to_usize()
            );
        }

        Some((current, remaining))
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
            },
            _ => RangeCursor::default(),
        }
    }
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

                    all_leaves.push(handle);

                    let max_key = leaf.last_key().cloned();
                    (max_key, leaf.key_count())
                }
                Node::Internal(internal) => {
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
                    // (last_child_max), or fall back to the last separator key if rightmost child is empty
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

    #[test]
    fn empty_leaf_neighbors_cause_bounds_to_return_none() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        // Use many keys to ensure multiple leaves.
        for i in 0..4000 {
            tree.insert(i * 2, i);
        }
        tree.validate_invariants();

        // Collect the leaf chain.
        let mut leaves: Vec<Handle> = Vec::new();
        let mut current = tree.first_leaf().expect("non-empty tree has a first leaf");
        loop {
            leaves.push(current);
            let leaf = tree.node(current).as_leaf();
            match leaf.next() {
                Some(next) => current = next,
                None => break,
            }
        }
        assert!(leaves.len() >= 5, "expected multiple leaves");

        let mid_index = leaves.len() / 2;
        let mid = leaves[mid_index];
        let prev = leaves[mid_index - 1];
        let next = leaves[mid_index + 1];

        let mid_key = *tree.node(mid).as_leaf().key(0);

        // Create empty neighbors around `mid` to exercise empty-leaf fallbacks.
        let _ = tree.node_mut(prev).as_leaf_mut().take_all();
        let _ = tree.node_mut(mid).as_leaf_mut().take_all();
        let _ = tree.node_mut(next).as_leaf_mut().take_all();

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
        let _ = tree.node_mut(first).as_leaf_mut().take_all();
        let _ = tree.node_mut(last).as_leaf_mut().take_all();

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
        let (tree, _root, left, _right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        tree.validate_invariants();

        // Exact boundary: key is the first key of the right leaf.
        assert_eq!(tree.lower_bound_exclusive(&6), Some((left, 2)));
        // Between leaves: not found in right leaf at index 0, so we step to the previous leaf.
        assert_eq!(tree.lower_bound_exclusive(&5), Some((left, 2)));
    }

    #[test]
    fn upper_bound_returns_none_when_next_leaf_is_empty() {
        let (mut tree, _root, _left, right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        let _ = tree.node_mut(right).as_leaf_mut().take_all();

        // Found at the last index with an empty next leaf should return None.
        assert_eq!(tree.upper_bound(&4), None);
    }

    #[test]
    fn lower_bound_exclusive_returns_none_when_prev_leaf_is_empty() {
        let (mut tree, _root, left, _right) = make_two_leaf_tree(&[0, 2, 4], &[6, 8, 10]);
        let _ = tree.node_mut(left).as_leaf_mut().take_all();

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

        tree.remove_from_parent_and_propagate(&mut path, 1);

        let bottom_size = tree.nodes.get(bottom_handle).as_internal().size();
        let root_child_size = tree.nodes.get(root_handle).as_internal().child_size(0);
        assert_eq!(root_child_size, bottom_size);
    }

    #[test]
    fn merge_leaves_updates_first_leaf_when_right_was_first() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

        let left_handle = tree.nodes.alloc(Node::new_leaf());
        let right_handle = tree.nodes.alloc(Node::new_leaf());

        let left_value = tree.values.alloc(10);
        let right_value = tree.values.alloc(20);
        tree.nodes.get_mut(left_handle).as_leaf_mut().push(1, left_value);
        tree.nodes.get_mut(right_handle).as_leaf_mut().push(2, right_value);
        tree.nodes.get_mut(left_handle).as_leaf_mut().set_next(Some(right_handle));
        tree.nodes.get_mut(right_handle).as_leaf_mut().set_prev(Some(left_handle));

        let parent_handle = tree.nodes.alloc(Node::new_internal());
        {
            let parent = tree.nodes.get_mut(parent_handle).as_internal_mut();
            parent.set_first_child(left_handle, Size::from_usize(1));
            parent.push_child(1, right_handle, Size::from_usize(1));
            parent.update_size();
        }

        tree.root = Some(parent_handle);
        // Intentionally point first/last at the right leaf to exercise the update branches.
        tree.first_leaf = Some(right_handle);
        tree.last_leaf = Some(right_handle);
        tree.len = 2;

        let mut path: Path = SmallVec::new();
        path.push(PathElement {
            node: parent_handle,
            child_index: 0,
        });

        tree.merge_leaves(left_handle, right_handle, &mut path, 0);

        assert_eq!(tree.first_leaf, Some(left_handle));
        assert_eq!(tree.last_leaf, Some(left_handle));
        tree.validate_invariants();
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

    #[test]
    #[should_panic(expected = "expected leaf")]
    fn merge_leaves_panics_on_internal_right() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();
        tree.insert(1, 1);

        let left_leaf = tree.first_leaf().expect("leaf exists");
        let internal_handle = tree.nodes.alloc(Node::Internal(InternalNode::new()));
        let mut path: Path = SmallVec::new();
        tree.merge_leaves(left_leaf, internal_handle, &mut path, 0);
    }

    #[test]
    #[should_panic(expected = "expected internal")]
    fn merge_internals_panics_on_leaf_right() {
        let mut tree: RawOSBTreeMap<i32, i32> = RawOSBTreeMap::new();

        let parent_handle = tree.nodes.alloc(Node::Internal(InternalNode::new()));
        let left_handle = tree.nodes.alloc(Node::Internal(InternalNode::new()));
        let right_leaf_handle = tree.nodes.alloc(Node::Leaf(LeafNode::new()));

        {
            let parent = tree.nodes.get_mut(parent_handle).as_internal_mut();
            parent.set_first_child(left_handle, Size::from_usize(0));
            parent.push_child(1, right_leaf_handle, Size::from_usize(0));
        }

        let mut path: Path = SmallVec::new();
        path.push(PathElement {
            node: parent_handle,
            child_index: 0,
        });

        tree.merge_internals(left_handle, right_leaf_handle, &mut path, 0);
    }
}

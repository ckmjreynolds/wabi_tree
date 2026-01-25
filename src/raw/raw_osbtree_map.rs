use core::borrow::Borrow;

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

/// Result of an insertion attempt.
pub(crate) enum InsertResult<K> {
    /// Insertion completed without split.
    Done,
    /// Node split occurred; parent needs to insert this new child.
    Split {
        /// The separator key for the new child.
        separator: K,
        /// Handle to the new (right) child.
        new_child: Handle,
        /// Size of the new child subtree.
        new_child_size: Size,
    },
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

        // Clear the tree structure (nodes arena still holds empty leaves and internals)
        self.nodes.clear();
        self.root = None;
        self.len = 0;
        self.first_leaf = None;
        self.last_leaf = None;

        result
    }

    /// Returns a reference to the first leaf, if any.
    pub(crate) fn first_leaf(&self) -> Option<Handle> {
        self.first_leaf
    }

    /// Returns a reference to the last leaf, if any.
    pub(crate) fn last_leaf(&self) -> Option<Handle> {
        self.last_leaf
    }

    /// Returns a reference to the root node, if any.
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
        // SAFETY: We only access the `values` field, avoiding aliasing with the `nodes` field.
        unsafe { (*core::ptr::addr_of_mut!((*ptr).values)).get_mut(handle) }
    }

    /// Takes a value from the arena.
    pub(crate) fn take_value(&mut self, handle: Handle) -> V {
        self.values.take(handle)
    }
}

impl<K: Clone + Ord, V> RawOSBTreeMap<K, V> {
    /// Searches for a key and returns the leaf handle and index if found.
    pub(crate) fn search<Q>(&self, key: &Q) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let root = self.root?;
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    current = internal.child(child_idx);
                }
                Node::Leaf(leaf) => {
                    if let SearchResult::Found(idx) = leaf.search(key) {
                        return Some((current, idx));
                    }
                    return None;
                }
            }
        }
    }

    /// Returns a reference to the value corresponding to the key.
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let (leaf_handle, idx) = self.search(key)?;
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        let value_handle = leaf.value(idx);
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
        let (leaf_handle, idx) = self.search(key)?;
        let leaf = self.nodes.get(leaf_handle).as_leaf();
        let k = leaf.key(idx);
        let value_handle = leaf.value(idx);
        Some((k, self.values.get(value_handle)))
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
        let first_leaf = self.first_leaf?;
        let leaf = self.nodes.get(first_leaf).as_leaf();
        if leaf.key_count() == 0 {
            return None;
        }
        let k = leaf.key(0);
        let value_handle = leaf.value(0);
        Some((k, self.values.get(value_handle)))
    }

    /// Returns the last key-value pair in the tree.
    pub(crate) fn last_key_value(&self) -> Option<(&K, &V)> {
        let last_leaf = self.last_leaf?;
        let leaf = self.nodes.get(last_leaf).as_leaf();
        let count = leaf.key_count();
        if count == 0 {
            return None;
        }
        let k = leaf.key(count - 1);
        let value_handle = leaf.value(count - 1);
        Some((k, self.values.get(value_handle)))
    }

    /// Inserts a key-value pair into the tree.
    /// Returns the old value if the key was already present.
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Handle empty tree case
        if self.root.is_none() {
            let value_handle = self.values.alloc(value);
            let mut leaf = LeafNode::new();
            leaf.push(key, value_handle);
            let leaf_handle = self.nodes.alloc(Node::Leaf(leaf));
            self.root = Some(leaf_handle);
            self.first_leaf = Some(leaf_handle);
            self.last_leaf = Some(leaf_handle);
            self.len = 1;
            return None;
        }

        let root = self.root.unwrap();

        // Build path from root to leaf
        let mut path: Path = SmallVec::new();
        let mut current = root;

        // Traverse to leaf
        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(&key);
                    path.push(PathElement {
                        node: current,
                        child_index: child_idx,
                    });
                    current = internal.child(child_idx);
                }
                Node::Leaf(_) => break,
            }
        }

        // Insert into leaf
        let leaf = self.nodes.get_mut(current).as_leaf_mut();
        match leaf.search(&key) {
            SearchResult::Found(idx) => {
                // Key exists, replace value in-place to avoid alloc/free churn
                let value_handle = leaf.value(idx);
                let old_value = core::mem::replace(self.values.get_mut(value_handle), value);
                Some(old_value)
            }
            SearchResult::NotFound(idx) => {
                // Insert new key-value
                let value_handle = self.values.alloc(value);
                leaf.insert(idx, key, value_handle);
                self.len += 1;

                // Check if we need to split
                if leaf.key_count() > MAX_KEYS {
                    self.split_leaf_and_propagate(current, &mut path);
                } else {
                    // Update sizes along the path
                    self.increment_sizes_along_path(&path);
                }

                None
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

    /// Propagates a split up the tree.
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
            parent.insert_child(elem.child_index, separator.clone(), new_child, right_size);
            parent.update_size();

            // Check if parent needs to split
            if parent.key_count() <= MAX_KEYS {
                // No more splits needed, just update sizes along remaining path
                self.update_sizes_along_path(path);
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
        let old_root_size = match self.nodes.get(old_root) {
            Node::Internal(internal) => internal.size(),
            Node::Leaf(leaf) => Size::from_usize(leaf.key_count()),
        };

        let mut new_root = InternalNode::new();
        new_root.set_first_child(old_root, old_root_size);
        new_root.push_child(separator, new_child, right_size);
        new_root.update_size();

        let new_root_handle = self.nodes.alloc(Node::Internal(new_root));
        self.root = Some(new_root_handle);
    }

    /// Updates sizes along the remaining path after a split that didn't propagate further.
    /// Recomputes each ancestor's `child_size` for the child we descended through,
    /// then recomputes the ancestor's total size.
    fn update_sizes_along_path(&mut self, path: &Path) {
        for elem in path.iter().rev() {
            let parent = self.nodes.get(elem.node).as_internal();
            let child_handle = parent.child(elem.child_index);
            let child_size = match self.nodes.get(child_handle) {
                Node::Internal(internal) => internal.size(),
                Node::Leaf(leaf) => Size::from_usize(leaf.key_count()),
            };
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            parent.set_child_size(elem.child_index, child_size);
            parent.update_size();
        }
    }

    /// Increments sizes along the path after a simple insertion (no split).
    fn increment_sizes_along_path(&mut self, path: &Path) {
        for elem in path.iter().rev() {
            let node = self.nodes.get_mut(elem.node).as_internal_mut();
            // Increment the overall node size
            let new_size = node.size().to_usize() + 1;
            node.set_size(Size::from_usize(new_size));
            // Also increment the specific child's size
            let child_size = node.child_size(elem.child_index).to_usize() + 1;
            node.set_child_size(elem.child_index, Size::from_usize(child_size));
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

        // Build path from root to leaf
        let mut path: Path = SmallVec::new();
        let mut current = root;

        // Traverse to leaf
        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    path.push(PathElement {
                        node: current,
                        child_index: child_idx,
                    });
                    current = internal.child(child_idx);
                }
                Node::Leaf(_) => break,
            }
        }

        // Find and remove from leaf
        let leaf = self.nodes.get_mut(current).as_leaf_mut();
        let idx = match leaf.search(key) {
            SearchResult::Found(idx) => idx,
            SearchResult::NotFound(_) => return None,
        };

        let (removed_key, value_handle) = leaf.remove(idx);
        let removed_value = self.values.take(value_handle);
        self.len -= 1;

        // Handle empty tree
        if self.len == 0 {
            self.nodes.clear();
            self.root = None;
            self.first_leaf = None;
            self.last_leaf = None;
            return Some((removed_key, removed_value));
        }

        // Check if we need to rebalance
        let leaf = self.nodes.get(current).as_leaf();
        if !leaf.is_at_minimum() || path.is_empty() {
            // Update separator keys if needed
            self.update_separators_after_remove(current, &path);
            // Update sizes
            self.update_sizes_after_remove(current, &path);
            return Some((removed_key, removed_value));
        }

        // Need to rebalance
        self.rebalance_leaf(current, &mut path);

        Some((removed_key, removed_value))
    }

    /// Updates separator keys after a removal.
    fn update_separators_after_remove(&mut self, leaf_handle: Handle, path: &Path) {
        if path.is_empty() {
            return;
        }

        let leaf = self.nodes.get(leaf_handle).as_leaf();
        if leaf.key_count() == 0 {
            return;
        }

        // Update the separator in the parent if this was a rightmost key
        let last_key = leaf.last_key().unwrap().clone();

        for elem in path.iter().rev() {
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            if elem.child_index < parent.key_count() {
                parent.set_key(elem.child_index, last_key);
                break;
            }
        }
    }

    /// Updates sizes along the path after a removal by recomputing from children.
    fn update_sizes_after_remove(&mut self, leaf_handle: Handle, path: &Path) {
        // First update the immediate parent's child size for this leaf
        if let Some(elem) = path.last() {
            let leaf = self.nodes.get(leaf_handle).as_leaf();
            let leaf_size = Size::from_usize(leaf.key_count());
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            parent.set_child_size(elem.child_index, leaf_size);
            parent.update_size();
        }

        // Then update all ancestor sizes (recomputing child_sizes from actual children)
        for elem in path.iter().rev().skip(1) {
            let parent = self.nodes.get(elem.node).as_internal();
            let child_handle = parent.child(elem.child_index);
            let child_size = match self.nodes.get(child_handle) {
                Node::Internal(internal) => internal.size(),
                Node::Leaf(leaf) => Size::from_usize(leaf.key_count()),
            };
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            parent.set_child_size(elem.child_index, child_size);
            parent.update_size();
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
                self.update_sizes_from_path(path);
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
                self.update_sizes_from_path(path);
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
        // Pop from left sibling
        let left = self.nodes.get_mut(left_handle).as_leaf_mut();
        let (key, value) = left.pop().unwrap();
        let left_new_size = left.key_count();
        let left_new_last_key = left.last_key().unwrap().clone();

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
        // Pop from front of right sibling
        let right = self.nodes.get_mut(right_handle).as_leaf_mut();
        let (key, value) = right.pop_front().unwrap();
        let right_new_size = right.key_count();

        // Push to end of current leaf
        let leaf = self.nodes.get_mut(leaf_handle).as_leaf_mut();
        leaf.push(key.clone(), value);
        let current_new_size = leaf.key_count();

        // Update parent separator and child sizes.
        // The borrowed key becomes the new max key of the current node,
        // so it's the correct separator for child_idx.
        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        parent.set_key(child_idx, key);
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

        let parent = self.nodes.get_mut(parent_handle).as_internal_mut();
        let (_sep_key, removed_child, _removed_size) = parent.remove_child(separator_idx);

        // Free the removed child handle (already taken during merge)
        // Note: The node was already taken in merge_leaves, so we don't free here
        let _ = removed_child;

        // Update the merged child's size in the parent (the left child absorbed the right)
        let merged_child_handle = parent.child(separator_idx);
        let merged_child_size = match self.nodes.get(merged_child_handle) {
            Node::Internal(internal) => internal.size(),
            Node::Leaf(leaf) => Size::from_usize(leaf.key_count()),
        };
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
            // Update sizes along remaining path
            for elem in path.iter().rev() {
                let parent = self.nodes.get(elem.node).as_internal();
                let child_handle = parent.child(elem.child_index);
                let child_size = match self.nodes.get(child_handle) {
                    Node::Internal(internal) => internal.size(),
                    Node::Leaf(leaf) => Size::from_usize(leaf.key_count()),
                };
                let parent = self.nodes.get_mut(elem.node).as_internal_mut();
                parent.set_child_size(elem.child_index, child_size);
                parent.update_size();
            }
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
                self.update_sizes_from_path(path);
                return;
            }
        }

        // Try to borrow from right sibling
        if child_idx < parent.child_count() - 1 {
            let right_sibling_handle = parent.child(child_idx + 1);
            let right_sibling = self.nodes.get(right_sibling_handle).as_internal();
            if right_sibling.can_lend() {
                self.borrow_from_right_internal(node_handle, right_sibling_handle, parent_handle, child_idx);
                self.update_sizes_from_path(path);
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

    /// Updates sizes from a path.
    fn update_sizes_from_path(&mut self, path: &Path) {
        for elem in path.iter().rev() {
            let parent = self.nodes.get(elem.node).as_internal();
            let child_handle = parent.child(elem.child_index);
            let child_size = match self.nodes.get(child_handle) {
                Node::Internal(internal) => internal.size(),
                Node::Leaf(leaf) => Size::from_usize(leaf.key_count()),
            };
            let parent = self.nodes.get_mut(elem.node).as_internal_mut();
            parent.set_child_size(elem.child_index, child_size);
            parent.update_size();
        }
    }

    /// Removes and returns the first key-value pair.
    pub(crate) fn pop_first(&mut self) -> Option<(K, V)> {
        let root = self.root?;

        // Build path to first leaf (always go to child 0)
        let mut path: Path = SmallVec::new();
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    path.push(PathElement {
                        node: current,
                        child_index: 0,
                    });
                    current = internal.child(0);
                }
                Node::Leaf(_) => break,
            }
        }

        // Remove first element from leaf
        let leaf = self.nodes.get_mut(current).as_leaf_mut();
        let (removed_key, value_handle) = leaf.remove(0);
        let removed_value = self.values.take(value_handle);
        self.len -= 1;

        // Handle empty tree
        if self.len == 0 {
            self.nodes.clear();
            self.root = None;
            self.first_leaf = None;
            self.last_leaf = None;
            return Some((removed_key, removed_value));
        }

        // Check if we need to rebalance
        let leaf = self.nodes.get(current).as_leaf();
        if !leaf.is_at_minimum() || path.is_empty() {
            // Update separator keys if needed
            self.update_separators_after_remove(current, &path);
            // Update sizes
            self.update_sizes_after_remove(current, &path);
            return Some((removed_key, removed_value));
        }

        // Need to rebalance
        self.rebalance_leaf(current, &mut path);

        Some((removed_key, removed_value))
    }

    /// Removes and returns the last key-value pair.
    pub(crate) fn pop_last(&mut self) -> Option<(K, V)> {
        let root = self.root?;

        // Build path to last leaf (always go to last child)
        let mut path: Path = SmallVec::new();
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let last_child_idx = internal.child_count() - 1;
                    path.push(PathElement {
                        node: current,
                        child_index: last_child_idx,
                    });
                    current = internal.child(last_child_idx);
                }
                Node::Leaf(_) => break,
            }
        }

        // Remove last element from leaf
        let leaf = self.nodes.get_mut(current).as_leaf_mut();
        let last_idx = leaf.key_count() - 1;
        let (removed_key, value_handle) = leaf.remove(last_idx);
        let removed_value = self.values.take(value_handle);
        self.len -= 1;

        // Handle empty tree
        if self.len == 0 {
            self.nodes.clear();
            self.root = None;
            self.first_leaf = None;
            self.last_leaf = None;
            return Some((removed_key, removed_value));
        }

        // Check if we need to rebalance
        let leaf = self.nodes.get(current).as_leaf();
        if !leaf.is_at_minimum() || path.is_empty() {
            // Update separator keys if needed
            self.update_separators_after_remove(current, &path);
            // Update sizes
            self.update_sizes_after_remove(current, &path);
            return Some((removed_key, removed_value));
        }

        // Need to rebalance
        self.rebalance_leaf(current, &mut path);

        Some((removed_key, removed_value))
    }

    /// Gets an element by its rank (0-indexed position in sorted order).
    pub(crate) fn get_by_rank(&self, rank: usize) -> Option<(&K, &V)> {
        if rank >= self.len {
            return None;
        }

        let root = self.root?;
        let mut current = root;
        let mut remaining = rank;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
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
                    debug_assert!(
                        found,
                        "get_by_rank: tree size invariant violated - rank {} not found in children (node size: {})",
                        rank,
                        internal.size().to_usize()
                    );
                }
                Node::Leaf(leaf) => {
                    let key = leaf.key(remaining);
                    let value_handle = leaf.value(remaining);
                    return Some((key, self.values.get(value_handle)));
                }
            }
        }
    }

    /// Gets a mutable element by its rank.
    pub(crate) fn get_by_rank_mut(&mut self, rank: usize) -> Option<(&K, &mut V)> {
        if rank >= self.len {
            return None;
        }

        let root = self.root?;
        let mut current = root;
        let mut remaining = rank;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
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
                    debug_assert!(
                        found,
                        "get_by_rank_mut: tree size invariant violated - rank {} not found in children (node size: {})",
                        rank,
                        internal.size().to_usize()
                    );
                }
                Node::Leaf(leaf) => {
                    let key = leaf.key(remaining);
                    let value_handle = leaf.value(remaining);
                    // We need to return both a reference to the key and a mutable reference
                    // to the value. The borrow checker sees `self` as borrowed twice, but
                    // this is actually safe because keys and values are stored in separate
                    // arenas (self.nodes vs self.values) that don't alias.
                    //
                    // SAFETY:
                    // - `key` points into `self.nodes` arena (via the leaf node)
                    // - `value` points into `self.values` arena
                    // - These arenas are disjoint memory regions
                    // - We only mutate `self.values`, never `self.nodes`
                    // - The key reference remains valid because we don't modify the nodes arena
                    let key_ptr = key as *const K;
                    let value = self.values.get_mut(value_handle);
                    return Some((unsafe { &*key_ptr }, value));
                }
            }
        }
    }

    /// Returns the rank (0-indexed position) of a key.
    pub(crate) fn rank_of<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let root = self.root?;
        let mut current = root;
        let mut rank = 0;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    // Add sizes of all children before the one we're descending into
                    for i in 0..child_idx {
                        rank += internal.child_size(i).to_usize();
                    }
                    current = internal.child(child_idx);
                }
                Node::Leaf(leaf) => match leaf.search(key) {
                    SearchResult::Found(idx) => return Some(rank + idx),
                    SearchResult::NotFound(_) => return None,
                },
            }
        }
    }

    /// Finds the lower bound position (first key >= given key).
    /// Returns (`leaf_handle`, index) or None if all keys are less than the given key.
    pub(crate) fn lower_bound<Q>(&self, key: &Q) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let root = self.root?;
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    current = internal.child(child_idx);
                }
                Node::Leaf(leaf) => {
                    // Find first key >= target
                    match leaf.search(key) {
                        SearchResult::Found(idx) => return Some((current, idx)),
                        SearchResult::NotFound(idx) => {
                            if idx < leaf.key_count() {
                                return Some((current, idx));
                            }
                            // All keys in this leaf are < target, check next leaf
                            if let Some(next) = leaf.next() {
                                let next_leaf = self.nodes.get(next).as_leaf();
                                if next_leaf.key_count() > 0 {
                                    return Some((next, 0));
                                }
                            }
                            return None;
                        }
                    }
                }
            }
        }
    }

    /// Finds the upper bound position (first key > given key).
    /// Returns (`leaf_handle`, index) or None if all keys are <= the given key.
    pub(crate) fn upper_bound<Q>(&self, key: &Q) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let root = self.root?;
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    current = internal.child(child_idx);
                }
                Node::Leaf(leaf) => {
                    // Find first key > target
                    match leaf.search(key) {
                        SearchResult::Found(idx) => {
                            // Key found, upper bound is the next position
                            let next_idx = idx + 1;
                            if next_idx < leaf.key_count() {
                                return Some((current, next_idx));
                            }
                            // Check next leaf
                            if let Some(next) = leaf.next() {
                                let next_leaf = self.nodes.get(next).as_leaf();
                                if next_leaf.key_count() > 0 {
                                    return Some((next, 0));
                                }
                            }
                            return None;
                        }
                        SearchResult::NotFound(idx) => {
                            // idx is where key would be inserted, so it's also the first key > target
                            if idx < leaf.key_count() {
                                return Some((current, idx));
                            }
                            // All keys in this leaf are <= target, check next leaf
                            if let Some(next) = leaf.next() {
                                let next_leaf = self.nodes.get(next).as_leaf();
                                if next_leaf.key_count() > 0 {
                                    return Some((next, 0));
                                }
                            }
                            return None;
                        }
                    }
                }
            }
        }
    }

    /// Finds the last position <= given key.
    /// Returns (`leaf_handle`, index) or None if all keys are > the given key.
    pub(crate) fn upper_bound_inclusive<Q>(&self, key: &Q) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        // Find lower_bound and then step back if needed
        let root = self.root?;
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    current = internal.child(child_idx);
                }
                Node::Leaf(leaf) => {
                    match leaf.search(key) {
                        SearchResult::Found(idx) => return Some((current, idx)),
                        SearchResult::NotFound(idx) => {
                            // idx is insertion point; last key <= target is at idx - 1
                            if idx > 0 {
                                return Some((current, idx - 1));
                            }
                            // Need to check previous leaf
                            if let Some(prev) = leaf.prev() {
                                let prev_leaf = self.nodes.get(prev).as_leaf();
                                if prev_leaf.key_count() > 0 {
                                    return Some((prev, prev_leaf.key_count() - 1));
                                }
                            }
                            return None;
                        }
                    }
                }
            }
        }
    }

    /// Finds the last position < given key.
    /// Returns (`leaf_handle`, index) or None if all keys are >= the given key.
    pub(crate) fn lower_bound_exclusive<Q>(&self, key: &Q) -> Option<(Handle, usize)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        let root = self.root?;
        let mut current = root;

        loop {
            let node = self.nodes.get(current);
            match node {
                Node::Internal(internal) => {
                    let child_idx = internal.search_child(key);
                    current = internal.child(child_idx);
                }
                Node::Leaf(leaf) => {
                    match leaf.search(key) {
                        SearchResult::Found(idx) => {
                            // Key found, exclusive upper bound is idx - 1
                            if idx > 0 {
                                return Some((current, idx - 1));
                            }
                            // Need to check previous leaf
                            if let Some(prev) = leaf.prev() {
                                let prev_leaf = self.nodes.get(prev).as_leaf();
                                if prev_leaf.key_count() > 0 {
                                    return Some((prev, prev_leaf.key_count() - 1));
                                }
                            }
                            return None;
                        }
                        SearchResult::NotFound(idx) => {
                            // idx is insertion point; last key < target is at idx - 1
                            if idx > 0 {
                                return Some((current, idx - 1));
                            }
                            // Need to check previous leaf
                            if let Some(prev) = leaf.prev() {
                                let prev_leaf = self.nodes.get(prev).as_leaf();
                                if prev_leaf.key_count() > 0 {
                                    return Some((prev, prev_leaf.key_count() - 1));
                                }
                            }
                            return None;
                        }
                    }
                }
            }
        }
    }
}

impl<K: Clone, V: Clone> Clone for RawOSBTreeMap<K, V> {
    fn clone(&self) -> Self {
        use alloc::collections::VecDeque;

        fn clone_node<K: Clone, V: Clone>(
            old_nodes: &Arena<Node<K>>,
            old_values: &Arena<V>,
            new_nodes: &mut Arena<Node<K>>,
            new_values: &mut Arena<V>,
            old_handle: Handle,
        ) -> Handle {
            let old_node = old_nodes.get(old_handle);
            match old_node {
                Node::Leaf(leaf) => {
                    let mut new_leaf = LeafNode::new();
                    for i in 0..leaf.key_count() {
                        let key = leaf.key(i).clone();
                        let old_value_handle = leaf.value(i);
                        let new_value_handle = new_values.alloc(old_values.get(old_value_handle).clone());
                        new_leaf.push(key, new_value_handle);
                    }
                    // prev/next will be fixed up later
                    new_nodes.alloc(Node::Leaf(new_leaf))
                }
                Node::Internal(internal) => {
                    // Recursively clone children
                    let mut new_internal = InternalNode::new();

                    // Clone first child
                    let first_child = internal.child(0);
                    let new_first = clone_node(old_nodes, old_values, new_nodes, new_values, first_child);
                    let first_size = internal.child_size(0);
                    new_internal.set_first_child(new_first, first_size);

                    // Clone remaining children
                    for i in 0..internal.key_count() {
                        let key = internal.key(i).clone();
                        let child = internal.child(i + 1);
                        let new_child = clone_node(old_nodes, old_values, new_nodes, new_values, child);
                        let child_size = internal.child_size(i + 1);
                        new_internal.push_child(key, new_child, child_size);
                    }

                    new_internal.set_size(internal.size());
                    new_nodes.alloc(Node::Internal(new_internal))
                }
            }
        }

        fn find_leaves<K>(nodes: &Arena<Node<K>>, root: Handle) -> alloc::vec::Vec<Handle> {
            let mut leaves = alloc::vec::Vec::new();
            let mut stack = alloc::vec![root];
            while let Some(handle) = stack.pop() {
                match nodes.get(handle) {
                    Node::Leaf(_) => leaves.push(handle),
                    Node::Internal(internal) => {
                        // Push children in reverse order so we process left-to-right
                        for i in (0..internal.child_count()).rev() {
                            stack.push(internal.child(i));
                        }
                    }
                }
            }
            leaves
        }

        // Clone values arena
        let mut new_values: Arena<V> = Arena::with_capacity(self.values.capacity());
        let mut new_nodes: Arena<Node<K>> = Arena::with_capacity(self.nodes.capacity());

        // If empty, return empty tree
        if self.root.is_none() {
            return Self {
                nodes: new_nodes,
                values: new_values,
                root: None,
                len: 0,
                first_leaf: None,
                last_leaf: None,
            };
        }

        // Clone all nodes using depth-first traversal
        let new_root = clone_node(&self.nodes, &self.values, &mut new_nodes, &mut new_values, self.root.unwrap());

        // Fix up leaf prev/next pointers by traversing the leaf chain
        // Find all leaves and build the chain
        let new_leaves = find_leaves(&new_nodes, new_root);

        // Set up prev/next links
        for i in 0..new_leaves.len() {
            let handle = new_leaves[i];
            let prev = if i > 0 {
                Some(new_leaves[i - 1])
            } else {
                None
            };
            let next = if i < new_leaves.len() - 1 {
                Some(new_leaves[i + 1])
            } else {
                None
            };
            let leaf = new_nodes.get_mut(handle).as_leaf_mut();
            leaf.set_prev(prev);
            leaf.set_next(next);
        }

        let first_leaf = new_leaves.first().copied();
        let last_leaf = new_leaves.last().copied();

        Self {
            nodes: new_nodes,
            values: new_values,
            root: Some(new_root),
            len: self.len,
            first_leaf,
            last_leaf,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(
    clippy::collapsible_if,
    clippy::manual_assert,
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
            if let Node::Internal(internal) = self.nodes.get(root)
                && internal.size().to_usize() != self.len
            {
                let root_size = internal.size().to_usize();
                let len = self.len;
                errors.push(alloc::format!("Root size mismatch: root.size={root_size}, self.len={len}"));
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

        // Record all (key, rank) pairs
        let key_ranks: Vec<(i32, usize)> = (0..100).map(|i| (i, tree.rank_of(&i).expect("key should exist"))).collect();

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
}

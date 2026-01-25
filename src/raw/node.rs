use core::borrow::Borrow;

use smallvec::SmallVec;

use super::handle::Handle;
use super::size::Size;

#[cfg(test)]
pub(crate) const ORDER: usize = 16;
#[cfg(not(test))]
pub(crate) const ORDER: usize = 128;

pub(crate) const MAX_CHILDREN: usize = ORDER;
pub(crate) const MIN_CHILDREN: usize = ORDER.div_ceil(2);
pub(crate) const MAX_KEYS: usize = MAX_CHILDREN - 1;
pub(crate) const MIN_INTERNAL_KEYS: usize = MIN_CHILDREN - 1;
pub(crate) const MIN_LEAF_KEYS: usize = MAX_KEYS.div_ceil(2);

#[allow(private_interfaces)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Node<K> {
    Internal(InternalNode<K>),
    Leaf(LeafNode<K>),
}

// B+Tree: Internal nodes store separator keys and child handles.
pub(crate) struct InternalNode<K> {
    // The number of key/value pairs in the subtree rooted at this node.
    size: Size,
    // We define separator keys such that key[i] = child[i].max_key().
    // +1 allows for more ergonomic split operations.
    keys: SmallVec<[K; MAX_KEYS + 1]>,
    children: SmallVec<[Handle; MAX_CHILDREN + 1]>,
    // Sizes of each child subtree, for order-statistic operations.
    child_sizes: SmallVec<[Size; MAX_CHILDREN + 1]>,
}

// B+Tree: Leaf nodes store keys and value handles.
pub(crate) struct LeafNode<K> {
    prev: Option<Handle>,
    next: Option<Handle>,
    // +1 allows for more ergonomic split operations.
    keys: SmallVec<[K; MAX_KEYS + 1]>,
    values: SmallVec<[Handle; MAX_KEYS + 1]>,
}

/// Result of searching for a key in a node.
pub(crate) enum SearchResult {
    /// Key was found at the given index.
    Found(usize),
    /// Key was not found; index is where it would be inserted.
    NotFound(usize),
}

impl<K> Node<K> {
    /// Creates a new empty leaf node.
    pub(crate) fn new_leaf() -> Self {
        Node::Leaf(LeafNode::new())
    }

    /// Creates a new internal node with given children.
    pub(crate) fn new_internal() -> Self {
        Node::Internal(InternalNode::new())
    }

    /// Returns true if this is a leaf node.
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf(_))
    }

    /// Returns true if this is an internal node.
    pub(crate) fn is_internal(&self) -> bool {
        matches!(self, Node::Internal(_))
    }

    /// Returns the leaf node, panicking if this is not a leaf.
    pub(crate) fn as_leaf(&self) -> &LeafNode<K> {
        match self {
            Node::Leaf(leaf) => leaf,
            Node::Internal(_) => panic!("expected leaf node"),
        }
    }

    /// Returns the leaf node mutably, panicking if this is not a leaf.
    pub(crate) fn as_leaf_mut(&mut self) -> &mut LeafNode<K> {
        match self {
            Node::Leaf(leaf) => leaf,
            Node::Internal(_) => panic!("expected leaf node"),
        }
    }

    /// Returns the internal node, panicking if this is not internal.
    pub(crate) fn as_internal(&self) -> &InternalNode<K> {
        match self {
            Node::Internal(internal) => internal,
            Node::Leaf(_) => panic!("expected internal node"),
        }
    }

    /// Returns the internal node mutably, panicking if this is not internal.
    pub(crate) fn as_internal_mut(&mut self) -> &mut InternalNode<K> {
        match self {
            Node::Internal(internal) => internal,
            Node::Leaf(_) => panic!("expected internal node"),
        }
    }

    /// Returns the number of keys in this node.
    pub(crate) fn key_count(&self) -> usize {
        match self {
            Node::Internal(internal) => internal.key_count(),
            Node::Leaf(leaf) => leaf.key_count(),
        }
    }
}

impl<K> InternalNode<K> {
    /// Creates a new empty internal node.
    pub(crate) fn new() -> Self {
        Self {
            size: Size::ZERO,
            keys: SmallVec::new(),
            children: SmallVec::new(),
            child_sizes: SmallVec::new(),
        }
    }

    /// Returns the number of keys in this node.
    pub(crate) fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Returns the number of children in this node.
    pub(crate) fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Returns true if this node has room for more keys.
    pub(crate) fn has_room(&self) -> bool {
        self.keys.len() < MAX_KEYS
    }

    /// Returns true if this node is below minimum capacity and needs rebalancing.
    pub(crate) fn is_at_minimum(&self) -> bool {
        self.keys.len() < MIN_INTERNAL_KEYS
    }

    /// Returns true if this node can lend a key to a sibling.
    pub(crate) fn can_lend(&self) -> bool {
        self.keys.len() > MIN_INTERNAL_KEYS
    }

    /// Returns the total size (number of key/value pairs) in subtree.
    pub(crate) fn size(&self) -> Size {
        self.size
    }

    /// Sets the total size of this subtree.
    pub(crate) fn set_size(&mut self, size: Size) {
        self.size = size;
    }

    /// Recalculates and updates the size from child sizes.
    pub(crate) fn update_size(&mut self) {
        let total: usize = self.child_sizes.iter().map(|s| s.to_usize()).sum();
        self.size = Size::from_usize(total);
    }

    /// Returns the key at the given index.
    #[inline]
    pub(crate) fn key(&self, index: usize) -> &K {
        &self.keys[index]
    }

    /// Returns all keys.
    pub(crate) fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Returns the child handle at the given index.
    #[inline]
    pub(crate) fn child(&self, index: usize) -> Handle {
        self.children[index]
    }

    /// Returns all children.
    pub(crate) fn children(&self) -> &[Handle] {
        &self.children
    }

    /// Returns all children mutably.
    pub(crate) fn children_mut(&mut self) -> &mut SmallVec<[Handle; MAX_CHILDREN + 1]> {
        &mut self.children
    }

    /// Returns the size of the child at the given index.
    #[inline]
    pub(crate) fn child_size(&self, index: usize) -> Size {
        self.child_sizes[index]
    }

    /// Sets the size of the child at the given index.
    pub(crate) fn set_child_size(&mut self, index: usize, size: Size) {
        self.child_sizes[index] = size;
    }

    /// Returns all child sizes.
    pub(crate) fn child_sizes(&self) -> &[Size] {
        &self.child_sizes
    }

    /// Searches for the child index that might contain the given key.
    /// Returns the index of the child to descend into.
    #[inline]
    pub(crate) fn search_child<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        // Keys[i] is the max key in child[i], so we find first key >= search key.
        // Binary search returns Ok(i) if found, or Err(i) where i is insertion point.
        // In both cases, i is the correct child index.
        match self.keys.binary_search_by(|k| k.borrow().cmp(key)) {
            Ok(idx) | Err(idx) => idx,
        }
    }

    /// Inserts a key and child at the given position.
    pub(crate) fn insert_child(&mut self, index: usize, key: K, child: Handle, child_size: Size) {
        self.keys.insert(index, key);
        self.children.insert(index + 1, child);
        self.child_sizes.insert(index + 1, child_size);
    }

    /// Removes a key and child at the given position.
    /// Returns the removed key and child handle.
    pub(crate) fn remove_child(&mut self, index: usize) -> (K, Handle, Size) {
        let key = self.keys.remove(index);
        let child = self.children.remove(index + 1);
        let size = self.child_sizes.remove(index + 1);
        (key, child, size)
    }

    /// Pushes a key and child to the end.
    pub(crate) fn push_child(&mut self, key: K, child: Handle, child_size: Size) {
        self.keys.push(key);
        self.children.push(child);
        self.child_sizes.push(child_size);
    }

    /// Pushes a child to the front (only used during splits).
    pub(crate) fn push_child_front(&mut self, key: K, child: Handle, child_size: Size) {
        self.keys.insert(0, key);
        self.children.insert(0, child);
        self.child_sizes.insert(0, child_size);
    }

    /// Sets the first child (before any keys).
    pub(crate) fn set_first_child(&mut self, child: Handle, child_size: Size) {
        if self.children.is_empty() {
            self.children.push(child);
            self.child_sizes.push(child_size);
        } else {
            self.children[0] = child;
            self.child_sizes[0] = child_size;
        }
    }

    /// Updates a separator key at the given index.
    pub(crate) fn set_key(&mut self, index: usize, key: K) {
        self.keys[index] = key;
    }

    /// Pops the last key and child.
    pub(crate) fn pop_child(&mut self) -> Option<(K, Handle, Size)> {
        if self.keys.is_empty() {
            None
        } else {
            let key = self.keys.pop().unwrap();
            let child = self.children.pop().unwrap();
            let size = self.child_sizes.pop().unwrap();
            Some((key, child, size))
        }
    }

    /// Pops the first key and child (keeping the first child in place conceptually).
    pub(crate) fn pop_child_front(&mut self) -> Option<(K, Handle, Size)> {
        if self.keys.is_empty() {
            None
        } else {
            let key = self.keys.remove(0);
            let child = self.children.remove(0);
            let size = self.child_sizes.remove(0);
            Some((key, child, size))
        }
    }

    /// Splits this node at the midpoint. Returns (`median_key`, `new_node`).
    /// The new node contains the right half of the children.
    pub(crate) fn split(&mut self) -> (K, InternalNode<K>) {
        let mid = self.keys.len() / 2;

        // Create the right node with keys[mid+1..] and children[mid+1..]
        let mut right = InternalNode::new();

        // Move keys after median to right node
        right.keys = self.keys.drain(mid + 1..).collect();

        // Move children after median to right node
        right.children = self.children.drain(mid + 1..).collect();
        right.child_sizes = self.child_sizes.drain(mid + 1..).collect();

        // Pop the median key
        let median_key = self.keys.pop().unwrap();

        // Update sizes
        self.update_size();
        right.update_size();

        (median_key, right)
    }

    /// Merges with a right sibling, given the separator key from parent.
    pub(crate) fn merge_with_right(&mut self, separator: K, mut right: InternalNode<K>) {
        self.keys.push(separator);
        self.keys.append(&mut right.keys);
        self.children.append(&mut right.children);
        self.child_sizes.append(&mut right.child_sizes);
        self.update_size();
    }
}

impl<K> LeafNode<K> {
    /// Creates a new empty leaf node.
    pub(crate) fn new() -> Self {
        Self {
            prev: None,
            next: None,
            keys: SmallVec::new(),
            values: SmallVec::new(),
        }
    }

    /// Returns the number of keys in this node.
    pub(crate) fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Returns true if this node has room for more keys.
    pub(crate) fn has_room(&self) -> bool {
        self.keys.len() < MAX_KEYS
    }

    /// Returns true if this node is below minimum capacity and needs rebalancing.
    pub(crate) fn is_at_minimum(&self) -> bool {
        self.keys.len() < MIN_LEAF_KEYS
    }

    /// Returns true if this node can lend a key to a sibling without going below minimum.
    pub(crate) fn can_lend(&self) -> bool {
        self.keys.len() > MIN_LEAF_KEYS
    }

    /// Returns the previous leaf handle.
    pub(crate) fn prev(&self) -> Option<Handle> {
        self.prev
    }

    /// Sets the previous leaf handle.
    pub(crate) fn set_prev(&mut self, prev: Option<Handle>) {
        self.prev = prev;
    }

    /// Returns the next leaf handle.
    pub(crate) fn next(&self) -> Option<Handle> {
        self.next
    }

    /// Sets the next leaf handle.
    pub(crate) fn set_next(&mut self, next: Option<Handle>) {
        self.next = next;
    }

    /// Returns the key at the given index.
    #[inline]
    pub(crate) fn key(&self, index: usize) -> &K {
        &self.keys[index]
    }

    /// Returns all keys.
    pub(crate) fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Returns the value handle at the given index.
    #[inline]
    pub(crate) fn value(&self, index: usize) -> Handle {
        self.values[index]
    }

    /// Returns all value handles.
    pub(crate) fn values(&self) -> &[Handle] {
        &self.values
    }

    /// Returns the last key, if any.
    pub(crate) fn last_key(&self) -> Option<&K> {
        self.keys.last()
    }

    /// Searches for a key in this leaf.
    #[inline]
    pub(crate) fn search<Q>(&self, key: &Q) -> SearchResult
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        match self.keys.binary_search_by(|k| k.borrow().cmp(key)) {
            Ok(idx) => SearchResult::Found(idx),
            Err(idx) => SearchResult::NotFound(idx),
        }
    }

    /// Inserts a key and value at the given position.
    pub(crate) fn insert(&mut self, index: usize, key: K, value: Handle) {
        self.keys.insert(index, key);
        self.values.insert(index, value);
    }

    /// Removes the key and value at the given position.
    /// Returns the removed key and value handle.
    pub(crate) fn remove(&mut self, index: usize) -> (K, Handle) {
        let key = self.keys.remove(index);
        let value = self.values.remove(index);
        (key, value)
    }

    /// Sets the value handle at the given index.
    pub(crate) fn set_value(&mut self, index: usize, value: Handle) {
        self.values[index] = value;
    }

    /// Pushes a key and value to the end.
    pub(crate) fn push(&mut self, key: K, value: Handle) {
        self.keys.push(key);
        self.values.push(value);
    }

    /// Pushes a key and value to the front.
    pub(crate) fn push_front(&mut self, key: K, value: Handle) {
        self.keys.insert(0, key);
        self.values.insert(0, value);
    }

    /// Pops the last key and value.
    pub(crate) fn pop(&mut self) -> Option<(K, Handle)> {
        if self.keys.is_empty() {
            None
        } else {
            let key = self.keys.pop().unwrap();
            let value = self.values.pop().unwrap();
            Some((key, value))
        }
    }

    /// Pops the first key and value.
    pub(crate) fn pop_front(&mut self) -> Option<(K, Handle)> {
        if self.keys.is_empty() {
            None
        } else {
            let key = self.keys.remove(0);
            let value = self.values.remove(0);
            Some((key, value))
        }
    }

    /// Takes ownership of all keys and value handles, leaving the leaf empty.
    pub(crate) fn take_all(&mut self) -> (SmallVec<[K; MAX_KEYS + 1]>, SmallVec<[Handle; MAX_KEYS + 1]>) {
        let keys = core::mem::take(&mut self.keys);
        let values = core::mem::take(&mut self.values);
        (keys, values)
    }

    /// Splits this leaf at the midpoint. Returns (`split_key`, `new_node`).
    /// The `split_key` is the last key in the left (current) node.
    pub(crate) fn split(&mut self) -> (K, LeafNode<K>)
    where
        K: Clone,
    {
        let mid = self.keys.len() / 2;

        // Create the right node
        let mut right = LeafNode::new();

        // Move keys[mid..] and values[mid..] to right node
        right.keys = self.keys.drain(mid..).collect();
        right.values = self.values.drain(mid..).collect();

        // The split key is the last key in the left node (for B+tree separator)
        let split_key = self.keys.last().unwrap().clone();

        (split_key, right)
    }

    /// Merges with a right sibling.
    pub(crate) fn merge_with_right(&mut self, mut right: LeafNode<K>) {
        self.keys.append(&mut right.keys);
        self.values.append(&mut right.values);
        self.next = right.next;
    }
}

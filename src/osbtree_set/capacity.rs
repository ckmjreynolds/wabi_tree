use super::OSBTreeSet;
use crate::OSBTreeMap;

impl<T> OSBTreeSet<T> {
    /// Creates an empty set with capacity for at least `capacity` elements.
    ///
    /// This is an extension and is not part of the standard `BTreeSet` API.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set: OSBTreeSet<i32> = OSBTreeSet::with_capacity(16);
    /// assert!(set.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(capacity) for memory allocation.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        OSBTreeSet {
            map: OSBTreeMap::with_capacity(capacity),
        }
    }

    /// Returns the current capacity for the set.
    ///
    /// This is an extension and is not part of the standard `BTreeSet` API.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set: OSBTreeSet<i32> = OSBTreeSet::with_capacity(32);
    /// assert_eq!(set.capacity(), 32);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }
}

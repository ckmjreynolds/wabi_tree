use super::OSBTreeMap;
use crate::raw::RawOSBTreeMap;

impl<K, V> OSBTreeMap<K, V> {
    /// Creates an empty map with capacity for at least `capacity` elements.
    ///
    /// This is an extension and is not part of the standard `BTreeMap` API.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let map: OSBTreeMap<i32, i32> = OSBTreeMap::with_capacity(32);
    /// assert!(map.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(capacity) for memory allocation.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        OSBTreeMap {
            raw: RawOSBTreeMap::with_capacity(capacity),
        }
    }

    /// Returns the current capacity for the map.
    ///
    /// This is an extension and is not part of the standard `BTreeMap` API.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let map: OSBTreeMap<i32, i32> = OSBTreeMap::with_capacity(32);
    /// assert_eq!(map.capacity(), 32);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.raw.capacity()
    }
}

use core::borrow::Borrow;
use core::ops::{Index, IndexMut};

use super::OSBTreeMap;
use crate::Rank;

impl<K: Clone + Ord, V> OSBTreeMap<K, V> {
    /// Returns the key-value pair at position `rank` in sorted order.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API.
    ///
    /// The rank is zero-based. Returns `None` if `rank` is out of bounds.
    ///
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert("a", 10);
    /// map.insert("c", 30);
    /// map.insert("b", 20);
    ///
    /// let (key, value) = map.get_by_rank(1).unwrap();
    /// assert_eq!((key, value), (&"b", &20));
    /// assert!(map.get_by_rank(3).is_none());
    /// ```
    #[must_use]
    pub fn get_by_rank(&self, rank: usize) -> Option<(&K, &V)> {
        self.raw.get_by_rank(rank)
    }

    /// Returns the key and a mutable reference to the value at position `rank`
    /// in sorted order.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API.
    ///
    /// The rank is zero-based. Returns `None` if `rank` is out of bounds.
    /// The key is returned as a shared reference because mutating it would
    /// violate the map's ordering invariants.
    ///
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(10, "a");
    /// map.insert(5, "b");
    ///
    /// if let Some((key, value)) = map.get_by_rank_mut(0) {
    ///     assert_eq!(*key, 5);
    ///     *value = "updated";
    /// }
    ///
    /// assert_eq!(map.get(&5), Some(&"updated"));
    /// ```
    #[must_use]
    pub fn get_by_rank_mut(&mut self, rank: usize) -> Option<(&K, &mut V)> {
        self.raw.get_by_rank_mut(rank)
    }

    /// Returns the zero-based rank of `key` in sorted order, or `None` if the
    /// key is not present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API.
    ///
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(10, "a");
    /// map.insert(20, "b");
    ///
    /// assert_eq!(map.rank_of(&10), Some(0));
    /// assert_eq!(map.rank_of(&15), None);
    /// ```
    #[must_use]
    pub fn rank_of<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.raw.rank_of(key)
    }

    /// Returns the zero-based rank of `key` in sorted order together with the
    /// stored key and value, or `None` if the key is not present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API. It finds the element and its rank in one descent, where
    /// [`rank_of`](Self::rank_of) followed by
    /// [`get_key_value`](Self::get_key_value) takes two.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let map = OSBTreeMap::from([("a", 10), ("b", 20), ("c", 30)]);
    ///
    /// assert_eq!(map.get_with_rank(&"b"), Some((1, &"b", &20)));
    /// assert_eq!(map.get_with_rank(&"z"), None);
    /// ```
    #[must_use]
    pub fn get_with_rank<Q>(&self, key: &Q) -> Option<(usize, &K, &V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.raw.get_with_rank(key)
    }

    /// Returns the zero-based rank of `key` in sorted order together with the
    /// stored key and a mutable reference to the value, or `None` if the key
    /// is not present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API. The key is returned as a shared reference because
    /// mutating it would violate the map's ordering invariants.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::from([("a", 10), ("b", 20)]);
    ///
    /// if let Some((rank, _, value)) = map.get_with_rank_mut(&"b") {
    ///     assert_eq!(rank, 1);
    ///     *value += 1;
    /// }
    /// assert_eq!(map[&"b"], 21);
    /// ```
    #[must_use]
    pub fn get_with_rank_mut<Q>(&mut self, key: &Q) -> Option<(usize, &K, &mut V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.raw.get_with_rank_mut(key)
    }

    /// Inserts a key-value pair into the map and returns the key's zero-based
    /// rank after the insertion, together with the old value if the key was
    /// already present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API. Apart from the rank it behaves like
    /// [`insert`](Self::insert): if the key is present, its value is replaced
    /// and the key itself is not updated.
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::from([("a", 10), ("c", 30)]);
    ///
    /// assert_eq!(map.insert_with_rank("b", 20), (1, None));
    /// assert_eq!(map.insert_with_rank("c", 31), (2, Some(30)));
    /// ```
    pub fn insert_with_rank(&mut self, key: K, value: V) -> (usize, Option<V>) {
        self.raw.insert_with_rank(key, value)
    }

    /// Removes a key from the map, returning its zero-based rank before the
    /// removal together with the stored key and value, or `None` if the key
    /// is not present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API. It finds the element and its rank in one descent, where
    /// [`rank_of`](Self::rank_of) followed by
    /// [`remove_entry`](Self::remove_entry) takes two.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::from([("a", 10), ("b", 20), ("c", 30)]);
    ///
    /// assert_eq!(map.remove_with_rank(&"b"), Some((1, "b", 20)));
    /// assert_eq!(map.remove_with_rank(&"b"), None);
    /// assert_eq!(map.len(), 2);
    /// ```
    pub fn remove_with_rank<Q>(&mut self, key: &Q) -> Option<(usize, K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.raw.remove_entry_with_rank(key)
    }

    /// Removes and returns the key-value pair at position `rank` in sorted
    /// order.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeMap` API.
    ///
    /// The rank is zero-based. Returns `None` if `rank` is out of bounds,
    /// where `Vec::remove` would panic.
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::from([("a", 10), ("b", 20), ("c", 30)]);
    ///
    /// assert_eq!(map.remove_by_rank(1), Some(("b", 20)));
    /// assert_eq!(map.remove_by_rank(2), None);
    /// assert_eq!(map.len(), 2);
    /// ```
    pub fn remove_by_rank(&mut self, rank: usize) -> Option<(K, V)> {
        self.raw.remove_by_rank(rank)
    }
}
/// Indexes into the map by rank.
///
/// # Panics
///
/// Panics if `rank` is out of bounds.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
/// use wabi_tree::Rank;
///
/// let mut map = OSBTreeMap::new();
/// map.insert("a", 1);
/// map.insert("b", 2);
///
/// assert_eq!(map[Rank(0)], 1);
/// ```
impl<K: Clone + Ord, V> Index<Rank> for OSBTreeMap<K, V> {
    type Output = V;

    fn index(&self, rank: Rank) -> &Self::Output {
        self.get_by_rank(rank.0).map(|(_, v)| v).expect("index out of bounds")
    }
}
/// Mutably indexes into the map by rank.
///
/// # Panics
///
/// Panics if `rank` is out of bounds.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
/// use wabi_tree::Rank;
///
/// let mut map = OSBTreeMap::from([("a", 1), ("b", 2)]);
/// map[Rank(1)] = 5;
///
/// assert_eq!(map.get(&"b"), Some(&5));
/// ```
impl<K: Clone + Ord, V> IndexMut<Rank> for OSBTreeMap<K, V> {
    fn index_mut(&mut self, rank: Rank) -> &mut Self::Output {
        self.get_by_rank_mut(rank.0).map(|(_, v)| v).expect("index out of bounds")
    }
}

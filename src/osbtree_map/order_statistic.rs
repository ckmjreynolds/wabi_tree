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

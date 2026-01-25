use core::borrow::Borrow;
use core::ops::Index;

use super::OSBTreeSet;
use crate::Rank;

impl<T: Clone + Ord> OSBTreeSet<T> {
    /// Returns the value at position `rank` in sorted order.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API.
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
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set = OSBTreeSet::from([10, 20, 30]);
    /// assert_eq!(set.get_by_rank(1), Some(&20));
    /// assert!(set.get_by_rank(3).is_none());
    /// ```
    #[must_use]
    pub fn get_by_rank(&self, rank: usize) -> Option<&T> {
        self.map.get_by_rank(rank).map(|(k, ())| k)
    }

    /// Returns the zero-based rank of `value` in sorted order, or `None` if
    /// the value is not present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API.
    ///
    ///
    /// # Complexity
    ///
    /// O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set = OSBTreeSet::from([10, 20]);
    ///
    /// assert_eq!(set.rank_of(&20), Some(1));
    /// assert_eq!(set.rank_of(&15), None);
    /// ```
    #[must_use]
    pub fn rank_of<Q>(&self, value: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.map.rank_of(value)
    }
}

/// Indexes into the set by rank.
///
/// # Panics
///
/// Panics if `rank` is out of bounds.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
/// use wabi_tree::Rank;
///
/// let set = OSBTreeSet::from([10, 20, 30]);
/// assert_eq!(set[Rank(1)], 20);
/// ```
impl<T: Clone + Ord> Index<Rank> for OSBTreeSet<T> {
    type Output = T;

    fn index(&self, rank: Rank) -> &Self::Output {
        self.get_by_rank(rank.0).expect("index out of bounds")
    }
}

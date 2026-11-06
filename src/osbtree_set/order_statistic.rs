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

    /// Returns the zero-based rank of `value` in sorted order together with
    /// the stored element, or `None` if the value is not present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API. It finds the element and its rank in one descent, where
    /// [`rank_of`](Self::rank_of) followed by [`get`](Self::get) takes two.
    ///
    /// The value may be any borrowed form of the set's element type, but the
    /// ordering on the borrowed form *must* match the ordering on the element
    /// type.
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
    ///
    /// assert_eq!(set.get_with_rank(&20), Some((1, &20)));
    /// assert_eq!(set.get_with_rank(&15), None);
    /// ```
    #[must_use]
    pub fn get_with_rank<Q>(&self, value: &Q) -> Option<(usize, &T)>
    where
        T: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.map.get_with_rank(value).map(|(rank, k, ())| (rank, k))
    }

    /// Adds a value to the set and returns its zero-based rank after the
    /// insertion, together with whether it was newly inserted.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API. Apart from the rank it behaves like
    /// [`insert`](Self::insert): if an equal element is present, the set is not
    /// modified and the rank is that element's.
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
    /// let mut set = OSBTreeSet::from([10, 30]);
    ///
    /// assert_eq!(set.insert_with_rank(20), (1, true));
    /// assert_eq!(set.insert_with_rank(30), (2, false));
    /// ```
    pub fn insert_with_rank(&mut self, value: T) -> (usize, bool) {
        let (rank, old) = self.map.insert_with_rank(value, ());
        (rank, old.is_none())
    }

    /// If the set contains an element equal to `value`, removes and drops it
    /// and returns its zero-based rank before the removal. Returns `None` if
    /// no such element was present.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API.
    ///
    /// The value may be any borrowed form of the set's element type, but the
    /// ordering on the borrowed form *must* match the ordering on the element
    /// type.
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
    /// let mut set = OSBTreeSet::from([10, 20, 30]);
    ///
    /// assert_eq!(set.remove_with_rank(&20), Some(1));
    /// assert_eq!(set.remove_with_rank(&20), None);
    /// ```
    pub fn remove_with_rank<Q>(&mut self, value: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.map.remove_with_rank(value).map(|(rank, _, ())| rank)
    }

    /// Removes and returns the element equal to `value`, if any, together with
    /// its zero-based rank before the removal.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API.
    ///
    /// The value may be any borrowed form of the set's element type, but the
    /// ordering on the borrowed form *must* match the ordering on the element
    /// type.
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
    /// let mut set = OSBTreeSet::from([10, 20, 30]);
    ///
    /// assert_eq!(set.take_with_rank(&30), Some((2, 30)));
    /// assert_eq!(set.take_with_rank(&30), None);
    /// ```
    pub fn take_with_rank<Q>(&mut self, value: &Q) -> Option<(usize, T)>
    where
        T: Borrow<Q>,
        Q: ?Sized + Ord,
    {
        self.map.remove_with_rank(value).map(|(rank, k, ())| (rank, k))
    }

    /// Removes and returns the element at position `rank` in sorted order.
    ///
    /// This is an order-statistic extension and is not part of the standard
    /// `BTreeSet` API.
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
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::from([10, 20, 30]);
    ///
    /// assert_eq!(set.remove_by_rank(0), Some(10));
    /// assert_eq!(set.remove_by_rank(2), None);
    /// assert_eq!(set.len(), 2);
    /// ```
    pub fn remove_by_rank(&mut self, rank: usize) -> Option<T> {
        self.map.remove_by_rank(rank).map(|(k, ())| k)
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

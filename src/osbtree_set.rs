use core::borrow::Borrow;
use core::cmp::Ordering::{self, Equal, Greater, Less};
use core::cmp::{max, min};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{FusedIterator, Peekable};
use core::marker::PhantomData;
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};

use crate::OSBTreeMap;
use crate::osbtree_map::{ExtractIfInner, IntoKeys, Keys, Range as MapRange};

mod capacity;
mod order_statistic;

/// An ordered set based on a B-Tree.
///
/// See [`OSBTreeMap`]'s documentation for a detailed discussion of this collection's performance
/// benefits and drawbacks.
///
/// It is a logic error for an item to be modified in such a way that the item's ordering relative
/// to any other item, as determined by the [`Ord`] trait, changes while it is in the set. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
/// The behavior resulting from such a logic error is not specified, but will be encapsulated to the
/// `OSBTreeSet` that observed the logic error and not result in undefined behavior. This could
/// include panics, incorrect results, aborts, memory leaks, and non-termination.
///
/// Iterators returned by [`OSBTreeSet::iter`] and [`OSBTreeSet::into_iter`] produce their items in
/// order, and take worst-case logarithmic and amortized constant time per item returned.
///
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `OSBTreeSet<&str>` in this example).
/// let mut books = OSBTreeSet::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons");
/// books.insert("To Kill a Mockingbird");
/// books.insert("The Odyssey");
/// books.insert("The Great Gatsby");
///
/// // Check for a specific one.
/// if !books.contains("The Winds of Winter") {
///     println!("We have {} books, but The Winds of Winter ain't one.",
///              books.len());
/// }
///
/// // Remove a book.
/// books.remove("The Odyssey");
///
/// // Iterate over everything.
/// for book in &books {
///     println!("{book}");
/// }
/// ```
///
/// A `OSBTreeSet` with a known list of items can be initialized from an array:
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let set = OSBTreeSet::from([1, 2, 3]);
/// ```
pub struct OSBTreeSet<T> {
    map: OSBTreeMap<T, ()>,
}

/// An iterator over the items of a `OSBTreeSet`.
///
/// This `struct` is created by the [`iter`] method on [`OSBTreeSet`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let set = OSBTreeSet::from([3, 1, 2]);
/// let mut iter = set.iter();
/// assert_eq!(iter.next(), Some(&1));
/// assert_eq!(iter.next_back(), Some(&3));
/// assert_eq!(iter.next(), Some(&2));
/// ```
///
/// [`iter`]: OSBTreeSet::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, T: 'a> {
    inner: Keys<'a, T, ()>,
}

/// An owning iterator over the items of a `OSBTreeSet` in ascending order.
///
/// This `struct` is created by the [`into_iter`] method on [`OSBTreeSet`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let set = OSBTreeSet::from([1, 2, 3]);
/// let mut iter = set.into_iter();
/// assert_eq!(iter.next(), Some(1));
/// assert_eq!(iter.next_back(), Some(3));
/// assert_eq!(iter.next(), Some(2));
/// ```
///
/// [`into_iter`]: OSBTreeSet#method.into_iter
pub struct IntoIter<T> {
    inner: IntoKeys<T, ()>,
}

/// An iterator over a sub-range of items in a `OSBTreeSet`.
///
/// This `struct` is created by the [`range`] method on [`OSBTreeSet`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let set = OSBTreeSet::from([1, 2, 3, 4]);
/// let mut range = set.range(2..=3);
/// assert_eq!(range.next(), Some(&2));
/// assert_eq!(range.next_back(), Some(&3));
/// assert_eq!(range.next(), None);
/// ```
///
/// [`range`]: OSBTreeSet::range
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Range<'a, T: 'a> {
    inner: MapRange<'a, T, ()>,
}

/// A lazy iterator producing elements in the difference of `OSBTreeSet`s.
///
/// This `struct` is created by the [`difference`] method on [`OSBTreeSet`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let a = OSBTreeSet::from([1, 2, 3]);
/// let b = OSBTreeSet::from([2]);
/// let diff: Vec<_> = a.difference(&b).copied().collect();
/// assert_eq!(diff, [1, 3]);
/// ```
///
/// [`difference`]: OSBTreeSet::difference
#[must_use = "this returns the difference as an iterator, \
              without modifying either input set"]
pub struct Difference<'a, T: 'a> {
    inner: DifferenceInner<'a, T>,
}

/// A lazy iterator producing elements in the symmetric difference of `OSBTreeSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on [`OSBTreeSet`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let a = OSBTreeSet::from([1, 2, 3]);
/// let b = OSBTreeSet::from([3, 4]);
/// let sym: Vec<_> = a.symmetric_difference(&b).copied().collect();
/// assert_eq!(sym, [1, 2, 4]);
/// ```
///
/// [`symmetric_difference`]: OSBTreeSet::symmetric_difference
#[must_use = "this returns the symmetric difference as an iterator, \
              without modifying either input set"]
pub struct SymmetricDifference<'a, T: 'a> {
    inner: MergeIterInner<Iter<'a, T>>,
}

/// A lazy iterator producing elements in the intersection of `OSBTreeSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`OSBTreeSet`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let a = OSBTreeSet::from([1, 2, 3]);
/// let b = OSBTreeSet::from([2, 4]);
/// let inter: Vec<_> = a.intersection(&b).copied().collect();
/// assert_eq!(inter, [2]);
/// ```
///
/// [`intersection`]: OSBTreeSet::intersection
#[must_use = "this returns the intersection as an iterator, \
              without modifying either input set"]
pub struct Intersection<'a, T: 'a> {
    inner: IntersectionInner<'a, T>,
}

/// A lazy iterator producing elements in the union of `OSBTreeSet`s.
///
/// This `struct` is created by the [`union`] method on [`OSBTreeSet`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let a = OSBTreeSet::from([1, 2]);
/// let b = OSBTreeSet::from([2, 3]);
/// let union: Vec<_> = a.union(&b).copied().collect();
/// assert_eq!(union, [1, 2, 3]);
/// ```
///
/// [`union`]: OSBTreeSet::union
#[must_use = "this returns the union as an iterator, \
              without modifying either input set"]
pub struct Union<'a, T: 'a> {
    inner: MergeIterInner<Iter<'a, T>>,
}

/// An iterator produced by calling `extract_if` on `OSBTreeSet`.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeSet;
///
/// let mut set = OSBTreeSet::from([1, 2, 3, 4]);
/// let extracted: Vec<_> = set.extract_if(.., |v| v % 2 == 0).collect();
/// assert_eq!(extracted, [2, 4]);
/// assert_eq!(set.iter().copied().collect::<Vec<_>>(), [1, 3]);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<'a, T, R, F> {
    pred: F,
    inner: ExtractIfInner<'a, T, (), R>,
}

enum DifferenceInner<'a, T: 'a> {
    Stitch {
        self_iter: Iter<'a, T>,
        other_iter: Peekable<Iter<'a, T>>,
    },
    Search {
        self_iter: Iter<'a, T>,
        other_set: &'a OSBTreeSet<T>,
    },
    Iterate(Iter<'a, T>),
}

enum IntersectionInner<'a, T: 'a> {
    Stitch {
        a: Iter<'a, T>,
        b: Iter<'a, T>,
    },
    Search {
        small_iter: Iter<'a, T>,
        large_set: &'a OSBTreeSet<T>,
    },
    Answer(Option<&'a T>),
}

const ITER_PERFORMANCE_TIPPING_SIZE_DIFF: usize = 16;

struct MergeIterInner<I: Iterator> {
    a: I,
    b: I,
    peeked: Option<Peeked<I>>,
}

#[derive(Clone, Debug)]
enum Peeked<I: Iterator> {
    A(I::Item),
    B(I::Item),
}

impl<I: Iterator> Clone for MergeIterInner<I>
where
    I: Clone,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        MergeIterInner {
            a: self.a.clone(),
            b: self.b.clone(),
            peeked: self.peeked.clone(),
        }
    }
}

impl<I: Iterator> fmt::Debug for MergeIterInner<I>
where
    I: fmt::Debug,
    I::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MergeIterInner").field("a", &self.a).field("b", &self.b).field("peeked", &self.peeked).finish()
    }
}

impl<T: fmt::Debug> fmt::Debug for DifferenceInner<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => f.debug_struct("Stitch").field("self_iter", self_iter).field("other_iter", other_iter).finish(),
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => f.debug_struct("Search").field("self_iter", self_iter).field("other_set", other_set).finish(),
            DifferenceInner::Iterate(iter) => f.debug_tuple("Iterate").field(iter).finish(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for IntersectionInner<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntersectionInner::Stitch {
                a,
                b,
            } => f.debug_struct("Stitch").field("a", a).field("b", b).finish(),
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => f.debug_struct("Search").field("small_iter", small_iter).field("large_set", large_set).finish(),
            IntersectionInner::Answer(answer) => f.debug_tuple("Answer").field(answer).finish(),
        }
    }
}

impl<I: Iterator> MergeIterInner<I> {
    fn new(a: I, b: I) -> Self {
        MergeIterInner {
            a,
            b,
            peeked: None,
        }
    }

    /// Returns the next pair of items from both iterators based on comparison.
    /// For union: returns (Some(a), None) if a < b, (None, Some(b)) if b < a, (Some(a), Some(b)) if equal
    /// For `symmetric_difference`: returns (Some(a), None) if a < b, (None, Some(b)) if b < a, skips both if equal
    fn nexts<Cmp: Fn(&I::Item, &I::Item) -> Ordering>(&mut self, cmp: Cmp) -> (Option<I::Item>, Option<I::Item>)
    where
        I: FusedIterator,
    {
        let a_next = match self.peeked.take() {
            Some(Peeked::A(a)) => Some(a),
            Some(Peeked::B(b)) => {
                self.peeked = Some(Peeked::B(b));
                self.a.next()
            }
            None => self.a.next(),
        };
        let b_next = match self.peeked.take() {
            Some(Peeked::B(b)) => Some(b),
            Some(Peeked::A(a)) => {
                self.peeked = Some(Peeked::A(a));
                self.b.next()
            }
            None => self.b.next(),
        };

        match (a_next, b_next) {
            (None, None) => (None, None),
            (Some(a), None) => (Some(a), None),
            (None, Some(b)) => (None, Some(b)),
            (Some(a), Some(b)) => match cmp(&a, &b) {
                Less => {
                    self.peeked = Some(Peeked::B(b));
                    (Some(a), None)
                }
                Greater => {
                    self.peeked = Some(Peeked::A(a));
                    (None, Some(b))
                }
                Equal => (Some(a), Some(b)),
            },
        }
    }

    fn lens(&self) -> (usize, usize)
    where
        I: ExactSizeIterator,
    {
        match &self.peeked {
            Some(Peeked::A(_)) => (1 + self.a.len(), self.b.len()),
            Some(Peeked::B(_)) => (self.a.len(), 1 + self.b.len()),
            None => (self.a.len(), self.b.len()),
        }
    }
}

impl<T> OSBTreeSet<T> {
    /// Makes a new, empty `OSBTreeSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    ///
    /// // entries can now be inserted into the empty set
    /// set.insert(1);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub const fn new() -> OSBTreeSet<T> {
        OSBTreeSet {
            map: OSBTreeMap::new(),
        }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the set.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::ops::Bound::Included;
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(3);
    /// set.insert(5);
    /// set.insert(8);
    /// for &elem in set.range((Included(&4), Included(&8))) {
    ///     println!("{elem}");
    /// }
    /// assert_eq!(Some(&5), set.range(4..).next());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) to create the iterator; each iteration step is O(1) amortized.
    pub fn range<K, R>(&self, range: R) -> Range<'_, T>
    where
        K: ?Sized + Ord,
        T: Borrow<K> + Ord + Clone,
        R: RangeBounds<K>,
    {
        Range {
            inner: self.map.range(range),
        }
    }

    /// Visits the elements representing the difference,
    /// i.e., the elements that are in `self` but not in `other`,
    /// in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = OSBTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let diff: Vec<_> = a.difference(&b).cloned().collect();
    /// assert_eq!(diff, [1]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(min(m, n)) for stitch iteration or O(m log n) for search-based iteration,
    /// where m and n are the sizes of the two sets.
    pub fn difference<'a>(&'a self, other: &'a OSBTreeSet<T>) -> Difference<'a, T>
    where
        T: Ord + Clone,
    {
        let (Some(self_min), Some(self_max)) = (self.first(), self.last()) else {
            return Difference {
                inner: DifferenceInner::Iterate(self.iter()),
            };
        };

        let (Some(other_min), Some(other_max)) = (other.first(), other.last()) else {
            return Difference {
                inner: DifferenceInner::Iterate(self.iter()),
            };
        };

        // Check for disjoint sets
        if self_max < other_min || other_max < self_min {
            return Difference {
                inner: DifferenceInner::Iterate(self.iter()),
            };
        }

        // Choose algorithm based on size difference
        let self_len = self.len();
        let other_len = other.len();

        if other_len > ITER_PERFORMANCE_TIPPING_SIZE_DIFF * self_len {
            // other is much larger, iterate self and search in other
            Difference {
                inner: DifferenceInner::Search {
                    self_iter: self.iter(),
                    other_set: other,
                },
            }
        } else {
            // similar sizes or self is larger, stitch iterate both
            Difference {
                inner: DifferenceInner::Stitch {
                    self_iter: self.iter(),
                    other_iter: other.iter().peekable(),
                },
            }
        }
    }

    /// Visits the elements representing the symmetric difference,
    /// i.e., the elements that are in `self` or in `other` but not in both,
    /// in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = OSBTreeSet::new();
    /// b.insert(4);
    /// b.insert(2);
    /// b.insert(3);
    /// b.insert(4);
    ///
    /// let sym_diff: Vec<_> = a.symmetric_difference(&b).cloned().collect();
    /// assert_eq!(sym_diff, [1, 4]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m + n) where m and n are the sizes of the two sets.
    pub fn symmetric_difference<'a>(&'a self, other: &'a OSBTreeSet<T>) -> SymmetricDifference<'a, T>
    where
        T: Ord,
    {
        SymmetricDifference {
            inner: MergeIterInner::new(self.iter(), other.iter()),
        }
    }

    /// Visits the elements representing the intersection,
    /// i.e., the elements that are both in `self` and `other`,
    /// in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = OSBTreeSet::new();
    /// b.insert(4);
    /// b.insert(2);
    /// b.insert(3);
    /// b.insert(4);
    ///
    /// let intersection: Vec<_> = a.intersection(&b).cloned().collect();
    /// assert_eq!(intersection, [2, 3]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(min(m, n)) for stitch iteration or O(m log n) for search-based iteration,
    /// where m and n are the sizes of the two sets.
    pub fn intersection<'a>(&'a self, other: &'a OSBTreeSet<T>) -> Intersection<'a, T>
    where
        T: Ord + Clone,
    {
        // Check for empty sets
        let (Some(self_min), Some(self_max)) = (self.first(), self.last()) else {
            return Intersection {
                inner: IntersectionInner::Answer(None),
            };
        };

        let (Some(other_min), Some(other_max)) = (other.first(), other.last()) else {
            return Intersection {
                inner: IntersectionInner::Answer(None),
            };
        };

        // Check for disjoint sets
        if self_max < other_min || other_max < self_min {
            return Intersection {
                inner: IntersectionInner::Answer(None),
            };
        }

        // Choose algorithm based on size difference
        let self_len = self.len();
        let other_len = other.len();

        if self_len > ITER_PERFORMANCE_TIPPING_SIZE_DIFF * other_len {
            // self is much larger, iterate other and search in self
            Intersection {
                inner: IntersectionInner::Search {
                    small_iter: other.iter(),
                    large_set: self,
                },
            }
        } else if other_len > ITER_PERFORMANCE_TIPPING_SIZE_DIFF * self_len {
            // other is much larger, iterate self and search in other
            Intersection {
                inner: IntersectionInner::Search {
                    small_iter: self.iter(),
                    large_set: other,
                },
            }
        } else {
            // similar sizes, stitch iterate both
            Intersection {
                inner: IntersectionInner::Stitch {
                    a: self.iter(),
                    b: other.iter(),
                },
            }
        }
    }

    /// Visits the elements representing the union,
    /// i.e., all the elements in `self` or `other`, without duplicates,
    /// in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = OSBTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let union: Vec<_> = a.union(&b).cloned().collect();
    /// assert_eq!(union, [1, 2, 3]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m + n) where m and n are the sizes of the two sets.
    pub fn union<'a>(&'a self, other: &'a OSBTreeSet<T>) -> Union<'a, T>
    where
        T: Ord,
    {
        Union {
            inner: MergeIterInner::new(self.iter(), other.iter()),
        }
    }

    /// Clears the set, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut v = OSBTreeSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(n)
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Returns `true` if the set contains a value.
    ///
    /// The value may be any borrowed form of the set's element type, but the
    /// ordering on the borrowed form *must* match the ordering on the
    /// element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set = OSBTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q> + Ord + Clone,
        Q: ?Sized + Ord,
    {
        self.map.contains_key(value)
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    ///
    /// The value may be any borrowed form of the set's element type, but the
    /// ordering on the borrowed form *must* match the ordering on the
    /// element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set = OSBTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord + Clone,
        Q: ?Sized + Ord,
    {
        self.map.get_key_value(value).map(|(k, ())| k)
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let a = OSBTreeSet::from([1, 2, 3]);
    /// let mut b = OSBTreeSet::new();
    /// b.insert(4);
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(min(m, n) log max(m, n)) where m and n are the sizes of the two sets.
    #[must_use]
    pub fn is_disjoint(&self, other: &OSBTreeSet<T>) -> bool
    where
        T: Ord + Clone,
    {
        // Two sets are disjoint if they have no elements in common
        // Use the smaller set to iterate and check against the larger
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e., `other` contains at least all the values in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let sup = OSBTreeSet::from([1, 2, 3]);
    /// let mut sub = OSBTreeSet::new();
    ///
    /// assert_eq!(sub.is_subset(&sup), true);
    /// sub.insert(2);
    /// assert_eq!(sub.is_subset(&sup), true);
    /// sub.insert(4);
    /// assert_eq!(sub.is_subset(&sup), false);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m log n) where m is the size of `self` and n is the size of `other`.
    #[must_use]
    pub fn is_subset(&self, other: &OSBTreeSet<T>) -> bool
    where
        T: Ord + Clone,
    {
        // self is a subset of other if all elements of self are in other
        self.len() <= other.len() && self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if the set is a superset of another,
    /// i.e., `self` contains at least all the values in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let sup = OSBTreeSet::from([1, 2, 3]);
    /// let mut sub = OSBTreeSet::new();
    ///
    /// assert_eq!(sup.is_superset(&sub), true);
    /// sub.insert(2);
    /// assert_eq!(sup.is_superset(&sub), true);
    /// sub.insert(4);
    /// assert_eq!(sup.is_superset(&sub), false);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m log n) where m is the size of `other` and n is the size of `self`.
    #[must_use]
    pub fn is_superset(&self, other: &OSBTreeSet<T>) -> bool
    where
        T: Ord + Clone,
    {
        // self is a superset of other if other is a subset of self
        other.is_subset(self)
    }

    /// Returns the first element in the set, if any.
    /// This is the minimum element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// assert_eq!(set.first(), None);
    /// set.insert(2);
    /// assert_eq!(set.first(), Some(&2));
    /// set.insert(1);
    /// assert_eq!(set.first(), Some(&1));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) - uses cached first leaf handle.
    #[must_use]
    pub fn first(&self) -> Option<&T>
    where
        T: Ord + Clone,
    {
        self.map.first_key_value().map(|(k, ())| k)
    }

    /// Returns the last element in the set, if any.
    /// This is the maximum element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// assert_eq!(set.last(), None);
    /// set.insert(1);
    /// assert_eq!(set.last(), Some(&1));
    /// set.insert(2);
    /// assert_eq!(set.last(), Some(&2));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) - uses cached last leaf handle.
    #[must_use]
    pub fn last(&self) -> Option<&T>
    where
        T: Ord + Clone,
    {
        self.map.last_key_value().map(|(k, ())| k)
    }

    /// Removes and returns the first element in the set.
    /// The first element is the minimum element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(1);
    /// set.insert(2);
    /// while let Some(n) = set.pop_first() {
    ///     assert!(set.iter().all(|&k| k > n));
    /// }
    /// assert!(set.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn pop_first(&mut self) -> Option<T>
    where
        T: Clone + Ord,
    {
        self.map.pop_first().map(|(k, ())| k)
    }

    /// Removes and returns the last element in the set.
    /// The last element is the maximum element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(1);
    /// set.insert(2);
    /// while let Some(n) = set.pop_last() {
    ///     assert!(set.iter().all(|&k| k < n));
    /// }
    /// assert!(set.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn pop_last(&mut self) -> Option<T>
    where
        T: Clone + Ord,
    {
        self.map.pop_last().map(|(k, ())| k)
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain an equal value, `true` is
    ///   returned.
    /// - If the set already contained an equal value, `false` is returned, and
    ///   the entry is not updated.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Clone + Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    /// Adds a value to the set, replacing the existing element, if any, that is
    /// equal to the value. Returns the replaced element.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(Vec::<i32>::new());
    ///
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 0);
    /// set.replace(Vec::with_capacity(10));
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 10);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Clone + Ord,
    {
        // Remove the existing key if present, then insert the new one
        let existing = self.map.remove_entry(&value).map(|(k, ())| k);
        self.map.insert(value, ());
        existing
    }

    /// If the set contains an element equal to the value, removes it from the
    /// set and drops it. Returns whether such an element was present.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + Clone + Ord,
        Q: ?Sized + Ord,
    {
        self.map.remove(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(2);
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q> + Clone + Ord,
        Q: ?Sized + Ord,
    {
        self.map.remove_entry(value).map(|(k, ())| k)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set: OSBTreeSet<i32> = (0..8).collect();
    /// // Keep only the elements with even-numbered values.
    /// set.retain(|&k| k % 2 == 0);
    /// assert!(set.into_iter().eq(vec![0, 2, 4, 6]));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(n log n) in the worst case (when many elements are removed).
    pub fn retain<F>(&mut self, mut f: F)
    where
        T: Clone + Ord,
        F: FnMut(&T) -> bool,
    {
        self.map.retain(|k, ()| f(k));
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a value from `other` is already present in `self`, it is not replaced.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = OSBTreeSet::new();
    /// b.insert(3);
    /// b.insert(4);
    /// b.insert(5);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert!(a.contains(&1));
    /// assert!(a.contains(&2));
    /// assert!(a.contains(&3));
    /// assert!(a.contains(&4));
    /// assert!(a.contains(&5));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m log(n + m)), where m is the size of `other` and n is the size of `self`.
    pub fn append(&mut self, other: &mut Self)
    where
        T: Clone + Ord,
    {
        self.map.append(&mut other.map);
    }

    /// Splits the collection into two at the given value. Returns everything after the given value,
    /// including the value. If the value is not present, the split will occur at the nearest
    /// greater value, or return an empty set if no such value exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    /// a.insert(17);
    /// a.insert(41);
    ///
    /// let b = a.split_off(&3);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 3);
    ///
    /// assert!(a.contains(&1));
    /// assert!(a.contains(&2));
    ///
    /// assert!(b.contains(&3));
    /// assert!(b.contains(&17));
    /// assert!(b.contains(&41));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m log(n)), where m is the number of elements being split off.
    #[allow(clippy::return_self_not_must_use)]
    pub fn split_off<Q: Ord>(&mut self, value: &Q) -> Self
    where
        T: Borrow<Q> + Clone + Ord,
    {
        OSBTreeSet {
            map: self.map.split_off(value),
        }
    }

    /// Creates an iterator that visits elements in the specified range in ascending order
    /// and uses a closure to determine if an element should be removed.
    ///
    /// If the closure returns `true`, the element is removed from the set and
    /// yielded. If the closure returns `false`, or panics, the element remains
    /// in the set and will not be yielded.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped without iterating
    /// or the iteration short-circuits, then the remaining elements will be retained.
    /// Use [`retain`] with a negated predicate if you do not need the returned iterator.
    ///
    /// [`retain`]: OSBTreeSet::retain
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// // Splitting a set into even and odd values, reusing the original set:
    /// let mut set: OSBTreeSet<i32> = (0..8).collect();
    /// let evens: OSBTreeSet<_> = set.extract_if(.., |v| v % 2 == 0).collect();
    /// let odds = set;
    /// assert_eq!(evens.iter().copied().collect::<Vec<_>>(), [0, 2, 4, 6]);
    /// assert_eq!(odds.iter().copied().collect::<Vec<_>>(), [1, 3, 5, 7]);
    ///
    /// // Splitting a set into low and high halves, reusing the original set:
    /// let mut set: OSBTreeSet<i32> = (0..8).collect();
    /// let low: OSBTreeSet<_> = set.extract_if(0..4, |_v| true).collect();
    /// let high = set;
    /// assert_eq!(low.iter().copied().collect::<Vec<_>>(), [0, 1, 2, 3]);
    /// assert_eq!(high.iter().copied().collect::<Vec<_>>(), [4, 5, 6, 7]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m + k log n) where m is the number of elements in the range and k is
    /// the number of elements extracted. All keys in the range are collected
    /// upfront, then each extracted element requires a O(log n) removal.
    pub fn extract_if<F, R>(&mut self, range: R, pred: F) -> ExtractIf<'_, T, R, F>
    where
        T: Clone + Ord,
        R: RangeBounds<T>,
        F: FnMut(&T) -> bool,
    {
        ExtractIf {
            pred,
            inner: self.map.extract_if_inner(range),
        }
    }

    /// Gets an iterator over the values in the set, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut set = OSBTreeSet::new();
    /// set.insert(3);
    /// set.insert(2);
    /// set.insert(1);
    ///
    /// for value in set.iter() {
    ///     println!("{value}");
    /// }
    ///
    /// let first = set.iter().next().unwrap();
    /// assert_eq!(*first, 1);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) to create the iterator; O(1) per iteration step via linked leaves.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.map.keys(),
        }
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1);
    /// assert_eq!(a.len(), 1);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub const fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let mut a = OSBTreeSet::new();
    /// assert!(a.is_empty());
    /// a.insert(1);
    /// assert!(!a.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<T: Hash> Hash for OSBTreeSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.map.hash(state);
    }
}

impl<T: PartialEq> PartialEq for OSBTreeSet<T> {
    fn eq(&self, other: &OSBTreeSet<T>) -> bool {
        self.map == other.map
    }
}

impl<T: Eq> Eq for OSBTreeSet<T> {}

impl<T: PartialOrd> PartialOrd for OSBTreeSet<T> {
    fn partial_cmp(&self, other: &OSBTreeSet<T>) -> Option<Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

impl<T: Ord> Ord for OSBTreeSet<T> {
    fn cmp(&self, other: &OSBTreeSet<T>) -> Ordering {
        self.map.cmp(&other.map)
    }
}

impl<T: Clone + Ord> Clone for OSBTreeSet<T> {
    fn clone(&self) -> Self {
        OSBTreeSet {
            map: self.map.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for OSBTreeSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T> Default for OSBTreeSet<T> {
    fn default() -> Self {
        OSBTreeSet::new()
    }
}

impl<T: Ord + Clone> FromIterator<T> for OSBTreeSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = OSBTreeSet::new();
        set.extend(iter);
        set
    }
}

impl<T: Ord + Clone> Extend<T> for OSBTreeSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl<'a, T: 'a + Ord + Copy> Extend<&'a T> for OSBTreeSet<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for &value in iter {
            self.insert(value);
        }
    }
}

impl<T: Ord + Clone> Sub<&OSBTreeSet<T>> for &OSBTreeSet<T> {
    type Output = OSBTreeSet<T>;

    /// Returns the difference of `self` and `rhs` as a new `OSBTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let a = OSBTreeSet::from([1, 2, 3]);
    /// let b = OSBTreeSet::from([2]);
    /// let diff = &a - &b;
    /// assert_eq!(diff.iter().copied().collect::<Vec<_>>(), [1, 3]);
    /// ```
    fn sub(self, rhs: &OSBTreeSet<T>) -> OSBTreeSet<T> {
        self.difference(rhs).cloned().collect()
    }
}

impl<T: Ord + Clone> BitXor<&OSBTreeSet<T>> for &OSBTreeSet<T> {
    type Output = OSBTreeSet<T>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `OSBTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let a = OSBTreeSet::from([1, 2, 3]);
    /// let b = OSBTreeSet::from([3, 4]);
    /// let sym = &a ^ &b;
    /// assert_eq!(sym.iter().copied().collect::<Vec<_>>(), [1, 2, 4]);
    /// ```
    fn bitxor(self, rhs: &OSBTreeSet<T>) -> OSBTreeSet<T> {
        self.symmetric_difference(rhs).cloned().collect()
    }
}

impl<T: Ord + Clone> BitAnd<&OSBTreeSet<T>> for &OSBTreeSet<T> {
    type Output = OSBTreeSet<T>;

    /// Returns the intersection of `self` and `rhs` as a new `OSBTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let a = OSBTreeSet::from([1, 2, 3]);
    /// let b = OSBTreeSet::from([2, 4]);
    /// let inter = &a & &b;
    /// assert_eq!(inter.iter().copied().collect::<Vec<_>>(), [2]);
    /// ```
    fn bitand(self, rhs: &OSBTreeSet<T>) -> OSBTreeSet<T> {
        self.intersection(rhs).cloned().collect()
    }
}

impl<T: Ord + Clone> BitOr<&OSBTreeSet<T>> for &OSBTreeSet<T> {
    type Output = OSBTreeSet<T>;

    /// Returns the union of `self` and `rhs` as a new `OSBTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let a = OSBTreeSet::from([1, 2]);
    /// let b = OSBTreeSet::from([2, 3]);
    /// let union = &a | &b;
    /// assert_eq!(union.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
    /// ```
    fn bitor(self, rhs: &OSBTreeSet<T>) -> OSBTreeSet<T> {
        self.union(rhs).cloned().collect()
    }
}

impl<T: Ord + Clone, const N: usize> From<[T; N]> for OSBTreeSet<T> {
    fn from(arr: [T; N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T: Clone + Ord> IntoIterator for OSBTreeSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Gets an iterator for moving out the `OSBTreeSet`'s contents in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeSet;
    ///
    /// let set = OSBTreeSet::from([1, 2, 3, 4]);
    ///
    /// let v: Vec<_> = set.into_iter().collect();
    /// assert_eq!(v, [1, 2, 3, 4]);
    /// ```
    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            inner: self.map.into_keys(),
        }
    }
}

impl<'a, T> IntoIterator for &'a OSBTreeSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.inner.next_back()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter").field("inner", &self.inner).finish()
    }
}

impl<T> Default for Iter<'_, T> {
    /// Creates an empty `osbtree_set::Iter`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_set;
    /// let iter: osbtree_set::Iter<'_, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Iter {
            inner: Keys::default(),
        }
    }
}

impl<T: Clone + Ord> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Clone + Ord> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<T: Clone + Ord> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: Clone + Ord> FusedIterator for IntoIter<T> {}

impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter").field("inner", &self.inner).finish()
    }
}

impl<T> Default for IntoIter<T> {
    /// Creates an empty `osbtree_set::IntoIter`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_set;
    /// let iter: osbtree_set::IntoIter<u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoIter {
            inner: IntoKeys::default(),
        }
    }
}

impl<'a, T> Iterator for Range<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.inner.next().map(|(k, ())| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Range<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.inner.next_back().map(|(k, ())| k)
    }
}

impl<T> FusedIterator for Range<'_, T> {}

impl<T> Clone for Range<'_, T> {
    fn clone(&self) -> Self {
        Range {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Default for Range<'_, T> {
    /// Creates an empty `osbtree_set::Range`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_set;
    /// let iter: osbtree_set::Range<'_, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        Range {
            inner: MapRange::default(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Range<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Range").field("inner", &self.inner).finish()
    }
}

impl<'a, T: Ord + Clone> Iterator for Difference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => loop {
                let self_next = self_iter.next()?;
                loop {
                    match other_iter.peek() {
                        None => return Some(self_next),
                        Some(&other_next) => match self_next.cmp(other_next) {
                            Less => return Some(self_next),
                            Equal => {
                                other_iter.next();
                                break;
                            }
                            Greater => {
                                other_iter.next();
                            }
                        },
                    }
                }
            },
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => loop {
                let self_next = self_iter.next()?;
                if !other_set.contains(self_next) {
                    return Some(self_next);
                }
            },
            DifferenceInner::Iterate(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = match &self.inner {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => {
                // For Stitch, we can bound by self_len - other_len since we're
                // merging two sorted iterators and at most other_len can be removed.
                let self_len = self_iter.len();
                let other_len = other_iter.len();
                (self_len.saturating_sub(other_len), self_len)
            }
            DifferenceInner::Search {
                self_iter,
                ..
            } => {
                // For Search, we can't know how many elements of self are in other,
                // so lower bound is 0 (all could be in other).
                (0, self_iter.len())
            }
            DifferenceInner::Iterate(iter) => {
                // For Iterate, sets are disjoint or other is empty, so all of self survives.
                let len = iter.len();
                (len, len)
            }
        };
        (lower, Some(upper))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord + Clone> FusedIterator for Difference<'_, T> {}

impl<T> Clone for Difference<'_, T> {
    fn clone(&self) -> Self {
        Difference {
            inner: match &self.inner {
                DifferenceInner::Stitch {
                    self_iter,
                    other_iter,
                } => DifferenceInner::Stitch {
                    self_iter: self_iter.clone(),
                    other_iter: other_iter.clone(),
                },
                DifferenceInner::Search {
                    self_iter,
                    other_set,
                } => DifferenceInner::Search {
                    self_iter: self_iter.clone(),
                    other_set,
                },
                DifferenceInner::Iterate(iter) => DifferenceInner::Iterate(iter.clone()),
            },
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Difference<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Difference").field("inner", &self.inner).finish()
    }
}

impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            match self.inner.nexts(core::cmp::Ord::cmp) {
                (None, None) => return None,
                (Some(a), None) => return Some(a),
                (None, Some(b)) => return Some(b),
                (Some(_), Some(_)) => {
                    // Equal elements - skip both and continue
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.inner.lens();
        // Could be 0 if all elements are equal, or sum if all different
        (0, Some(a_len + b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord> FusedIterator for SymmetricDifference<'_, T> {}

impl<T> Clone for SymmetricDifference<'_, T> {
    fn clone(&self) -> Self {
        SymmetricDifference {
            inner: self.inner.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for SymmetricDifference<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymmetricDifference").field("inner", &self.inner).finish()
    }
}

impl<'a, T: Ord + Clone> Iterator for Intersection<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            IntersectionInner::Stitch {
                a,
                b,
            } => {
                let mut a_next = a.next()?;
                let mut b_next = b.next()?;
                loop {
                    match a_next.cmp(b_next) {
                        Less => a_next = a.next()?,
                        Greater => b_next = b.next()?,
                        Equal => return Some(a_next),
                    }
                }
            }
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => loop {
                let small_next = small_iter.next()?;
                if large_set.contains(small_next) {
                    return Some(small_next);
                }
            },
            IntersectionInner::Answer(answer) => answer.take(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IntersectionInner::Stitch {
                a,
                b,
            } => (0, Some(min(a.len(), b.len()))),
            IntersectionInner::Search {
                small_iter,
                ..
            } => (0, Some(small_iter.len())),
            IntersectionInner::Answer(Some(_)) => (1, Some(1)),
            IntersectionInner::Answer(None) => (0, Some(0)),
        }
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord + Clone> FusedIterator for Intersection<'_, T> {}

impl<T> Clone for Intersection<'_, T> {
    fn clone(&self) -> Self {
        Intersection {
            inner: match &self.inner {
                IntersectionInner::Stitch {
                    a,
                    b,
                } => IntersectionInner::Stitch {
                    a: a.clone(),
                    b: b.clone(),
                },
                IntersectionInner::Search {
                    small_iter,
                    large_set,
                } => IntersectionInner::Search {
                    small_iter: small_iter.clone(),
                    large_set,
                },
                IntersectionInner::Answer(answer) => IntersectionInner::Answer(*answer),
            },
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Intersection<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Intersection").field("inner", &self.inner).finish()
    }
}

impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match self.inner.nexts(core::cmp::Ord::cmp) {
            (None, None) => None,
            (Some(a), None | Some(_)) => Some(a),
            (None, Some(b)) => Some(b),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.inner.lens();
        // At minimum, it's max(a, b), at maximum it's a + b
        (max(a_len, b_len), Some(a_len + b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord> FusedIterator for Union<'_, T> {}

impl<T> Clone for Union<'_, T> {
    fn clone(&self) -> Self {
        Union {
            inner: self.inner.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Union<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Union").field("inner", &self.inner).finish()
    }
}

impl<T, R, F> fmt::Debug for ExtractIf<'_, T, R, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf").finish_non_exhaustive()
    }
}

impl<T, R, F> Iterator for ExtractIf<'_, T, R, F>
where
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
    T: Clone + Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let pred = &mut self.pred;
        self.inner.next(&mut |k, ()| pred(k)).map(|(k, ())| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, R, F> FusedIterator for ExtractIf<'_, T, R, F>
where
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
    T: Clone + Ord,
{
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn intersection_answer_some_size_hint() {
        let set: OSBTreeSet<i32> = [1].into_iter().collect();
        let value = set.first().expect("set contains one value");

        let mut intersection = Intersection {
            inner: IntersectionInner::Answer(Some(value)),
        };
        assert_eq!(intersection.size_hint(), (1, Some(1)));
        assert_eq!(intersection.next(), Some(&1));
        assert_eq!(intersection.next(), None);
    }
}

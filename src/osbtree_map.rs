use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops::{Bound, Index, RangeBounds};

use crate::raw::{Handle, LeafNode, Node, RawOSBTreeMap};

mod capacity;
mod entry;
mod order_statistic;

pub use crate::Rank;
pub use entry::{Entry, OccupiedEntry, VacantEntry};

/// Validates that the start bound does not exceed the end bound.
///
/// # Panics
///
/// Panics if `start > end` or if `start == end` and both bounds are `Excluded`.
fn validate_range_bounds<T, R>(range: &R)
where
    T: ?Sized + Ord,
    R: RangeBounds<T>,
{
    if let (Bound::Included(start) | Bound::Excluded(start), Bound::Included(end) | Bound::Excluded(end)) =
        (range.start_bound(), range.end_bound())
    {
        let valid =
            if matches!(range.start_bound(), Bound::Excluded(_)) && matches!(range.end_bound(), Bound::Excluded(_)) {
                start < end
            } else {
                start <= end
            };
        assert!(valid, "range start is greater than range end in OSBTreeMap");
    }
}

/// An ordered map based on a [B-Tree].
///
/// Given a key type with a [total order], an ordered map stores its entries in key order.
/// That means that keys must be of a type that implements the [`Ord`] trait,
/// such that two keys can always be compared to determine their [`Ordering`].
/// Examples of keys with a total order are strings with lexicographical order,
/// and numbers with their natural order.
///
/// Iterators obtained from functions such as [`OSBTreeMap::iter`], [`OSBTreeMap::into_iter`],
/// [`OSBTreeMap::values`], or [`OSBTreeMap::keys`] produce their items in key order, and take
/// worst-case logarithmic and amortized constant time per item returned.
///
/// It is a logic error for a key to be modified in such a way that the key's ordering relative to
/// any other key, as determined by the [`Ord`] trait, changes while it is in the map. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
/// The behavior resulting from such a logic error is not specified, but will be encapsulated to the
/// `OSBTreeMap` that observed the logic error and not result in undefined behavior. This could
/// include panics, incorrect results, aborts, memory leaks, and non-termination.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `OSBTreeMap<&str, &str>` in this example).
/// let mut movie_reviews = OSBTreeMap::new();
///
/// // review some movies.
/// movie_reviews.insert("Office Space",       "Deals with real issues in the workplace.");
/// movie_reviews.insert("Pulp Fiction",       "Masterpiece.");
/// movie_reviews.insert("The Godfather",      "Very enjoyable.");
/// movie_reviews.insert("The Blues Brothers", "Eye lyked it a lot.");
///
/// // check for a specific one.
/// if !movie_reviews.contains_key("Les Miserables") {
///     println!("We've got {} reviews, but Les Miserables ain't one.",
///              movie_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// movie_reviews.remove("The Blues Brothers");
///
/// // look up the values associated with some keys.
/// let to_find = ["Up!", "Office Space"];
/// for movie in &to_find {
///     match movie_reviews.get(movie) {
///        Some(review) => println!("{movie}: {review}"),
///        None => println!("{movie} is unreviewed.")
///     }
/// }
///
/// // Look up the value for a key (will panic if the key is not found).
/// println!("Movie review: {}", movie_reviews["Office Space"]);
///
/// // iterate over everything.
/// for (movie, review) in &movie_reviews {
///     println!("{movie}: \"{review}\"");
/// }
/// ```
///
/// An `OSBTreeMap` with a known list of items can be initialized from an array:
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let solar_distance = OSBTreeMap::from([
///     ("Mercury", 0.4),
///     ("Venus", 0.7),
///     ("Earth", 1.0),
///     ("Mars", 1.5),
/// ]);
/// ```
///
/// ## `Entry` API
///
/// `OSBTreeMap` implements an [`Entry API`], which allows for complex
/// methods of getting, setting, updating and removing keys and their values:
///
/// [`Entry API`]: OSBTreeMap::entry
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `OSBTreeMap<&str, u8>` in this example).
/// let mut player_stats = OSBTreeMap::new();
///
/// fn random_stat_buff() -> u8 {
///     // could actually return some random value here - let's just return
///     // some fixed value for now
///     42
/// }
///
/// // insert a key only if it doesn't already exist
/// player_stats.entry("health").or_insert(100);
///
/// // insert a key using a function that provides a new value only if it
/// // doesn't already exist
/// player_stats.entry("defence").or_insert_with(random_stat_buff);
///
/// // update a key, guarding against the key possibly not being set
/// let stat = player_stats.entry("attack").or_insert(100);
/// *stat += random_stat_buff();
///
/// // modify an entry before an insert with in-place mutation
/// player_stats.entry("mana").and_modify(|mana| *mana += 200).or_insert(100);
/// ```
///
/// # Background
///
/// A B-tree is (like) a [binary search tree], but adapted to the natural granularity that modern
/// machines like to consume data at. This means that each node contains an entire array of elements,
/// instead of just a single element.
///
/// B-Trees represent a fundamental compromise between cache-efficiency and actually minimizing
/// the amount of work performed in a search. In theory, a binary search tree (BST) is the optimal
/// choice for a sorted map, as a perfectly balanced BST performs the theoretical minimum number of
/// comparisons necessary to find an element (log<sub>2</sub>n). However, in practice the way this
/// is done is *very* inefficient for modern computer architectures. In particular, every element
/// is stored in its own individually heap-allocated node. This means that every single insertion
/// triggers a heap-allocation, and every comparison is a potential cache-miss due to the indirection.
/// Since both heap-allocations and cache-misses are notably expensive in practice, we are forced to,
/// at the very least, reconsider the BST strategy.
///
/// A B-Tree instead makes each node contain B-1 to 2B-1 elements in a contiguous array. By doing
/// this, we reduce the number of allocations by a factor of B, and improve cache efficiency in
/// searches. However, this does mean that searches will have to do *more* comparisons on average.
/// The precise number of comparisons depends on the node search strategy used. For optimal cache
/// efficiency, one could search the nodes linearly. For optimal comparisons, one could search
/// the node using binary search. As a compromise, one could also perform a linear search
/// that initially only checks every i<sup>th</sup> element for some choice of i.
///
/// Our implementation uses binary search within each node, giving O(log B) comparisons per node
/// and O(log B * log n) = O(log n) total comparisons for tree operations. This matches the
/// asymptotic complexity of a standard BST while providing better cache locality due to the
/// larger node size.
///
/// [B-Tree]: https://en.wikipedia.org/wiki/B-tree
/// [binary search tree]: https://en.wikipedia.org/wiki/Binary_search_tree
/// [total order]: https://en.wikipedia.org/wiki/Total_order
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
pub struct OSBTreeMap<K, V> {
    raw: RawOSBTreeMap<K, V>,
}

/// An iterator over the entries of a `OSBTreeMap`.
///
/// This `struct` is created by the [`iter`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(1, "a"), (2, "b")]);
/// let mut iter = map.iter();
/// assert_eq!(iter.next(), Some((&1, &"a")));
/// assert_eq!(iter.next_back(), Some((&2, &"b")));
/// assert_eq!(iter.next(), None);
/// ```
///
/// [`iter`]: OSBTreeMap::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, K, V> {
    tree: *const RawOSBTreeMap<K, V>,
    front_leaf: Option<Handle>,
    front_index: usize,
    back_leaf: Option<Handle>,
    back_index: usize,
    remaining: usize,
    _marker: PhantomData<&'a RawOSBTreeMap<K, V>>,
}

// SAFETY: Iter behaves as &RawOSBTreeMap<K, V>, so it is Send/Sync when the tree is Sync.
unsafe impl<K: Sync, V: Sync> Send for Iter<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for Iter<'_, K, V> {}

/// A mutable iterator over the entries of a `OSBTreeMap`.
///
/// This `struct` is created by the [`iter_mut`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let mut map = OSBTreeMap::from([(1, 10), (2, 20)]);
/// for (_, value) in map.iter_mut() {
///     *value += 1;
/// }
/// let values: Vec<_> = map.values().copied().collect();
/// assert_eq!(values, [11, 21]);
/// ```
///
/// [`iter_mut`]: OSBTreeMap::iter_mut
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IterMut<'a, K: 'a, V: 'a> {
    tree: *mut RawOSBTreeMap<K, V>,
    front_leaf: Option<Handle>,
    front_index: usize,
    back_leaf: Option<Handle>,
    back_index: usize,
    remaining: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: IterMut behaves as &mut RawOSBTreeMap<K, V>, so it is Send when K and V are Send.
// It is NOT Sync because mutable iterators should not be shared across threads.
unsafe impl<K: Send, V: Send> Send for IterMut<'_, K, V> {}

/// An owning iterator over the entries of a `OSBTreeMap`, sorted by key.
///
/// This `struct` is created by the [`into_iter`] method on [`OSBTreeMap`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(1, "a"), (2, "b")]);
/// let mut iter = map.into_iter();
/// assert_eq!(iter.next(), Some((1, "a")));
/// assert_eq!(iter.next_back(), Some((2, "b")));
/// assert_eq!(iter.next(), None);
/// ```
///
/// [`into_iter`]: IntoIterator::into_iter
pub struct IntoIter<K, V> {
    inner: alloc::vec::IntoIter<(K, V)>,
}

/// An iterator over the keys of a `OSBTreeMap`.
///
/// This `struct` is created by the [`keys`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(2, "b"), (1, "a")]);
/// let keys: Vec<_> = map.keys().copied().collect();
/// assert_eq!(keys, [1, 2]);
/// ```
///
/// [`keys`]: OSBTreeMap::keys
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

/// An iterator over the values of a `OSBTreeMap`.
///
/// This `struct` is created by the [`values`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(1, "a"), (2, "b")]);
/// let values: Vec<_> = map.values().copied().collect();
/// assert_eq!(values, ["a", "b"]);
/// ```
///
/// [`values`]: OSBTreeMap::values
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

/// A mutable iterator over the values of a `OSBTreeMap`.
///
/// This `struct` is created by the [`values_mut`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let mut map = OSBTreeMap::from([
///     (1, String::from("hello")),
///     (2, String::from("goodbye")),
/// ]);
/// for value in map.values_mut() {
///     value.push('!');
/// }
/// let values: Vec<_> = map.values().cloned().collect();
/// assert_eq!(values, [String::from("hello!"), String::from("goodbye!")]);
/// ```
///
/// [`values_mut`]: OSBTreeMap::values_mut
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

// SAFETY: ValuesMut is Send when its inner IterMut is Send.
unsafe impl<K: Send, V: Send> Send for ValuesMut<'_, K, V> {}

/// An owning iterator over the keys of a `OSBTreeMap`.
///
/// This `struct` is created by the [`into_keys`] method on [`OSBTreeMap`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(2, "b"), (1, "a")]);
/// let mut keys = map.into_keys();
/// assert_eq!(keys.next(), Some(1));
/// assert_eq!(keys.next_back(), Some(2));
/// assert_eq!(keys.next(), None);
/// ```
///
/// [`into_keys`]: OSBTreeMap::into_keys
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoKeys<K, V> {
    inner: IntoIter<K, V>,
}

/// An owning iterator over the values of a `OSBTreeMap`.
///
/// This `struct` is created by the [`into_values`] method on [`OSBTreeMap`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(1, "hello"), (2, "goodbye")]);
/// let mut values = map.into_values();
/// assert_eq!(values.next(), Some("hello"));
/// assert_eq!(values.next_back(), Some("goodbye"));
/// assert_eq!(values.next(), None);
/// ```
///
/// [`into_values`]: OSBTreeMap::into_values
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoValues<K, V> {
    inner: IntoIter<K, V>,
}

/// An iterator over a sub-range of entries in a `OSBTreeMap`.
///
/// This `struct` is created by the [`range`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let map = OSBTreeMap::from([(1, "a"), (2, "b"), (3, "c")]);
/// let mut range = map.range(2..=3);
/// assert_eq!(range.next(), Some((&2, &"b")));
/// assert_eq!(range.next_back(), Some((&3, &"c")));
/// assert_eq!(range.next(), None);
/// ```
///
/// [`range`]: OSBTreeMap::range
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Range<'a, K: 'a, V: 'a> {
    tree: *const RawOSBTreeMap<K, V>,
    front_leaf: Option<Handle>,
    front_index: usize,
    back_leaf: Option<Handle>,
    back_index: usize,
    remaining: usize,
    /// Tracks whether the iterator has been exhausted (front and back have crossed).
    finished: bool,
    _marker: PhantomData<&'a RawOSBTreeMap<K, V>>,
}

// SAFETY: Range behaves as &RawOSBTreeMap<K, V>, so it is Send/Sync when the tree is Sync.
unsafe impl<K: Sync, V: Sync> Send for Range<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for Range<'_, K, V> {}

/// A mutable iterator over a sub-range of entries in a `OSBTreeMap`.
///
/// This `struct` is created by the [`range_mut`] method on [`OSBTreeMap`]. See its
/// documentation for more.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let mut map = OSBTreeMap::from([(1, 10), (2, 20), (3, 30)]);
/// for (_, value) in map.range_mut(2..=3) {
///     *value += 1;
/// }
/// assert_eq!(map.get(&1), Some(&10));
/// assert_eq!(map.get(&2), Some(&21));
/// assert_eq!(map.get(&3), Some(&31));
/// ```
///
/// [`range_mut`]: OSBTreeMap::range_mut
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct RangeMut<'a, K: 'a, V: 'a> {
    tree: *mut RawOSBTreeMap<K, V>,
    front_leaf: Option<Handle>,
    front_index: usize,
    back_leaf: Option<Handle>,
    back_index: usize,
    remaining: usize,
    /// Tracks whether the iterator has been exhausted (front and back have crossed).
    finished: bool,
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: RangeMut behaves as &mut RawOSBTreeMap<K, V>, so it is Send when K and V are Send.
// It is NOT Sync because mutable iterators should not be shared across threads.
unsafe impl<K: Send, V: Send> Send for RangeMut<'_, K, V> {}

/// An iterator produced by calling `extract_if` on `OSBTreeMap`.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
///
/// let mut map = OSBTreeMap::from([(1, "a"), (2, "b"), (3, "c")]);
/// let extracted: Vec<_> = map.extract_if(1..=2, |_k, _v| true).collect();
/// assert_eq!(extracted, [(1, "a"), (2, "b")]);
/// assert_eq!(map.len(), 1);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<'a, K, V, R, F> {
    tree: &'a mut RawOSBTreeMap<K, V>,
    /// Keys to potentially extract (collected upfront)
    keys: alloc::vec::Vec<K>,
    /// Current index in keys
    index: usize,
    pred: F,
    _marker: PhantomData<R>,
}

pub(crate) struct ExtractIfInner<'a, K, V, R> {
    tree: &'a mut RawOSBTreeMap<K, V>,
    keys: alloc::vec::Vec<K>,
    index: usize,
    _marker: PhantomData<R>,
}

impl<K: Clone + Ord, V, R: RangeBounds<K>> ExtractIfInner<'_, K, V, R> {
    pub(crate) fn next<F>(&mut self, pred: &mut F) -> Option<(K, V)>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        while self.index < self.keys.len() {
            let key = &self.keys[self.index];
            self.index += 1;

            // Get mutable access to the value to check predicate.
            // The borrow must end before we call remove_entry, so we scope it.
            let should_remove = self.tree.get_mut(key).is_some_and(|value| pred(key, value));
            if should_remove {
                // Remove and return
                if let Some((k, v)) = self.tree.remove_entry(key) {
                    return Some((k, v));
                }
            }
        }
        None
    }

    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.keys.len() - self.index))
    }
}

impl<K, V> OSBTreeMap<K, V> {
    /// Makes a new, empty `OSBTreeMap`.
    ///
    /// Does not allocate anything on its own.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    ///
    /// // entries can now be inserted into the empty map
    /// map.insert(1, "a");
    /// ```
    #[must_use]
    pub const fn new() -> OSBTreeMap<K, V> {
        OSBTreeMap {
            raw: RawOSBTreeMap::new(),
        }
    }

    /// Clears the map, removing all elements.
    ///
    /// # Complexity
    ///
    /// O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.raw.clear();
    }

    /// Returns a reference to the value corresponding to the key.
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
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord + Clone,
        Q: ?Sized + Ord,
    {
        self.raw.get(key)
    }

    /// Returns the key-value pair corresponding to the supplied key. This is
    /// potentially useful:
    /// - for key types where non-identical keys can be considered equal;
    /// - for getting the `&K` stored key value from a borrowed `&Q` lookup key; or
    /// - for getting a reference to a key with the same lifetime as the collection.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::cmp::Ordering;
    /// use wabi_tree::OSBTreeMap;
    ///
    /// #[derive(Clone, Copy, Debug)]
    /// struct S {
    ///     id: u32,
    /// #   #[allow(unused)] // prevents a "field `name` is never read" error
    ///     name: &'static str, // ignored by equality and ordering operations
    /// }
    ///
    /// impl PartialEq for S {
    ///     fn eq(&self, other: &S) -> bool {
    ///         self.id == other.id
    ///     }
    /// }
    ///
    /// impl Eq for S {}
    ///
    /// impl PartialOrd for S {
    ///     fn partial_cmp(&self, other: &S) -> Option<Ordering> {
    ///         self.id.partial_cmp(&other.id)
    ///     }
    /// }
    ///
    /// impl Ord for S {
    ///     fn cmp(&self, other: &S) -> Ordering {
    ///         self.id.cmp(&other.id)
    ///     }
    /// }
    ///
    /// let j_a = S { id: 1, name: "Jessica" };
    /// let j_b = S { id: 1, name: "Jess" };
    /// let p = S { id: 2, name: "Paul" };
    /// assert_eq!(j_a, j_b);
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(j_a, "Paris");
    /// assert_eq!(map.get_key_value(&j_a), Some((&j_a, &"Paris")));
    /// assert_eq!(map.get_key_value(&j_b), Some((&j_a, &"Paris"))); // the notable case
    /// assert_eq!(map.get_key_value(&p), None);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn get_key_value<Q>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord + Clone,
        Q: ?Sized + Ord,
    {
        self.raw.get_key_value(k)
    }

    /// Returns the first key-value pair in the map.
    /// The key in this pair is the minimum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// assert_eq!(map.first_key_value(), None);
    /// map.insert(1, "b");
    /// map.insert(2, "a");
    /// assert_eq!(map.first_key_value(), Some((&1, &"b")));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) - uses cached first leaf handle.
    #[allow(clippy::must_use_candidate)]
    pub fn first_key_value(&self) -> Option<(&K, &V)>
    where
        K: Ord + Clone,
    {
        self.raw.first_key_value()
    }

    /// Returns the first entry in the map for in-place manipulation.
    /// The key of this entry is the minimum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// if let Some(mut entry) = map.first_entry() {
    ///     if *entry.key() > 0 {
    ///         entry.insert("first");
    ///     }
    /// }
    /// assert_eq!(*map.get(&1).unwrap(), "first");
    /// assert_eq!(*map.get(&2).unwrap(), "b");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) - uses cached first leaf handle.
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>>
    where
        K: Ord + Clone,
    {
        let first_leaf = self.raw.first_leaf()?;
        let leaf = self.raw.node(first_leaf).as_leaf();
        if leaf.key_count() == 0 {
            return None;
        }
        let key = leaf.key(0).clone();
        Some(OccupiedEntry {
            key,
            leaf_handle: first_leaf,
            index: 0,
            tree: &mut self.raw,
        })
    }

    /// Removes and returns the first element in the map.
    /// The key of this element is the minimum key that was in the map.
    ///
    /// # Examples
    ///
    /// Draining elements in ascending order, while keeping a usable map each iteration.
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// while let Some((key, _val)) = map.pop_first() {
    ///     assert!(map.iter().all(|(k, _v)| *k > key));
    /// }
    /// assert!(map.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn pop_first(&mut self) -> Option<(K, V)>
    where
        K: Clone + Ord,
    {
        self.raw.pop_first()
    }

    /// Returns the last key-value pair in the map.
    /// The key in this pair is the maximum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// assert_eq!(map.last_key_value(), None);
    /// map.insert(1, "b");
    /// map.insert(2, "a");
    /// assert_eq!(map.last_key_value(), Some((&2, &"a")));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) - uses cached last leaf handle.
    #[allow(clippy::must_use_candidate)]
    pub fn last_key_value(&self) -> Option<(&K, &V)>
    where
        K: Ord + Clone,
    {
        self.raw.last_key_value()
    }

    /// Returns the last entry in the map for in-place manipulation.
    /// The key of this entry is the maximum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// if let Some(mut entry) = map.last_entry() {
    ///     if *entry.key() > 0 {
    ///         entry.insert("last");
    ///     }
    /// }
    /// assert_eq!(*map.get(&1).unwrap(), "a");
    /// assert_eq!(*map.get(&2).unwrap(), "last");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) - uses cached last leaf handle.
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>>
    where
        K: Ord + Clone,
    {
        let last_leaf = self.raw.last_leaf()?;
        let leaf = self.raw.node(last_leaf).as_leaf();
        if leaf.key_count() == 0 {
            return None;
        }
        let index = leaf.key_count() - 1;
        let key = leaf.key(index).clone();
        Some(OccupiedEntry {
            key,
            leaf_handle: last_leaf,
            index,
            tree: &mut self.raw,
        })
    }

    /// Removes and returns the last element in the map.
    /// The key of this element is the maximum key that was in the map.
    ///
    /// # Examples
    ///
    /// Draining elements in descending order, while keeping a usable map each iteration.
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// while let Some((key, _val)) = map.pop_last() {
    ///     assert!(map.iter().all(|(k, _v)| *k < key));
    /// }
    /// assert!(map.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn pop_last(&mut self) -> Option<(K, V)>
    where
        K: Clone + Ord,
    {
        self.raw.pop_last()
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord + Clone,
        Q: ?Sized + Ord,
    {
        self.raw.contains_key(key)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord + Clone,
        Q: ?Sized + Ord,
    {
        self.raw.get_mut(key)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Clone + Ord,
    {
        self.raw.insert(key, value)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Clone + Ord,
        Q: ?Sized + Ord,
    {
        self.raw.remove(key)
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove_entry(&1), None);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Clone + Ord,
        Q: ?Sized + Ord,
    {
        self.raw.remove_entry(key)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)` returns `false`.
    /// The elements are visited in ascending key order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<i32, i32> = (0..8).map(|x| (x, x * 10)).collect();
    /// // Keep only the elements with even-numbered keys.
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert!(map.into_iter().eq(vec![(0, 0), (2, 20), (4, 40), (6, 60)]));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(n log n) in the worst case (when many elements are removed).
    pub fn retain<F>(&mut self, mut f: F)
    where
        K: Clone + Ord,
        F: FnMut(&K, &mut V) -> bool,
    {
        // Collect keys to remove first, since we can't modify while iterating
        let keys_to_remove: alloc::vec::Vec<K> = self
            .iter_mut()
            .filter_map(|(k, v)| {
                if f(k, v) {
                    None
                } else {
                    Some(k.clone())
                }
            })
            .collect();

        for key in keys_to_remove {
            self.raw.remove(&key);
        }
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c"); // Note: Key (3) also present in b.
    ///
    /// let mut b = OSBTreeMap::new();
    /// b.insert(3, "d"); // Note: Key (3) also present in a.
    /// b.insert(4, "e");
    /// b.insert(5, "f");
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert_eq!(a[&1], "a");
    /// assert_eq!(a[&2], "b");
    /// assert_eq!(a[&3], "d");
    /// assert_eq!(a[&4], "e");
    /// assert_eq!(a[&5], "f");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m log(n + m)), where m is the size of `other` and n is the size of `self`.
    pub fn append(&mut self, other: &mut Self)
    where
        K: Clone + Ord,
    {
        if other.is_empty() {
            return;
        }

        // Fast path: if self is empty, just swap the raw trees (O(1))
        if self.is_empty() {
            core::mem::swap(&mut self.raw, &mut other.raw);
            return;
        }

        // Drain the other tree in O(n) via leaf-chain walk, then insert each entry.
        // This avoids the O(log n) rebalancing cost per pop_first from the old approach.
        let entries = other.raw.drain_to_vec();
        for (k, v) in entries {
            self.raw.insert(k, v);
        }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the map.
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
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(3, "a");
    /// map.insert(5, "b");
    /// map.insert(8, "c");
    /// for (&key, &value) in map.range((Included(&4), Included(&8))) {
    ///     println!("{key}: {value}");
    /// }
    /// assert_eq!(Some((&5, &"b")), map.range(4..).next());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) to create the iterator; each iteration step is O(1) amortized.
    pub fn range<T, R>(&self, range: R) -> Range<'_, K, V>
    where
        T: ?Sized + Ord,
        K: Borrow<T> + Ord + Clone,
        R: RangeBounds<T>,
    {
        validate_range_bounds(&range);

        // Find front position based on start bound
        let (front_leaf, front_index) = match range.start_bound() {
            Bound::Unbounded => {
                if let Some(first) = self.raw.first_leaf() {
                    (Some(first), 0)
                } else {
                    (None, 0)
                }
            }
            Bound::Included(start) => {
                if let Some((handle, idx)) = self.raw.lower_bound(start) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
            Bound::Excluded(start) => {
                if let Some((handle, idx)) = self.raw.upper_bound(start) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
        };

        // Find back position based on end bound
        let (back_leaf, back_index) = match range.end_bound() {
            Bound::Unbounded => {
                if let Some(last) = self.raw.last_leaf() {
                    let leaf = self.raw.node(last).as_leaf();
                    let count = leaf.key_count();
                    if count > 0 {
                        (Some(last), count - 1)
                    } else {
                        (None, 0)
                    }
                } else {
                    (None, 0)
                }
            }
            Bound::Included(end) => {
                if let Some((handle, idx)) = self.raw.upper_bound_inclusive(end) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
            Bound::Excluded(end) => {
                if let Some((handle, idx)) = self.raw.lower_bound_exclusive(end) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
        };

        // Check if the range is empty (front position > back position, even across different leaves)
        let (finished, remaining) = match (front_leaf, back_leaf) {
            (Some(front), Some(back)) => {
                let front_key = self.raw.node(front).as_leaf().key(front_index);
                let back_key = self.raw.node(back).as_leaf().key(back_index);
                if front_key > back_key {
                    (true, 0)
                } else {
                    // Compute exact count using order statistics (O(log n) each).
                    // Explicitly specify K as the query type since T is also in scope.
                    let front_rank = self.raw.rank_of::<K>(front_key).unwrap_or(0);
                    let back_rank = self.raw.rank_of::<K>(back_key).unwrap_or(0);
                    (false, back_rank - front_rank + 1)
                }
            }
            _ => (true, 0), // One or both bounds are out of range
        };

        Range {
            tree: &raw const self.raw,
            front_leaf,
            front_index,
            back_leaf,
            back_index,
            remaining,
            finished,
            _marker: PhantomData,
        }
    }

    /// Constructs a mutable double-ended iterator over a sub-range of elements in the map.
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
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, i32> =
    ///     [("Alice", 0), ("Bob", 0), ("Carol", 0), ("Cheryl", 0)].into();
    /// for (_, balance) in map.range_mut("B".."Cheryl") {
    ///     *balance += 100;
    /// }
    /// for (name, balance) in &map {
    ///     println!("{name} => {balance}");
    /// }
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) to create the iterator; each iteration step is O(1) amortized.
    pub fn range_mut<T, R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        T: ?Sized + Ord,
        K: Borrow<T> + Ord + Clone,
        R: RangeBounds<T>,
    {
        validate_range_bounds(&range);

        // Find front position based on start bound
        let (front_leaf, front_index) = match range.start_bound() {
            Bound::Unbounded => {
                if let Some(first) = self.raw.first_leaf() {
                    (Some(first), 0)
                } else {
                    (None, 0)
                }
            }
            Bound::Included(start) => {
                if let Some((handle, idx)) = self.raw.lower_bound(start) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
            Bound::Excluded(start) => {
                if let Some((handle, idx)) = self.raw.upper_bound(start) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
        };

        // Find back position based on end bound
        let (back_leaf, back_index) = match range.end_bound() {
            Bound::Unbounded => {
                if let Some(last) = self.raw.last_leaf() {
                    let leaf = self.raw.node(last).as_leaf();
                    let count = leaf.key_count();
                    if count > 0 {
                        (Some(last), count - 1)
                    } else {
                        (None, 0)
                    }
                } else {
                    (None, 0)
                }
            }
            Bound::Included(end) => {
                if let Some((handle, idx)) = self.raw.upper_bound_inclusive(end) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
            Bound::Excluded(end) => {
                if let Some((handle, idx)) = self.raw.lower_bound_exclusive(end) {
                    (Some(handle), idx)
                } else {
                    (None, 0)
                }
            }
        };

        // Check if the range is empty (front position > back position, even across different leaves)
        let (finished, remaining) = match (front_leaf, back_leaf) {
            (Some(front), Some(back)) => {
                let front_key = self.raw.node(front).as_leaf().key(front_index);
                let back_key = self.raw.node(back).as_leaf().key(back_index);
                if front_key > back_key {
                    (true, 0)
                } else {
                    // Compute exact count using order statistics (O(log n) each).
                    // Explicitly specify K as the query type since T is also in scope.
                    let front_rank = self.raw.rank_of::<K>(front_key).unwrap_or(0);
                    let back_rank = self.raw.rank_of::<K>(back_key).unwrap_or(0);
                    (false, back_rank - front_rank + 1)
                }
            }
            _ => (true, 0), // One or both bounds are out of range
        };

        RangeMut {
            tree: &raw mut self.raw,
            front_leaf,
            front_index,
            back_leaf,
            back_index,
            remaining,
            finished,
            _marker: PhantomData,
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut count: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    ///
    /// // count the number of occurrences of letters in the vec
    /// for x in ["a", "b", "a", "c", "a", "b"] {
    ///     count.entry(x).and_modify(|curr| *curr += 1).or_insert(1);
    /// }
    ///
    /// assert_eq!(count["a"], 3);
    /// assert_eq!(count["b"], 2);
    /// assert_eq!(count["c"], 1);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V>
    where
        K: Ord + Clone,
    {
        if let Some((leaf_handle, index)) = self.raw.search(&key) {
            Entry::Occupied(OccupiedEntry {
                key,
                leaf_handle,
                index,
                tree: &mut self.raw,
            })
        } else {
            Entry::Vacant(VacantEntry {
                key,
                tree: &mut self.raw,
            })
        }
    }

    /// Splits the collection into two at the given key. Returns everything after the given key,
    /// including the key. If the key is not present, the split will occur at the nearest
    /// greater key, or return an empty map if no such key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c");
    /// a.insert(17, "d");
    /// a.insert(41, "e");
    ///
    /// let b = a.split_off(&3);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 3);
    ///
    /// assert_eq!(a[&1], "a");
    /// assert_eq!(a[&2], "b");
    ///
    /// assert_eq!(b[&3], "c");
    /// assert_eq!(b[&17], "d");
    /// assert_eq!(b[&41], "e");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m log(n)), where m is the number of elements being split off.
    #[allow(clippy::return_self_not_must_use)]
    pub fn split_off<Q: Ord>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q> + Clone + Ord,
    {
        // Collect keys to move (keys >= split key)
        let keys_to_move: alloc::vec::Vec<K> =
            self.range::<Q, _>((Bound::Included(key), Bound::Unbounded)).map(|(k, _)| k.clone()).collect();

        // Build the new map and remove from self
        let mut other = OSBTreeMap::new();
        for k in keys_to_move {
            // Use remove::<K> to remove by the key type K
            if let Some(v) = self.remove::<K>(&k) {
                other.insert(k, v);
            }
        }
        other
    }

    /// Creates an iterator that visits elements (key-value pairs) in the specified range in
    /// ascending key order and uses a closure to determine if an element
    /// should be removed.
    ///
    /// If the closure returns `true`, the element is removed from the map and
    /// yielded. If the closure returns `false`, or panics, the element remains
    /// in the map and will not be yielded.
    ///
    /// The iterator also lets you mutate the value of each element in the
    /// closure, regardless of whether you choose to keep or remove it.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped without iterating
    /// or the iteration short-circuits, then the remaining elements will be retained.
    /// Use [`retain`] with a negated predicate if you do not need the returned iterator.
    ///
    /// [`retain`]: OSBTreeMap::retain
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// // Splitting a map into even and odd keys, reusing the original map:
    /// let mut map: OSBTreeMap<i32, i32> = (0..8).map(|x| (x, x)).collect();
    /// let evens: OSBTreeMap<_, _> = map.extract_if(.., |k, _v| k % 2 == 0).collect();
    /// let odds = map;
    /// assert_eq!(evens.keys().copied().collect::<Vec<_>>(), [0, 2, 4, 6]);
    /// assert_eq!(odds.keys().copied().collect::<Vec<_>>(), [1, 3, 5, 7]);
    ///
    /// // Splitting a map into low and high halves, reusing the original map:
    /// let mut map: OSBTreeMap<i32, i32> = (0..8).map(|x| (x, x)).collect();
    /// let low: OSBTreeMap<_, _> = map.extract_if(0..4, |_k, _v| true).collect();
    /// let high = map;
    /// assert_eq!(low.keys().copied().collect::<Vec<_>>(), [0, 1, 2, 3]);
    /// assert_eq!(high.keys().copied().collect::<Vec<_>>(), [4, 5, 6, 7]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(m + k log n) where m is the number of elements in the range and k is
    /// the number of elements extracted. All keys in the range are collected
    /// upfront, then each extracted element requires a O(log n) removal.
    pub fn extract_if<F, R>(&mut self, range: R, pred: F) -> ExtractIf<'_, K, V, R, F>
    where
        K: Clone + Ord,
        R: RangeBounds<K>,
        F: FnMut(&K, &mut V) -> bool,
    {
        // Collect keys in range upfront
        let keys: alloc::vec::Vec<K> = self
            .range::<K, _>((range.start_bound().cloned(), range.end_bound().cloned()))
            .map(|(k, _)| k.clone())
            .collect();

        ExtractIf {
            tree: &mut self.raw,
            keys,
            index: 0,
            pred,
            _marker: PhantomData,
        }
    }

    pub(crate) fn extract_if_inner<R>(&mut self, range: R) -> ExtractIfInner<'_, K, V, R>
    where
        K: Clone + Ord,
        R: RangeBounds<K>,
    {
        // Collect keys in range upfront
        let keys: alloc::vec::Vec<K> = self
            .range::<K, _>((range.start_bound().cloned(), range.end_bound().cloned()))
            .map(|(k, _)| k.clone())
            .collect();

        ExtractIfInner {
            tree: &mut self.raw,
            keys,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// Creates a consuming iterator visiting all the keys, in sorted order.
    /// The map cannot be used after calling this.
    /// The iterator element type is `K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(2, "b");
    /// a.insert(1, "a");
    ///
    /// let keys: Vec<_> = a.into_keys().collect();
    /// assert_eq!(keys, [1, 2]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(n) to create the iterator (drains all elements); iteration is O(1) per element.
    pub fn into_keys(mut self) -> IntoKeys<K, V> {
        IntoKeys {
            inner: IntoIter {
                inner: self.raw.drain_to_vec().into_iter(),
            },
        }
    }

    /// Creates a consuming iterator visiting all the values, in order by key.
    /// The map cannot be used after calling this.
    /// The iterator element type is `V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(1, "hello");
    /// a.insert(2, "goodbye");
    ///
    /// let values: Vec<_> = a.into_values().collect();
    /// assert_eq!(values, ["hello", "goodbye"]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(n) to create the iterator (drains all elements); iteration is O(1) per element.
    pub fn into_values(mut self) -> IntoValues<K, V> {
        IntoValues {
            inner: IntoIter {
                inner: self.raw.drain_to_vec().into_iter(),
            },
        }
    }

    /// Gets an iterator over the entries of the map, sorted by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::new();
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{key}: {value}");
    /// }
    ///
    /// let (first_key, first_value) = map.iter().next().unwrap();
    /// assert_eq!((*first_key, *first_value), (1, "a"));
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) to create the iterator; O(1) per iteration step via linked leaves.
    pub fn iter(&self) -> Iter<'_, K, V> {
        let first_leaf = self.raw.first_leaf();
        let last_leaf = self.raw.last_leaf();
        let back_index = if let Some(h) = last_leaf {
            let leaf = self.raw.node(h).as_leaf();
            if leaf.key_count() > 0 {
                leaf.key_count() - 1
            } else {
                0
            }
        } else {
            0
        };
        Iter {
            tree: &raw const self.raw,
            front_leaf: first_leaf,
            front_index: 0,
            back_leaf: last_leaf,
            back_index,
            remaining: self.raw.len(),
            _marker: PhantomData,
        }
    }

    /// Gets a mutable iterator over the entries of the map, sorted by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map = OSBTreeMap::from([
    ///    ("a", 1),
    ///    ("b", 2),
    ///    ("c", 3),
    /// ]);
    ///
    /// // add 10 to the value if the key isn't "a"
    /// for (key, value) in map.iter_mut() {
    ///     if key != &"a" {
    ///         *value += 10;
    ///     }
    /// }
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) to create the iterator; O(1) per iteration step via linked leaves.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let first_leaf = self.raw.first_leaf();
        let last_leaf = self.raw.last_leaf();
        let back_index = if let Some(h) = last_leaf {
            let leaf = self.raw.node(h).as_leaf();
            if leaf.key_count() > 0 {
                leaf.key_count() - 1
            } else {
                0
            }
        } else {
            0
        };
        IterMut {
            tree: &raw mut self.raw,
            front_leaf: first_leaf,
            front_index: 0,
            back_leaf: last_leaf,
            back_index,
            remaining: self.raw.len(),
            _marker: PhantomData,
        }
    }

    /// Gets an iterator over the keys of the map, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(2, "b");
    /// a.insert(1, "a");
    ///
    /// let keys: Vec<_> = a.keys().cloned().collect();
    /// assert_eq!(keys, [1, 2]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) to create the iterator; each iteration step is O(1) amortized.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys {
            inner: self.iter(),
        }
    }

    /// Gets an iterator over the values of the map, in order by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(1, "hello");
    /// a.insert(2, "goodbye");
    ///
    /// let values: Vec<&str> = a.values().cloned().collect();
    /// assert_eq!(values, ["hello", "goodbye"]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) to create the iterator; each iteration step is O(1) amortized.
    pub fn values(&self) -> Values<'_, K, V> {
        Values {
            inner: self.iter(),
        }
    }

    /// Gets a mutable iterator over the values of the map, in order by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// a.insert(1, String::from("hello"));
    /// a.insert(2, String::from("goodbye"));
    ///
    /// for value in a.values_mut() {
    ///     value.push_str("!");
    /// }
    ///
    /// let values: Vec<String> = a.values().cloned().collect();
    /// assert_eq!(values, [String::from("hello!"),
    ///                     String::from("goodbye!")]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1) to create the iterator; each iteration step is O(1) amortized.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub const fn len(&self) -> usize {
        self.raw.len()
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut a = OSBTreeMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}

impl<K: Clone + Ord, V: Clone> Clone for OSBTreeMap<K, V> {
    fn clone(&self) -> Self {
        OSBTreeMap {
            raw: self.raw.clone(),
        }
    }
}

impl<K: Hash, V: Hash> Hash for OSBTreeMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for (k, v) in self {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for OSBTreeMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<K: Eq, V: Eq> Eq for OSBTreeMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for OSBTreeMap<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for OSBTreeMap<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for OSBTreeMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> Default for OSBTreeMap<K, V> {
    fn default() -> Self {
        OSBTreeMap::new()
    }
}

impl<K: Ord + Clone, V> FromIterator<(K, V)> for OSBTreeMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = OSBTreeMap::new();
        map.extend(iter);
        map
    }
}

impl<K: Ord + Clone, V> Extend<(K, V)> for OSBTreeMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Ord + Copy, V: Copy> Extend<(&'a K, &'a V)> for OSBTreeMap<K, V> {
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (&k, &v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K, V> IntoIterator for &'a OSBTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut OSBTreeMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K: Clone + Ord, V> IntoIterator for OSBTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    /// Gets an owning iterator over the entries of the map, sorted by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let map = OSBTreeMap::from([(2, "b"), (1, "a")]);
    /// let mut iter = map.into_iter();
    /// assert_eq!(iter.next(), Some((1, "a")));
    /// assert_eq!(iter.next_back(), Some((2, "b")));
    /// ```
    fn into_iter(mut self) -> IntoIter<K, V> {
        let entries = self.raw.drain_to_vec();
        IntoIter {
            inner: entries.into_iter(),
        }
    }
}

impl<K, Q, V> Index<&Q> for OSBTreeMap<K, V>
where
    K: Borrow<Q> + Ord + Clone,
    Q: ?Sized + Ord,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K: Ord + Clone, V, const N: usize> From<[(K, V); N]> for OSBTreeMap<K, V> {
    fn from(arr: [(K, V); N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let leaf_handle = self.front_leaf?;

        // SAFETY: When remaining > 0 and front_leaf is Some, self.tree is a valid pointer
        // obtained from a live reference in iter().
        let tree = unsafe { &*self.tree };
        let leaf = tree.node(leaf_handle).as_leaf();

        let key = leaf.key(self.front_index);
        let value_handle = leaf.value(self.front_index);
        let value = tree.value(value_handle);

        self.remaining -= 1;
        self.front_index += 1;

        // Move to next leaf if needed
        if self.front_index >= leaf.key_count() {
            self.front_leaf = leaf.next();
            self.front_index = 0;
        }

        Some((key, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let leaf_handle = self.back_leaf?;

        // SAFETY: When remaining > 0 and back_leaf is Some, self.tree is a valid pointer.
        let tree = unsafe { &*self.tree };
        let leaf = tree.node(leaf_handle).as_leaf();

        let key = leaf.key(self.back_index);
        let value_handle = leaf.value(self.back_index);
        let value = tree.value(value_handle);

        self.remaining -= 1;

        // Move to previous element/leaf
        if self.back_index == 0 {
            self.back_leaf = leaf.prev();
            if let Some(prev_handle) = self.back_leaf {
                let prev_leaf = tree.node(prev_handle).as_leaf();
                self.back_index = prev_leaf.key_count().saturating_sub(1);
            }
        } else {
            self.back_index -= 1;
        }

        Some((key, value))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter").field("remaining", &self.remaining).finish()
    }
}

impl<'a, K: 'a, V: 'a> Default for Iter<'a, K, V> {
    /// Creates an empty `osbtree_map::Iter`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::Iter<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Iter {
            // SAFETY: tree is never dereferenced when remaining == 0 and front_leaf/back_leaf
            // are None, so a dangling pointer is safe here.
            tree: core::ptr::NonNull::dangling().as_ptr(),
            front_leaf: None,
            front_index: 0,
            back_leaf: None,
            back_index: 0,
            remaining: 0,
            _marker: PhantomData,
        }
    }
}

impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter {
            tree: self.tree,
            front_leaf: self.front_leaf,
            front_index: self.front_index,
            back_leaf: self.back_leaf,
            back_index: self.back_index,
            remaining: self.remaining,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let leaf_handle = self.front_leaf?;

        // SAFETY: We have exclusive access to the tree through the raw pointer.
        // We're traversing elements in order and never visit the same element twice.
        // Keys live in the nodes arena and values in the values arena (separate allocations).
        // We access nodes and values through separate raw pointers to avoid aliasing violations.
        unsafe {
            let leaf = RawOSBTreeMap::node_ptr(self.tree, leaf_handle).as_leaf();

            let key = leaf.key(self.front_index);
            let value_handle = leaf.value(self.front_index);

            // Access value through raw pointer to the values arena only.
            let value = RawOSBTreeMap::value_mut_ptr(self.tree, value_handle);

            self.remaining -= 1;
            self.front_index += 1;

            // Move to next leaf if needed
            if self.front_index >= leaf.key_count() {
                self.front_leaf = leaf.next();
                self.front_index = 0;
            }

            Some((key, value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> DoubleEndedIterator for IterMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let leaf_handle = self.back_leaf?;

        // SAFETY: Same as in next() - we have exclusive access and never visit same element twice.
        // Keys and values are in separate arenas, accessed independently via raw pointers.
        unsafe {
            let leaf = RawOSBTreeMap::node_ptr(self.tree, leaf_handle).as_leaf();

            let key = leaf.key(self.back_index);
            let value_handle = leaf.value(self.back_index);

            let value = RawOSBTreeMap::value_mut_ptr(self.tree, value_handle);

            self.remaining -= 1;

            // Move to previous element/leaf
            if self.back_index == 0 {
                self.back_leaf = leaf.prev();
                if let Some(prev_handle) = self.back_leaf {
                    let prev_leaf = RawOSBTreeMap::node_ptr(self.tree, prev_handle).as_leaf();
                    self.back_index = prev_leaf.key_count().saturating_sub(1);
                }
            } else {
                self.back_index -= 1;
            }

            Some((key, value))
        }
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K, V> FusedIterator for IterMut<'_, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IterMut").field("remaining", &self.remaining).finish()
    }
}

impl<'a, K: 'a, V: 'a> Default for IterMut<'a, K, V> {
    /// Creates an empty `osbtree_map::IterMut`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::IterMut<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IterMut {
            tree: core::ptr::null_mut(),
            front_leaf: None,
            front_index: 0,
            back_leaf: None,
            back_index: 0,
            remaining: 0,
            _marker: PhantomData,
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IntoIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter").field("len", &self.inner.len()).finish()
    }
}

impl<K, V> Default for IntoIter<K, V> {
    /// Creates an empty `osbtree_map::IntoIter`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::IntoIter<u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoIter {
            inner: alloc::vec::Vec::new().into_iter(),
        }
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for Keys<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for Keys<'_, K, V> {}

impl<K: fmt::Debug, V> fmt::Debug for Keys<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Keys").field("remaining", &self.inner.remaining).finish()
    }
}

impl<K, V> Default for Keys<'_, K, V> {
    /// Creates an empty `osbtree_map::Keys`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::Keys<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Keys {
            inner: Iter::default(),
        }
    }
}

impl<K, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for Values<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for Values<'_, K, V> {}

impl<K, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V> Default for Values<'_, K, V> {
    /// Creates an empty `osbtree_map::Values`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::Values<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Values {
            inner: Iter::default(),
        }
    }
}

impl<K, V: fmt::Debug> fmt::Debug for Values<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Values").field("remaining", &self.inner.remaining).finish()
    }
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for ValuesMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

impl<K, V: fmt::Debug> fmt::Debug for ValuesMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValuesMut").field("remaining", &self.inner.remaining).finish()
    }
}

impl<K, V> Default for ValuesMut<'_, K, V> {
    /// Creates an empty `osbtree_map::ValuesMut`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::ValuesMut<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        ValuesMut {
            inner: IterMut::default(),
        }
    }
}

impl<K: Clone + Ord, V> Iterator for IntoKeys<K, V> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K: Clone + Ord, V> DoubleEndedIterator for IntoKeys<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

impl<K: Clone + Ord, V> ExactSizeIterator for IntoKeys<K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K: Clone + Ord, V> FusedIterator for IntoKeys<K, V> {}

impl<K: fmt::Debug, V> fmt::Debug for IntoKeys<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoKeys").field("len", &self.inner.len()).finish()
    }
}

impl<K, V> Default for IntoKeys<K, V> {
    /// Creates an empty `osbtree_map::IntoKeys`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::IntoKeys<u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoKeys {
            inner: IntoIter::default(),
        }
    }
}

impl<K: Clone + Ord, V> Iterator for IntoValues<K, V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K: Clone + Ord, V> DoubleEndedIterator for IntoValues<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K: Clone + Ord, V> ExactSizeIterator for IntoValues<K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K: Clone + Ord, V> FusedIterator for IntoValues<K, V> {}

impl<K, V: fmt::Debug> fmt::Debug for IntoValues<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoValues").field("len", &self.inner.len()).finish()
    }
}

impl<K, V> Default for IntoValues<K, V> {
    /// Creates an empty `osbtree_map::IntoValues`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::IntoValues<u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoValues {
            inner: IntoIter::default(),
        }
    }
}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've already exhausted the range
        if self.finished {
            return None;
        }

        let front_leaf = self.front_leaf?;
        let back_leaf = self.back_leaf?;

        // Check if front has passed back (same leaf, front index > back index)
        if front_leaf == back_leaf && self.front_index > self.back_index {
            self.finished = true;
            return None;
        }

        // Check if this is the last element (front and back at same position)
        let is_last = front_leaf == back_leaf && self.front_index == self.back_index;

        // SAFETY: When the range is not finished and leaves are Some, self.tree is a valid pointer.
        let tree = unsafe { &*self.tree };
        let leaf = tree.node(front_leaf).as_leaf();
        let key = leaf.key(self.front_index);
        let value_handle = leaf.value(self.front_index);
        let value = tree.value(value_handle);

        // Advance front position
        self.front_index += 1;
        if self.front_index >= leaf.key_count() {
            self.front_leaf = leaf.next();
            self.front_index = 0;
        }

        if is_last {
            self.finished = true;
        }

        if self.remaining > 0 {
            self.remaining -= 1;
        }

        Some((key, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Exact count is known via order statistics
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> DoubleEndedIterator for Range<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // Check if we've already exhausted the range
        if self.finished {
            return None;
        }

        let front_leaf = self.front_leaf?;
        let back_leaf = self.back_leaf?;

        // Check if back has passed front (same leaf, back index < front index)
        if front_leaf == back_leaf && self.back_index < self.front_index {
            self.finished = true;
            return None;
        }

        // Check if this is the last element (front and back at same position)
        let is_last = front_leaf == back_leaf && self.front_index == self.back_index;

        // SAFETY: When the range is not finished and leaves are Some, self.tree is a valid pointer.
        let tree = unsafe { &*self.tree };
        let leaf = tree.node(back_leaf).as_leaf();
        let key = leaf.key(self.back_index);
        let value_handle = leaf.value(self.back_index);
        let value = tree.value(value_handle);

        // Move back position backward
        if self.back_index == 0 {
            self.back_leaf = leaf.prev();
            if let Some(prev) = self.back_leaf {
                let prev_leaf = tree.node(prev).as_leaf();
                self.back_index = prev_leaf.key_count().saturating_sub(1);
            }
        } else {
            self.back_index -= 1;
        }

        if is_last {
            self.finished = true;
        }

        if self.remaining > 0 {
            self.remaining -= 1;
        }

        Some((key, value))
    }
}

impl<K, V> FusedIterator for Range<'_, K, V> {}

impl<K, V> ExactSizeIterator for Range<'_, K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Range<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Range").field("remaining", &self.remaining).finish()
    }
}

impl<K, V> Default for Range<'_, K, V> {
    /// Creates an empty `osbtree_map::Range`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::Range<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        Range {
            // SAFETY: tree is never dereferenced when finished == true and leaves are None.
            tree: core::ptr::NonNull::dangling().as_ptr(),
            front_leaf: None,
            front_index: 0,
            back_leaf: None,
            back_index: 0,
            remaining: 0,
            finished: true,
            _marker: PhantomData,
        }
    }
}

impl<K, V> Clone for Range<'_, K, V> {
    fn clone(&self) -> Self {
        Range {
            tree: self.tree,
            front_leaf: self.front_leaf,
            front_index: self.front_index,
            back_leaf: self.back_leaf,
            back_index: self.back_index,
            remaining: self.remaining,
            finished: self.finished,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've already exhausted the range
        if self.finished {
            return None;
        }

        let front_leaf = self.front_leaf?;
        let back_leaf = self.back_leaf?;

        // Check if front has passed back (same leaf, front index > back index)
        if front_leaf == back_leaf && self.front_index > self.back_index {
            self.finished = true;
            return None;
        }

        // Check if this is the last element (front and back at same position)
        let is_last = front_leaf == back_leaf && self.front_index == self.back_index;

        // SAFETY: We have exclusive access to the tree through the raw pointer.
        // Keys and values are in separate arenas, accessed independently via raw pointers.
        unsafe {
            let leaf = RawOSBTreeMap::node_ptr(self.tree, front_leaf).as_leaf();
            let key = leaf.key(self.front_index);
            let value_handle = leaf.value(self.front_index);

            let value = RawOSBTreeMap::value_mut_ptr(self.tree, value_handle);

            // Advance front position
            self.front_index += 1;
            if self.front_index >= leaf.key_count() {
                self.front_leaf = leaf.next();
                self.front_index = 0;
            }

            if is_last {
                self.finished = true;
            }

            if self.remaining > 0 {
                self.remaining -= 1;
            }

            Some((key, value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Exact count is known via order statistics
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> DoubleEndedIterator for RangeMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // Check if we've already exhausted the range
        if self.finished {
            return None;
        }

        let front_leaf = self.front_leaf?;
        let back_leaf = self.back_leaf?;

        // Check if back has passed front (same leaf, back index < front index)
        if front_leaf == back_leaf && self.back_index < self.front_index {
            self.finished = true;
            return None;
        }

        // Check if this is the last element (front and back at same position)
        let is_last = front_leaf == back_leaf && self.front_index == self.back_index;

        // SAFETY: We have exclusive access to the tree through the raw pointer.
        // Keys and values are in separate arenas, accessed independently via raw pointers.
        unsafe {
            let leaf = RawOSBTreeMap::node_ptr(self.tree, back_leaf).as_leaf();
            let key = leaf.key(self.back_index);
            let value_handle = leaf.value(self.back_index);

            let value = RawOSBTreeMap::value_mut_ptr(self.tree, value_handle);

            // Move back position backward
            if self.back_index == 0 {
                self.back_leaf = leaf.prev();
                if let Some(prev) = self.back_leaf {
                    let prev_leaf = RawOSBTreeMap::node_ptr(self.tree, prev).as_leaf();
                    self.back_index = prev_leaf.key_count().saturating_sub(1);
                }
            } else {
                self.back_index -= 1;
            }

            if is_last {
                self.finished = true;
            }

            if self.remaining > 0 {
                self.remaining -= 1;
            }

            Some((key, value))
        }
    }
}

impl<K, V> FusedIterator for RangeMut<'_, K, V> {}

impl<K, V> ExactSizeIterator for RangeMut<'_, K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for RangeMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeMut").field("remaining", &self.remaining).finish()
    }
}

impl<K, V> Default for RangeMut<'_, K, V> {
    /// Creates an empty `osbtree_map::RangeMut`.
    ///
    /// ```
    /// # use wabi_tree::osbtree_map;
    /// let iter: osbtree_map::RangeMut<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        RangeMut {
            tree: core::ptr::null_mut(),
            front_leaf: None,
            front_index: 0,
            back_leaf: None,
            back_index: 0,
            remaining: 0,
            finished: true,
            _marker: PhantomData,
        }
    }
}

impl<K, V, R, F> fmt::Debug for ExtractIf<'_, K, V, R, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf")
            .field("index", &self.index)
            .field("remaining", &(self.keys.len() - self.index))
            .finish()
    }
}

impl<K, V, R, F> Iterator for ExtractIf<'_, K, V, R, F>
where
    K: Clone + Ord,
    R: RangeBounds<K>,
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.keys.len() {
            let key = &self.keys[self.index];
            self.index += 1;

            // Get mutable access to the value to check predicate.
            // The borrow must end before we call remove_entry, so we scope it.
            let should_remove = self.tree.get_mut(key).is_some_and(|value| (self.pred)(key, value));
            if should_remove {
                // Remove and return
                if let Some((k, v)) = self.tree.remove_entry(key) {
                    return Some((k, v));
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.keys.len() - self.index))
    }
}

impl<K, V, R, F> FusedIterator for ExtractIf<'_, K, V, R, F>
where
    K: Clone + Ord,
    R: RangeBounds<K>,
    F: FnMut(&K, &mut V) -> bool,
{
}

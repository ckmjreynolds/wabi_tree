use core::fmt;
use core::mem;

use crate::raw::{Handle, RawOSBTreeMap};

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`crate::OSBTreeMap`].
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
/// use wabi_tree::osbtree_map::Entry;
///
/// let mut map = OSBTreeMap::new();
///
/// match map.entry("oz") {
///     Entry::Vacant(v) => {
///         v.insert(1);
///     }
///     Entry::Occupied(mut o) => {
///         *o.get_mut() += 1;
///     }
/// }
/// assert_eq!(map["oz"], 1);
/// ```
///
/// [`entry`]: crate::OSBTreeMap::entry
pub enum Entry<'a, K: 'a, V: 'a> {
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),

    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),
}

impl<K: fmt::Debug + Ord + Clone, V: fmt::Debug> fmt::Debug for Entry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Vacant(v) => f.debug_tuple("Entry").field(v).finish(),
            Entry::Occupied(o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// A view into a vacant entry in a `OSBTreeMap`.
/// It is part of the [`Entry`] enum.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
/// use wabi_tree::osbtree_map::Entry;
///
/// let mut map = OSBTreeMap::new();
///
/// if let Entry::Vacant(v) = map.entry("oz") {
///     v.insert(5);
/// }
/// assert_eq!(map["oz"], 5);
/// ```
pub struct VacantEntry<'a, K, V> {
    pub(crate) key: K,
    pub(crate) tree: &'a mut RawOSBTreeMap<K, V>,
}

impl<K: fmt::Debug + Ord + Clone, V> fmt::Debug for VacantEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantEntry").field("key", &self.key).finish()
    }
}

/// A view into an occupied entry in a `OSBTreeMap`.
/// It is part of the [`Entry`] enum.
///
/// # Examples
///
/// ```
/// use wabi_tree::OSBTreeMap;
/// use wabi_tree::osbtree_map::Entry;
///
/// let mut map = OSBTreeMap::new();
/// map.insert("oz", 1);
///
/// if let Entry::Occupied(mut o) = map.entry("oz") {
///     *o.get_mut() += 1;
/// }
/// assert_eq!(map["oz"], 2);
/// ```
pub struct OccupiedEntry<'a, K, V> {
    pub(crate) key: K,
    pub(crate) leaf_handle: Handle,
    pub(crate) index: usize,
    pub(crate) tree: &'a mut RawOSBTreeMap<K, V>,
}

impl<K: fmt::Debug + Ord + Clone, V: fmt::Debug> fmt::Debug for OccupiedEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry").field("key", self.key()).field("value", self.get()).finish()
    }
}

impl<'a, K: Ord + Clone, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) if vacant (insertion), O(1) if occupied.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, String> = OSBTreeMap::new();
    /// let s = "hoho".to_string();
    ///
    /// map.entry("poneyland").or_insert_with(|| s);
    ///
    /// assert_eq!(map["poneyland"], "hoho".to_string());
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) if vacant (insertion), O(1) if occupied.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of the default function.
    ///
    /// This method allows for generating key-derived values for insertion by providing the default
    /// function a reference to the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying the key is
    /// unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    ///
    /// map.entry("poneyland").or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(map["poneyland"], 9);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) if vacant (insertion), O(1) if occupied.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                let value = default(&v.key);
                v.insert(value)
            }
        }
    }

    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[allow(clippy::must_use_candidate)]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(o) => o.key(),
            Entry::Vacant(v) => v.key(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(map["poneyland"], 42);
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(map["poneyland"], 43);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[allow(clippy::return_self_not_must_use)]
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Entry::Occupied(ref mut o) = self {
            f(o.get_mut());
        }
        self
    }

    /// Sets the value of the entry, and returns an `OccupiedEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, String> = OSBTreeMap::new();
    /// let entry = map.entry("poneyland").insert_entry("hoho".to_string());
    ///
    /// assert_eq!(entry.key(), &"poneyland");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) if vacant (insertion), O(1) if occupied.
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(mut o) => {
                o.insert(value);
                o
            }
            Entry::Vacant(v) => v.insert_entry(value),
        }
    }
}

impl<'a, K: Ord + Clone, V: Default> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, Option<usize>> = OSBTreeMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map["poneyland"], None);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n) if vacant (insertion), O(1) if occupied.
    #[allow(clippy::must_use_candidate)]
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(V::default()),
        }
    }
}

impl<'a, K: Ord + Clone, V> VacantEntry<'a, K, V> {
    /// Gets a reference to the key that would be used when inserting a value
    /// through the `VacantEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[allow(clippy::must_use_candidate)]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take ownership of the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    ///
    /// if let Entry::Vacant(v) = map.entry("poneyland") {
    ///     v.into_key();
    /// }
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[allow(clippy::must_use_candidate)]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    ///
    /// # Panics
    ///
    /// Panics if the tree's internal state is corrupted (should never happen in normal use).
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, u32> = OSBTreeMap::new();
    ///
    /// if let Entry::Vacant(o) = map.entry("poneyland") {
    ///     o.insert(37);
    /// }
    /// assert_eq!(map["poneyland"], 37);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn insert(self, value: V) -> &'a mut V {
        let key = self.key.clone();
        self.tree.insert(self.key, value);
        // Get a mutable reference to the inserted value
        // SAFETY: We know the key exists because we just inserted it
        self.tree.get_mut(&key).expect("just inserted")
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns an `OccupiedEntry`.
    ///
    /// # Panics
    ///
    /// Panics if the tree's internal state is corrupted (should never happen in normal use).
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, u32> = OSBTreeMap::new();
    ///
    /// if let Entry::Vacant(o) = map.entry("poneyland") {
    ///     let entry = o.insert_entry(37);
    ///     assert_eq!(entry.get(), &37);
    /// }
    /// assert_eq!(map["poneyland"], 37);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        self.tree.insert(self.key.clone(), value);
        let (leaf_handle, index) = self.tree.search(&self.key).expect("just inserted");
        OccupiedEntry {
            key: self.key,
            leaf_handle,
            index,
            tree: self.tree,
        }
    }
}

impl<'a, K: Ord + Clone, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the key in the entry.
    ///
    /// Note: This returns the key that is actually stored in the map, not the
    /// key that was used to probe the entry. For types where `Ord` is based on
    /// a subset of fields, these may differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub fn key(&self) -> &K {
        // Return the key stored in the tree, not the probe key
        let leaf = self.tree.node(self.leaf_handle).as_leaf();
        leaf.key(self.index)
    }

    /// Take ownership of the key and value from the map.
    ///
    /// # Panics
    ///
    /// Panics if the entry no longer exists (should never happen in normal use).
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     // We delete the entry from the map.
    ///     o.remove_entry();
    /// }
    ///
    /// // If now try to get the value, it will panic:
    /// // println!("{}", map["poneyland"]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    #[allow(clippy::must_use_candidate)]
    pub fn remove_entry(self) -> (K, V) {
        self.tree.remove_entry(&self.key).expect("entry must exist")
    }

    /// Gets a reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     assert_eq!(o.get(), &12);
    /// }
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use]
    pub fn get(&self) -> &V {
        let leaf = self.tree.node(self.leaf_handle).as_leaf();
        let value_handle = leaf.value(self.index);
        self.tree.value(value_handle)
    }

    /// Gets a mutable reference to the value in the entry.
    ///
    /// If you need a reference to the `OccupiedEntry` that may outlive the
    /// destruction of the `Entry` value, see [`into_mut`].
    ///
    /// [`into_mut`]: OccupiedEntry::into_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     *o.get_mut() += 10;
    ///     assert_eq!(*o.get(), 22);
    ///
    ///     // We can use the same Entry multiple times.
    ///     *o.get_mut() += 2;
    /// }
    /// assert_eq!(map["poneyland"], 24);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    pub fn get_mut(&mut self) -> &mut V {
        let leaf = self.tree.node(self.leaf_handle).as_leaf();
        let value_handle = leaf.value(self.index);
        self.tree.value_mut(value_handle)
    }

    /// Converts the entry into a mutable reference to its value.
    ///
    /// If you need multiple references to the `OccupiedEntry`, see [`get_mut`].
    ///
    /// [`get_mut`]: OccupiedEntry::get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     *o.into_mut() += 10;
    /// }
    /// assert_eq!(map["poneyland"], 22);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_mut(self) -> &'a mut V {
        let leaf = self.tree.node(self.leaf_handle).as_leaf();
        let value_handle = leaf.value(self.index);
        self.tree.value_mut(value_handle)
    }

    /// Sets the value of the entry with the `OccupiedEntry`'s key,
    /// and returns the entry's old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     assert_eq!(o.insert(15), 12);
    /// }
    /// assert_eq!(map["poneyland"], 15);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(1)
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Takes the value of the entry out of the map, and returns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use wabi_tree::OSBTreeMap;
    /// use wabi_tree::osbtree_map::Entry;
    ///
    /// let mut map: OSBTreeMap<&str, usize> = OSBTreeMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     assert_eq!(o.remove(), 12);
    /// }
    /// // If we try to get "poneyland"'s value, it'll panic:
    /// // println!("{}", map["poneyland"]);
    /// ```
    ///
    /// # Complexity
    ///
    /// O(log n)
    #[allow(clippy::must_use_candidate)]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

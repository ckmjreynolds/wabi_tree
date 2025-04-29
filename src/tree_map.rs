use std::{
    fmt,
    fmt::{Debug, Formatter},
    ops::{Index, IndexMut},
};

// DEFAULT_CAPACITY - SWAG for an initial capacity when not specified.
static _DEFAULT_CAPACITY: usize = 1024;

// ALPHA - SWAG for α, the balance factor.
// https://en.wikipedia.org/wiki/Weight-balanced_tree
//
// > Larger values of α produce "more balanced" trees, but not all values of
// > α are appropriate; Nievergelt and Reingold proved that: α < 1 - (1 / √2)
// >
// > Later work showed a lower bound of 2⁄11 for α.
//
// These statements suggest that α should be between 0.1818 and 0.2929.
static _ALPHA: f32 = 0.2928;

pub struct WabiTreeMap<K, V> {
    _nodes: Vec<Option<Node<K, V>>>,
    _free: Vec<usize>,
    _root: Option<usize>,
}

pub struct Iter<'a, K, V> {
    _tree: &'a WabiTreeMap<K, V>,
    _next: Option<usize>,
    _back: Option<usize>,
}

// Debug trait implementation *************************************************
impl<K, V> Debug for WabiTreeMap<K, V> {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

// Construction and initialization ********************************************
impl<K, V> WabiTreeMap<K, V> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn with_capacity(_capacity: usize) -> Self {
        todo!()
    }
}

// Default trait implementation ***********************************************
impl<K, V> Default for WabiTreeMap<K, V> {
    fn default() -> Self {
        todo!()
    }
}

// State / capacity ***********************************************************
impl<K, V> WabiTreeMap<K, V> {
    pub fn len(&self) -> usize {
        todo!()
    }

    pub fn is_empty(&self) -> bool {
        todo!()
    }

    pub fn capacity(&self) -> usize {
        todo!()
    }
}

// CRUD operations ************************************************************
impl<K, V> WabiTreeMap<K, V> {
    pub fn insert(&mut self, _key: K, _value: V) -> Option<V> {
        todo!()
    }

    pub fn get(&self, _key: &K) -> Option<&V> {
        todo!()
    }

    pub fn get_mut(&mut self, _key: &K) -> Option<&mut V> {
        todo!()
    }

    pub fn contains_key(&self, _key: &K) -> bool {
        todo!()
    }

    pub fn remove(&mut self, _key: &K) -> Option<V> {
        todo!()
    }

    pub fn select(&self, _index: &usize) -> Option<(K, V)> {
        todo!()
    }

    pub fn rank(&self, _key: &K) -> usize {
        todo!()
    }

    pub fn clear(&mut self) {
        todo!()
    }
}

// Index trait implementation *************************************************
impl<K, V> Index<&K> for WabiTreeMap<K, V> {
    type Output = V;

    fn index(&self, _key: &K) -> &Self::Output {
        todo!()
    }
}

// IndexMut trait implementation **********************************************
impl<K, V> IndexMut<&K> for WabiTreeMap<K, V> {
    fn index_mut(&mut self, _key: &K) -> &mut Self::Output {
        todo!()
    }
}

// Iteration ******************************************************************
impl<K, V> WabiTreeMap<K, V> {
    pub fn iter(&self) -> Iter<'_, K, V> {
        todo!()
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        todo!()
    }
}

impl<K, V> DoubleEndedIterator for Iter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'a, K, V> IntoIterator for &'a WabiTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Rage iterator **************************************************************
impl<K, V> WabiTreeMap<K, V> {
    pub fn range(&self, _lo: &K, _hi: &K) -> Iter<'_, K, V> {
        todo!()
    }

    pub fn range_by_rank(&self, _lo: usize, _hi: usize) -> Iter<'_, K, V> {
        todo!()
    }
}

// Node ***********************************************************************
struct Node<K, V> {
    _key: K,
    _val: V,

    _size: usize,
    _parent: Option<usize>,
    _left: Option<usize>,
    _right: Option<usize>,
}

// Construction and initialization ********************************************
impl<K, V> Node<K, V> {
    fn _new(_key: K, _val: V, _parent: Option<usize>) -> Self {
        todo!()
    }
}

// Private helper functions ***************************************************

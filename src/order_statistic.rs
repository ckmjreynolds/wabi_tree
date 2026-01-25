/// A zero-based rank into the sorted order of a map or set.
///
/// This is an order-statistic extension and is not part of the standard
/// `BTreeMap` or `BTreeSet` APIs.
///
/// # Examples
///
/// ```
/// use wabi_tree::{OSBTreeMap, Rank};
///
/// let mut map = OSBTreeMap::new();
/// map.insert("a", 10);
/// map.insert("b", 20);
///
/// assert_eq!(map[Rank(0)], 10);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rank(pub usize);

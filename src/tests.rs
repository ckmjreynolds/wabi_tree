use super::*;
use pretty_assertions::assert_eq;

#[test]
fn test_empty_map() {
    let map = WabiTreeMap::<char, u8>::default();
    assert_eq!(map.len(), 0);
    assert_eq!(map.is_empty(), true);
    assert!(map.capacity() >= 1);
}

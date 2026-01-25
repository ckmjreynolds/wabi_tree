use core::num::NonZero;

use super::handle::Handle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct Size(Handle);

impl Size {
    pub(crate) const MAX: usize = Handle::MAX;
    pub(crate) const ZERO: Self = Self::from_usize(0);

    #[inline]
    pub(crate) const fn from_usize(size: usize) -> Self {
        assert!(size <= Self::MAX, "`Size::from_usize()` - `size` > `Size::MAX`!");
        Self(Handle::from_index(size))
    }

    #[inline]
    pub(crate) const fn to_usize(self) -> usize {
        self.0.to_index()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use static_assertions::assert_eq_size;

    // Verify our assumptions about `Size` and the niche optimization.
    assert_eq_size!(Size, Option<Size>);
    assert_eq_size!(Size, Handle);

    #[test]
    #[should_panic(expected = "`Size::from_usize()` - `size` > `Size::MAX`!")]
    fn invalid_size() {
        let _ = Size::from_usize(Size::MAX + 1);
    }

    proptest! {
        #[test]
        fn size_round_trip(index in 0..=Size::MAX) {
            let size = Size::from_usize(index);
            assert_eq!(size.to_usize(), index);
        }
    }
}

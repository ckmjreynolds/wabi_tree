use core::num::NonZero;

#[cfg(test)]
type RawHandle = u16;
#[cfg(not(test))]
type RawHandle = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct Handle(NonZero<RawHandle>);

impl Handle {
    pub(crate) const MAX: usize = (RawHandle::MAX - 1) as usize;

    #[inline]
    pub(crate) const fn from_index(index: usize) -> Self {
        assert!(index <= Self::MAX, "`Handle::from_index()` - `index` > `Handle::MAX`!");
        // SAFETY: `index + 1` cannot be zero and cannot overflow.
        Self(NonZero::new((index + 1) as RawHandle).unwrap())
    }

    #[inline]
    pub(crate) const fn to_index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    // Verify our assumptions about `Handle` and the niche optimization.
    const _: () = assert!(size_of::<Handle>() == size_of::<Option<Handle>>());
    const _: () = assert!(size_of::<Handle>() == size_of::<RawHandle>());

    #[test]
    #[should_panic(expected = "`Handle::from_index()` - `index` > `Handle::MAX`!")]
    fn invalid_handle() {
        let _ = Handle::from_index(Handle::MAX + 1);
    }

    proptest! {
        #[test]
        fn handle_round_trip(index in 0..=Handle::MAX) {
            let handle = Handle::from_index(index);
            assert_eq!(handle.to_index(), index);
        }
    }
}

use alloc::vec::Vec;

use super::handle::Handle;

#[derive(Clone)]
pub(crate) struct Arena<T> {
    slots: Vec<Option<T>>,
    free: Vec<Handle>,
}

impl<T> Arena<T> {
    pub(crate) const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    pub(crate) const fn len(&self) -> usize {
        self.slots.len().saturating_sub(self.free.len())
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn alloc(&mut self, element: T) -> Handle {
        if let Some(h) = self.free.pop() {
            // Reuse a free slot/handle.
            self.slots[h.to_index()] = Some(element);
            h
        } else {
            // Use strict less-than to ensure total element count doesn't exceed Size::MAX.
            // Size::MAX == Handle::MAX, so we need slots.len() < Handle::MAX before push,
            // which means at most Handle::MAX elements after push.
            assert!(
                self.slots.len() < Handle::MAX,
                "`Arena::alloc()` - arena is at maximum capacity ({})",
                Handle::MAX
            );
            // Allocate a new slot/handle.
            self.slots.push(Some(element));
            Handle::from_index(self.slots.len() - 1)
        }
    }

    #[inline]
    pub(crate) fn get(&self, handle: Handle) -> &T {
        self.slots[handle.to_index()].as_ref().expect("`Arena::get()` - `handle` is invalid!")
    }

    /// Returns a reference to an element by handle from a raw pointer.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, allocated `Arena<T>`.
    #[inline]
    pub(crate) unsafe fn get_ptr<'a>(ptr: *const Self, handle: Handle) -> &'a T {
        // SAFETY: Caller guarantees ptr is valid. We only read from the slots field.
        // The explicit reference is intentional to index into the Vec.
        unsafe { (&(*ptr).slots)[handle.to_index()].as_ref().expect("`Arena::get_ptr()` - `handle` is invalid!") }
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, handle: Handle) -> &mut T {
        self.slots[handle.to_index()].as_mut().expect("`Arena::get_mut()` - `handle` is invalid!")
    }

    pub(crate) fn take(&mut self, handle: Handle) -> T {
        let element = self.slots[handle.to_index()].take().expect("`Arena::take()` - `handle` is invalid!");
        self.free.push(handle);
        element
    }

    pub(crate) fn free(&mut self, handle: Handle) {
        drop(self.take(handle));
    }

    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn arena_capacity() {
        let arena: Arena<u32> = Arena::with_capacity(10);
        assert_eq!(arena.capacity(), 10);
    }

    proptest! {
        #[test]
        fn arena_behaves_like_vec(operations in prop::collection::vec(strategy(), 0..256)) {
            let mut model: Vec<(Handle, u32)> = Vec::new();
            let mut arena: Arena<u32> = Arena::new();

            for operation in operations {
                match operation {
                    Operation::Alloc(value) => {
                        let handle = arena.alloc(value);
                        model.push((handle, value));
                    }
                    Operation::Get(which) => {
                        if model.is_empty() {
                            continue;
                        }

                        let index = which % model.len();
                        let handle = model[index].0;
                        prop_assert_eq!(*arena.get(handle), model[index].1);
                    }
                    Operation::GetMut(which, value) => {
                        if model.is_empty() {
                            continue;
                        }

                        let index = which % model.len();
                        let handle = model[index].0;
                        *arena.get_mut(handle) = value;
                        model[index].1 = value;
                    }
                    Operation::Take(which) => {
                        if model.is_empty() {
                            continue;
                        }

                        let index = which % model.len();
                        let handle = model[index].0;
                        let value1 = arena.take(handle);
                        let (_, value2) = model.swap_remove(index);
                        prop_assert_eq!(value1, value2);
                    }
                    Operation::Free(which) => {
                        if model.is_empty() {
                            continue;
                        }

                        let index = which % model.len();
                        let handle = model[index].0;
                        arena.free(handle);
                        model.swap_remove(index);
                    }
                    Operation::Clear => {
                        arena.clear();
                        model.clear();
                    }
                }

                prop_assert_eq!(arena.len(), model.len());
                prop_assert_eq!(arena.is_empty(), model.is_empty());

                for &(handle, value) in &model {
                    prop_assert_eq!(*arena.get(handle), value);
                }
            }
        }
    }

    #[derive(Clone, Debug)]
    enum Operation {
        Alloc(u32),
        Get(usize),
        GetMut(usize, u32),
        Take(usize),
        Free(usize),
        Clear,
    }

    fn strategy() -> impl Strategy<Value = Operation> {
        prop_oneof![
            20 => any::<u32>().prop_map(Operation::Alloc),
            5 => any::<usize>().prop_map(Operation::Get),
            5 => (any::<usize>(), any::<u32>()).prop_map(|(which, value)| Operation::GetMut(which, value)),
            5 => any::<usize>().prop_map(Operation::Take),
            5 => any::<usize>().prop_map(Operation::Free),
            1 => Just(Operation::Clear),
        ]
    }
}

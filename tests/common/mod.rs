//! Fixtures shared by the map and set integration tests.
#![allow(dead_code)]

/// A key that compares by `key` alone, so `tag` shows which of several equal keys was kept.
#[derive(Clone, Copy, Debug)]
pub struct Tagged {
    pub key: u8,
    pub tag: usize,
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Tagged {}

impl PartialOrd for Tagged {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tagged {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

pub const fn tagged(key: u8, tag: usize) -> Tagged {
    Tagged {
        key,
        tag,
    }
}

/// Helper function to generate deterministic pseudo-random keys using LCG.
pub fn random_keys_deterministic(n: usize) -> Vec<i64> {
    let mut keys = Vec::with_capacity(n);
    let mut x: u64 = 12345; // Fixed seed for reproducibility
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        keys.push((x >> 33) as i64);
    }
    keys
}

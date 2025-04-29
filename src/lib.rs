#![forbid(unsafe_code)]
#![warn(unused)]
#![warn(clippy::all)]

mod tree_map;

#[cfg(test)]
mod tests;

pub use tree_map::WabiTreeMap;

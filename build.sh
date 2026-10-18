#!/bin/bash
clear
cargo +stable fmt || exit
cargo +stable clippy --all-targets -- -Dwarnings || exit
cargo +stable test --release || exit

# Check the unsafe code paths for undefined behavior under both aliasing models.
cargo +nightly miri test --test miri || exit
MIRIFLAGS=-Zmiri-tree-borrows cargo +nightly miri test --test miri || exit

# Use nightly to get coverage reports.
cargo +nightly llvm-cov --doctests --branch --lcov --output-path lcov.info || exit
cargo +nightly llvm-cov report 

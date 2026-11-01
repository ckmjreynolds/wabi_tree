#!/bin/bash
clear

# Cap parallel Miri tests (each one pins a core); override with NEXTEST_TEST_THREADS=N ./build.sh.
export NEXTEST_TEST_THREADS=${NEXTEST_TEST_THREADS:-4}

cargo +stable fmt || exit
cargo +stable clippy --all-targets -- -Dwarnings || exit
cargo +stable test --release || exit

# Check the unsafe code paths for undefined behavior under both aliasing models. Miri is
# single-threaded, so nextest runs each test in its own process to use every core.
cargo +nightly miri nextest run --test miri || exit
MIRIFLAGS=-Zmiri-tree-borrows cargo +nightly miri nextest run --test miri || exit

# Use nightly to get coverage reports.
cargo +nightly llvm-cov --doctests --branch --lcov --output-path lcov.info || exit
cargo +nightly llvm-cov report 

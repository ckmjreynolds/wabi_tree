#!/bin/bash
clear

# Cap nextest at the four cores a CI runner has (a Miri test pins one each, and using every
# core here is loud); override with NEXTEST_TEST_THREADS=N ./build.sh.
export NEXTEST_TEST_THREADS=${NEXTEST_TEST_THREADS:-4}

cargo +stable fmt || exit
cargo +stable clippy --all-targets -- -Dwarnings || exit
# nextest doesn't run doc tests, so they get their own pass.
cargo +stable test --doc --release || exit
cargo +stable nextest run --release || exit

# Check the unsafe code paths for undefined behavior under both aliasing models. Miri is
# single-threaded, so nextest runs each test in its own process to use every core.
cargo +nightly miri nextest run --test miri || exit
MIRIFLAGS=-Zmiri-tree-borrows cargo +nightly miri nextest run --test miri || exit

# Use nightly to get coverage reports: collect from the nextest and doc test runs, then report
# on both together.
cargo +nightly llvm-cov clean --workspace || exit
cargo +nightly llvm-cov --no-report --branch nextest || exit
cargo +nightly llvm-cov --no-report --branch --doc || exit
cargo +nightly llvm-cov report --doctests --branch --lcov --output-path lcov.info || exit
cargo +nightly llvm-cov report

#!/bin/bash
clear
cargo +stable fmt || exit
cargo +stable clippy --all-targets -- -Dwarnings || exit

# Use nightly to get coverage reports.
cargo +nightly llvm-cov --doctests --branch --lcov --output-path lcov.info || exit
cargo +nightly llvm-cov report 

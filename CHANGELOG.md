# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Changed
- Relax exact (`=`) dependency pins to caret requirements so downstream crates can resolve compatible versions.
- Bump `smallvec` dependency from 1.15.1 to 1.16.3.
- Bump `proptest` dev-dependency from 1.10.0 to 1.11.0 (1.12 requires Rust 1.88, above the 1.87 MSRV).
- Refresh `Cargo.lock` to the latest compatible versions.
- Fix new Clippy 1.99 lints (`ref_as_ptr`, `for_kv_map`).

### CI
- Update GitHub Actions dependencies (actions/checkout v7, codecov-action v7).

## [0.1.2] - 2026-02-10

### Changed
- Bump `criterion` dev-dependency from 0.8.1 to 0.8.2.
- Bump `proptest` dev-dependency from 1.9.0 to 1.10.0.

## [0.1.1] - 2026-01-27

### Changed
- Improve test coverage, including doctests in coverage runs.

### CI
- Update GitHub Actions dependencies (actions/checkout v6, codecov-action v5).
- General CI and repository housekeeping.

## [0.1.0] - 2026-01-25

### Added
- Initial public release of `OSBTreeMap` and `OSBTreeSet` with rank operations.

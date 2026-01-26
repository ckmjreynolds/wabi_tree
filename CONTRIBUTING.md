# Contributing

Thanks for your interest in contributing to `wabi_tree`!

## How decisions are made (technical merit)

All discussions and changes are evaluated on technical merit: correctness, tests,
clarity, performance, API design, maintainability, and documentation. Decisions
are not based on who proposes a change. Please keep threads focused on the work
at hand and support claims with evidence (tests, benchmarks, minimal repros, or
references).

## Quick start

1. Fork the repo and create a feature branch:
   - `git checkout -b my-change`

2. Make your changes (and add tests if applicable).

3. Run the full local check suite:
   - `cargo fmt --check`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test --all-targets`
   - (optional) `cargo test --release`
   - (optional) `cargo bench` (if your change affects performance)

4. Open a Pull Request.

## What we look for

### Correctness and tests
- Bug fixes should include a regression test when reasonable.
- New behavior should come with tests covering edge cases (empty trees, small trees, off-by-one ranks, etc.).

### Style
- Formatting is enforced via `rustfmt`.
- Lints are enforced via `clippy` with warnings treated as errors.

### Performance
`wabi_tree` is a data structure crate, so perf matters:
- Avoid unnecessary allocations and cloning.
- Prefer iterator-friendly APIs when possible.
- If you change algorithms or invariants, include benchmarks or reasoning.

### Public API
- Changes to the public API should be clearly motivated and documented.
- Consider adding a small example to the README if you add a user-facing feature.

## Commit and PR guidance

- Keep PRs focused (one logical change).
- Include a clear description:
  - What changed and why
  - How it was tested
  - Any trade-offs

## Release-related changes

If your PR is intended to ship in the next release, please also:
- Note whether it is breaking/change/add/fix/perf/docs.
- (If you maintain a changelog) add an entry.

## Security issues

Please do not open public issues for security vulnerabilities. Follow the instructions in `SECURITY.md`.
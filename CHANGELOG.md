# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.10.0] — 2026-04-11

### Added
- **Lab 10: PySpark — First Contact** — introduction to distributed computing
  with Apache Spark in local mode, covering lazy evaluation, DAG execution,
  shuffles, partitions, and the Spark UI.
- `src/lab10.py` — student skeleton with 8 TODOs (word count, analytics
  pipeline, partition analysis).
- `src/lab10_solutions.py` — reference solutions.
- `tests/test_lab10.py` — 37 tests across 6 test classes.
- `docs/labs/lab10_instructions.md` — step-by-step instructions with reflection
  questions.
- `docs/labs/lab10_guide.md` — theory reference covering Spark architecture,
  RDD vs DataFrame, Catalyst optimizer, shuffles, partitioning, and Spark UI.
- Java requirement documented in pre-flight checklist and guide.
- Common gotcha sections in guide: `(line,)` tuple syntax, `[(idx, count)]`
  list wrapper for `mapPartitionsWithIndex`.
- PySpark added to docs index, README, and technology stack.

### Fixed
- Test count for "the" in `small_corpus` corrected from 5 to 6 in both
  `TestWordcountRdd` and `TestWordcountDataframe`.
- Partition distribution narrative updated to reflect actual hash-collision
  behavior (empty partitions) rather than idealized uniform distribution.

## [0.9.1] — 2025

### Fixed
- Lab 09 deployment docs and documentation updates.

## [0.9.0] — 2025

### Added
- Lab 09: Final Project Setup — Git workflows, remote setup, project templating.

## [0.8.0] — 2025

### Added
- Lab 08: Kernel Approximation Methods — exact RBF kernels, RFF, ORF, Nystrom.

## [0.7.0] — 2025

### Added
- Lab 07: Probabilistic Data Structures & Polars — Bloom filters, HyperLogLog,
  Count-Min Sketch, Polars introduction.

## [0.6.0] — 2025

### Added
- Lab 06: Streaming Algorithms — reservoir sampling, streaming statistics.

## [0.5.1] — 2025

### Fixed
- Refined Lab 05 content and exercises.

## [0.5.0] — 2025

### Added
- Lab 05: Out-of-Core, Streaming & Parallel Processing.

## [0.4.1] — 2025

### Fixed
- Documentation issues.

## [0.4.0] — 2025

### Added
- Lab 04: Vectorization and Broadcasting.

## [0.1.0] — 2025

### Added
- Initial release with Lab 01 (Setup and I/O Benchmarking), Lab 02 (Complexity),
  and Lab 03 (Data Types and Efficient Formats).

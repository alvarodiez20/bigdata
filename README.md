# Big Data Course Labs

[![CI Status](https://github.com/alvarodiez20/bigdata/actions/workflows/ci.yml/badge.svg)](https://github.com/alvarodiez20/bigdata/actions/workflows/ci.yml)
[![Docs Status](https://github.com/alvarodiez20/bigdata/actions/workflows/docs.yml/badge.svg)](https://github.com/alvarodiez20/bigdata/actions/workflows/docs.yml)
[![Deployment](https://img.shields.io/github/deployments/alvarodiez20/bigdata/github-pages?label=docs)](https://alvarodiez20.github.io/bigdata/)
[![Latest Release](https://img.shields.io/github/v/release/alvarodiez20/bigdata)](https://github.com/alvarodiez20/bigdata/releases)
[![License](https://img.shields.io/github/license/alvarodiez20/bigdata)](https://github.com/alvarodiez20/bigdata/blob/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)

<div align="center">
  <img src="docs/bigdata_logo.png" alt="Big Data Logo" width="300">

  <p>
    <strong>Fundamental concepts and practical skills for working with large-scale data processing</strong>
  </p>

  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="https://alvarodiez20.github.io/bigdata/">Documentation</a> •
    <a href="#labs">Labs</a>
  </p>
</div>

---

## About

This repository contains laboratory exercises and materials for the UNIE Big Data course. Students learn fundamental concepts and practical skills for working with large-scale data processing.

## Course Topics

- Environment setup and package management
- File I/O performance (CSV vs Parquet)
- Data processing with pandas and numpy
- Performance benchmarking and optimization
- Computational complexity (Big O) and profiling
- Data type optimization and storage formats
- Out-of-core, streaming, and parallel processing
- Probabilistic data structures (Bloom filters, HyperLogLog, Count-Min Sketch)
- Kernel approximation methods (RFF, ORF, Nyström)

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alvarodiez20/bigdata.git
   cd bigdata
   ```

2. **Install uv (package manager)**
   ```bash
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Start working on labs**
   ```bash
   uv run jupyter lab notebooks/lab01_setup_io.ipynb
   ```

## Documentation

Full course documentation is available at: **[alvarodiez20.github.io/bigdata](https://alvarodiez20.github.io/bigdata/)**

## Repository Structure

```
bigdata/
├── docs/              # Course documentation (MkDocs)
│   └── labs/          # Lab instructions and guides
├── notebooks/         # Jupyter notebooks for labs
├── src/               # Source code and lab scripts
├── tests/             # Test files
└── pyproject.toml     # Project dependencies
```

## Labs

| Lab | Topic |
|-----|-------|
| Lab 01 | Environment Setup & I/O Benchmarking |
| Lab 02 | Complexity Analysis and Profiling |
| Lab 03 | Data Types and Efficient Formats |
| Lab 04 | Vectorization and Broadcasting |
| Lab 05 | Out-of-Core, Streaming & Parallel Processing |
| Lab 06 | Streaming Algorithms |
| Lab 07 | Probabilistic Data Structures & Polars |
| Lab 08 | Kernel Approximation Methods |

## Technology Stack

- **Python 3.11+** — programming language
- **uv** — fast Python package manager
- **pandas / numpy** — data manipulation and numerical computing
- **scipy** — scientific computing (linear algebra, statistics)
- **polars** — high-performance DataFrame library
- **pyarrow / Parquet** — columnar storage format
- **matplotlib** — visualization
- **Jupyter** — interactive notebooks
- **MkDocs Material** — documentation site

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Álvaro Díez** — [@alvarodiez20](https://github.com/alvarodiez20)

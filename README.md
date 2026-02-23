# Big Data Course Labs 🐜

[![CI Status](https://github.com/alvarodiez20/bigdata/actions/workflows/ci.yml/badge.svg)](https://github.com/alvarodiez20/bigdata/actions/workflows/ci.yml)
[![Docs Status](https://github.com/alvarodiez20/bigdata/actions/workflows/docs.yml/badge.svg)](https://github.com/alvarodiez20/bigdata/actions/workflows/docs.yml)
[![Deployment](https://img.shields.io/github/deployments/alvarodiez20/bigdata/github-pages?label=docs)](https://alvarodiez20.github.io/bigdata/)
[![Latest Release](https://img.shields.io/github/v/release/alvarodiez20/bigdata)](https://github.com/alvarodiez20/bigdata/releases)
[![License](https://img.shields.io/github/license/alvarodiez20/bigdata)](https://github.com/alvarodiez20/bigdata/blob/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)

<div align="center">
  <img src="docs/bigdata_logo.png" alt="Big Data Logo" width="300">
  
  <h3>Big Data Course Labs</h3>
  
  <p>
    <strong>Fundamental concepts and practical skills for working with large-scale data processing</strong>
  </p>
  
  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="https://alvarodiez20.github.io/bigdata/">Documentation</a> •
    <a href="#-course-topics">Topics</a> •
    <a href="#-labs">Labs</a>
  </p>
</div>

---

## 📚 About

This repository contains laboratory exercises and materials for the UNIE Big Data course. Students will learn fundamental concepts and practical skills for working with large-scale data processing.

## 🎯 Course Topics

- Environment setup and package management
- File I/O performance (CSV vs Parquet)
- Data processing with pandas and numpy
- Performance benchmarking and optimization
- Computational complexity (Big O) and data flow
- Big data formats and storage

## 🚀 Quick Start

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
   # Open the first lab notebook
   uv run jupyter lab notebooks/lab01_setup_io.ipynb
   ```

## 📖 Documentation

Full course documentation is available at: **[alvarodiez20.github.io/bigdata](https://alvarodiez20.github.io/bigdata/)**

## 📁 Repository Structure

```
bigdata/
├── docs/              # Course documentation (MkDocs)
│   └── labs/          # Lab instructions and guides
├── notebooks/         # Jupyter notebooks for labs
├── src/               # Source code and utilities
├── tests/             # Test files
└── pyproject.toml     # Project dependencies
```

## 🧪 Labs

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 01 | Environment Setup & I/O Benchmarking | ✅ Available |
| Lab 02 | Complexity and Data Flow | ✅ Available |
| Lab 03 | Data Types and Efficient Formats | ✅ Available |
| Lab 04 | Efficient Formats and Vectorization | ✅ Available |
| Lab 05 | Out-of-Core, Streaming & Parallel Processing | ✅ Available |

## 🛠️ Technology Stack

- **Python 3.11+** - Programming language
- **uv** - Fast Python package manager
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **Jupyter** - Interactive notebooks
- **MkDocs Material** - Documentation site

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Álvaro Díez**
- GitHub: [@alvarodiez20](https://github.com/alvarodiez20)

---

<p align="center">
  Made with ❤️ for Big Data education
</p>

<p align="center">
  <img src="bigdata_logo.png" alt="Big Data Logo" width="300"/>
</p>

# Big Data Course Lab

Welcome to the **Big Data** course lab! This repository contains hands-on labs that progressively build your skills on both Python and Big Data.

## 📖 About This Course

In the era of massive datasets, traditional data processing tools are no longer sufficient. This course teaches you how to work with large-scale data efficiently by understanding:

- **Storage optimization**: Why file formats matter and how columnar storage (Parquet) outperforms row-based formats (CSV)
- **Performance measurement**: How to benchmark I/O operations and identify bottlenecks
- **Resource management**: Understanding memory constraints and handling out-of-memory scenarios
- **Modern tooling**: Using professional Python tools like `uv` for fast dependency management
- **Best practices**: Writing clean, testable code with proper logging and type hints

## 🎯 Learning Outcomes

By the end of this course, you will be able to:

1. **Analyze and optimize** data storage formats for different use cases
2. **Measure and compare** performance of different data processing approaches
3. **Design systems** that handle datasets larger than available RAM
4. **Use modern tools** for reproducible data science workflows
5. **Apply engineering principles** to data-intensive applications

## 📚 Course Structure

This repository contains hands-on labs that progressively build your skills:

### Lab 01: Environment Setup and I/O Benchmarking

In this first session ([see lab guide](labs/lab01_setup_io.md)) we will focus on:

1. **Modern development environment**: Using `uv` to manage Python and dependencies ultra-fast
2. **I/O Benchmark**: Compare CSV vs Parquet to experience the performance difference firsthand
3. **Memory management**: Understand what happens when your data doesn't fit in RAM
4. **Philosophy**: *"What isn't measured, can't be improved"* - We'll measure read times, disk space, and memory usage

### Lab 02: Complexity and the Data Flow

Understanding that $N=1,000$ is not the same as $N=1,000,000$ ([see lab guide](labs/lab02_guide.md)):

1. **The Scale Factor**: Why "fast enough" code fails at scale
2. **Memory Hierarchy**: Proving via benchmarks that RAM is faster than Disk
3. **Big O Notation**: Practical application in code profiling and optimization
4. **Data Flow**: Chunking, streaming, and full loading strategies

### Coming Soon

- **Lab 03**: Working with real-world datasets
- **Lab 04**: Advanced optimization techniques

## 🚀 Getting Started

New to the course? Start with [Lab 01](labs/lab01_setup_io.md) which includes:

- Complete setup instructions from scratch (Python, VS Code, Git, uv)
- Step-by-step guidance for beginners
- Hands-on exercises with TODO functions
- Common troubleshooting tips

## 💡 Philosophy

> "The best way to learn Big Data is to work with it hands-on. Theory is important, but nothing beats the experience of seeing a 10x speedup or running out of memory and learning how to fix it."

This course emphasizes **practical experience** over pure theory. You'll write code, run benchmarks, and see real performance differences.

## 🛠️ Tools We Use

- **Python 3.12+**: Modern Python with type hints
- **uv**: Ultra-fast Python package manager
- **pandas**: Data manipulation and analysis
- **pyarrow/Parquet**: Efficient columnar storage
- **Jupyter**: Interactive notebooks for exploration
- **VS Code**: Professional code editor
- **Git**: Version control

## 📊 What Makes This Course Different

Unlike traditional database courses, we focus on:

- **File formats** rather than databases
- **Benchmarking** rather than just implementation
- **Resource constraints** (memory, disk) as first-class concerns
- **Modern Python tooling** used in industry
- **Hands-on learning** with real performance measurements

## Navigation

Use the menu on the left to access detailed guides for each lab.

---

**Questions?** Ask your instructor or TA. Let's dive into Big Data! 🎉


<p align="center">
  <img src="bigdata_logo.png" alt="Big Data Logo" width="300"/>
</p>

# Big Data Course Labs

Welcome to the **Big Data** course lab repository. These hands-on labs progressively build your skills in large-scale data processing, from file I/O fundamentals through to kernel approximation methods used in machine learning.

## About This Course

In the era of massive datasets, traditional data processing tools are no longer sufficient. This course teaches you how to work with large-scale data efficiently by understanding:

- **Storage optimization**: Why file formats matter and how columnar storage (Parquet) outperforms row-based formats (CSV)
- **Performance measurement**: How to benchmark I/O operations and identify bottlenecks
- **Resource management**: Understanding memory constraints and handling out-of-memory scenarios
- **Algorithmic thinking**: How to choose data structures and algorithms that scale
- **Modern tooling**: Using professional Python tools like `uv` for fast dependency management

## Learning Outcomes

By the end of this course, you will be able to:

1. Analyze and optimize data storage formats for different use cases
2. Measure and compare the performance of different data processing approaches
3. Design systems that handle datasets larger than available RAM
4. Implement probabilistic data structures with provable error guarantees
5. Apply kernel approximation methods to make kernel ML feasible at scale

## Course Structure

### Lab 01: Environment Setup and I/O Benchmarking

In this first session ([see lab guide](labs/lab01_setup_io.md)) we focus on:

1. **Modern development environment**: Using `uv` to manage Python and dependencies
2. **I/O Benchmark**: Compare CSV vs Parquet to experience the performance difference firsthand
3. **Memory management**: Understand what happens when your data doesn't fit in RAM
4. **Philosophy**: *"What isn't measured, can't be improved"* — we'll measure read times, disk space, and memory usage

### Lab 02: Complexity and the Data Flow

Understanding that $N=1{,}000$ is not the same as $N=1{,}000{,}000$ ([see lab guide](labs/lab02_guide.md)):

1. **The Scale Factor**: Why "fast enough" code fails at scale
2. **Memory Hierarchy**: Proving via benchmarks that RAM is faster than disk
3. **Big O Notation**: Practical application in code profiling and optimization
4. **Data Flow**: Chunking, streaming, and full loading strategies

### Lab 03: Data Types and Efficient Formats

Understanding that data types matter for performance ([see lab guide](labs/lab03_guide.md)):

1. **Data Type Optimization**: Reduce memory 5–10x with proper dtype selection
2. **Format Comparison**: CSV vs Parquet vs Feather trade-offs
3. **Parquet Deep Dive**: Row groups, compression, predicate pushdown
4. **Partitioning Strategies**: Organize data for fast analytical queries

### Lab 04: Vectorization and Broadcasting

Combining efficient storage with fast computation ([see lab guide](labs/lab04_guide.md)):

1. **Format Comparison**: CSV vs Parquet (Snappy, Zstd) vs Feather — size, speed, and features
2. **Column Pruning & Predicate Pushdown**: Read only what you need from Parquet
3. **Vectorization**: Replace Python loops with NumPy/Pandas (100–200x speedup)
4. **Broadcasting**: Apply operations across arrays without explicit loops
5. **Pipeline Optimization**: Combine format choice and vectorization for maximum performance

### Lab 05: Out-of-Core, Streaming & Parallel Processing

Processing datasets larger than RAM ([see lab guide](labs/lab05_guide.md)):

1. **PyArrow Direct**: When to use PyArrow vs Pandas for I/O and projection pushdown
2. **Out-of-Core Processing**: Handle datasets that don't fit in RAM using chunking
3. **Online Statistics**: Compute mean/std in a single pass with Welford's algorithm
4. **Parallelization**: Threading for I/O-bound, multiprocessing for CPU-bound tasks
5. **Pipeline Design**: Combine chunking and parallelization for scalable processing

### Lab 06: Streaming Algorithms

Computing statistics over data streams without storing the full dataset ([see lab guide](labs/lab06_guide.md)):

1. **Reservoir Sampling**: Uniform random samples from a stream of unknown length
2. **Count-Min Sketch**: Approximate frequency counting with bounded error
3. **HyperLogLog**: Cardinality estimation using $O(\log \log n)$ memory
4. **Heavy Hitters**: Find frequent items in a stream with limited memory
5. **Error Analysis**: Understanding space–accuracy trade-offs in streaming algorithms

### Lab 07: Probabilistic Data Structures & Polars

Exact answers are expensive — approximate answers at scale ([see lab guide](labs/lab07_guide.md)):

1. **Bloom Filters**: Membership testing with zero false negatives and bounded false positives
2. **Count-Min Sketch (extended)**: Frequency estimation and join size approximation
3. **HyperLogLog (extended)**: Cardinality estimation under real-world skew
4. **Polars Introduction**: High-performance DataFrames with a lazy evaluation engine
5. **Benchmarking**: Compare probabilistic vs exact approaches across dataset sizes

### Lab 08: Kernel Approximation Methods

Making kernel machine learning feasible for large datasets ([see lab guide](labs/lab08_guide.md)):

1. **Exact RBF Kernel**: Understanding the $O(n^2)$ scalability problem
2. **Random Fourier Features (RFF)**: Randomized approximation via Bochner's theorem
3. **Orthogonal Random Features (ORF)**: Reduced-variance alternative to RFF
4. **Nyström Approximation**: Landmark-based low-rank kernel factorization
5. **Kernel Ridge Regression**: End-to-end regression with approximate kernels
6. **Benchmarking**: Time, memory, and approximation error across all methods

## Getting Started

New to the course? Start with [Lab 01](labs/lab01_setup_io.md), which includes complete setup instructions for Python, VS Code, Git, and `uv`.

## Tools We Use

- **Python 3.11+**: Modern Python with type hints
- **uv**: Ultra-fast Python package manager
- **pandas / numpy**: Data manipulation and numerical computing
- **scipy**: Scientific computing — linear algebra, statistics
- **polars**: High-performance DataFrame library with lazy evaluation
- **pyarrow / Parquet**: Efficient columnar storage
- **matplotlib**: Visualization
- **Jupyter**: Interactive notebooks for exploration
- **VS Code**: Professional code editor

## Navigation

Use the menu on the left to access instructions and reference guides for each lab.

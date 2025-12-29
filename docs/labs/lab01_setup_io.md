# Lab 01: Environment Setup and Basic I/O Benchmarking

Welcome to your first Big Data laboratory session! This lab will help you verify your development environment and introduce you to simple performance measurement.

## 📚 Additional Resources

- **[Tips & Reference Guide](lab01_guide.md)** - Complete guide with detailed tips, code examples, and quick reference

## 🎯 What You Will Learn

- How to setup and verify your Python environment is working correctly with `uv` and VS Code
- Basic file I/O operations: reading and writing CSV and Parquet formats
- Simple time measurement using Python's built-in `time` module

## ✅ Pre-flight Checklist

Before starting the lab, make sure your environment is ready. Open your terminal (Git Bash on Windows, Terminal on macOS/Linux) and run these commands:

### Check Python version
```bash
python --version
```
Expected output: `Python 3.11.x` or `Python 3.12.x`

### Check `uv` is installed
```bash
uv --version
```
Expected output: `uv 0.x.x` (any recent version)

### Check required libraries
```bash
uv run python -c "import pandas, pyarrow; print('All imports OK ✓')"
```
Expected output: `All imports OK ✓`

If any of these commands fail, refer to the **Environment Setup** section below.

---

## 🛠️ Environment Setup from Scratch

If you're starting fresh, follow these steps to install all required tools.

### 1. Install Python

**Windows:**

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer and **check "Add Python to PATH"**
3. Complete the installation

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.12
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

Verify installation:
```bash
python --version
```

### 2. Install VS Code

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install the **Python extension** by Microsoft (search "Python" in the Extensions panel)
3. Install the **Jupyter extension** by Microsoft (search "Jupyter")

### 3. Install Git

**Windows:**

- Download and install from [git-scm.com](https://git-scm.com/downloads)
- This also installs **Git Bash**, which we'll use for terminal commands

**macOS:**
```bash
# Using Homebrew
brew install git
```

**Linux:**
```bash
sudo apt install git
```

Verify:
```bash
git --version
```

### 4. Install `uv` Package Manager

`uv` is a modern, fast Python package manager we'll use throughout this course. Install it using `pip`:

```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

Expected output: `uv 0.x.x` (any recent version)

> **Note:** `uv` is much faster than traditional `pip` for installing packages and managing virtual environments, which is why we use it in this course.

### 5. Clone the Repository

Navigate to where you want to store the course files:

```bash
# Example: Go to your Documents folder
cd ~/Documents

# Clone the repository (replace YOUR_USERNAME with your GitHub username)
git clone https://github.com/alvarodiez20/bigdata.git

# Enter the project directory
cd bigdata
```

### 6. Install Dependencies

Install all required Python packages using `uv`:

```bash
uv sync
```

This creates a virtual environment (`.venv`) and installs `pandas`, `pyarrow`, `jupyter`, and other dependencies listed in `pyproject.toml`.

---

## 💻 Working with VS Code

### Step 1: Open the Project Folder

1. Open VS Code
2. **File → Open Folder...**
3. Select the `bigdata` folder you just cloned
4. Trust the folder when prompted

### Step 2: Select the Python Interpreter

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
2. Type **"Python: Select Interpreter"**
3. Choose the interpreter from `.venv` (it will show something like `./venv/bin/python`)

### Step 3: Open the Notebook

1. In the Explorer panel (left sidebar), navigate to `notebooks/`
2. Click on `lab01_setup_io.ipynb`
3. The notebook will open in VS Code

### Step 4: Select the Kernel

1. In the top-right corner of the notebook, click **"Select Kernel"**
2. Choose **"Python Environments..."**
3. Select the `.venv` interpreter (same as Step 2)

### Step 5: Run Your First Cell

1. Click on the first cell in the notebook
2. Press `Shift+Enter` to run it
3. You should see the output below the cell

✅ If you see output without errors, you're ready to go!

---

## 📝 Lab Steps

Follow along in the notebook `notebooks/lab01_setup_io.ipynb`. Here's what you'll do:

### A. Create Required Folders

Make sure these folders exist:

- `data/raw/` — for raw CSV files
- `data/processed/` — for processed Parquet files
- `results/` — for benchmark results

The notebook will create them automatically using your `ensure_dir()` function.

### B. Generate a Tiny Synthetic Dataset

You'll write a function `write_synthetic_csv()` that:

- Creates a simple DataFrame with 200,000 rows
- Columns: `timestamp`, `user_id`, `value`, `category`
- Saves it to `data/raw/synthetic.csv`
- Returns metadata (number of rows, columns, file size)

### C. Time Reading the CSV (3 Repeats)

You'll implement `time_it()` to measure how long it takes to read the CSV file.

- Use `time.perf_counter()` for precise timing
- Repeat the read operation 3 times
- Calculate the median time

### D. Convert CSV to Parquet

You'll write `write_parquet()` to:

- Read the CSV file
- Save it as `data/processed/synthetic.parquet`
- Return metadata (file size, rows, columns)

### E. Time Reading the Parquet (3 Repeats)

Same as step C, but for the Parquet file.

### F. Save Results to JSON

You'll implement `save_json()` to save your benchmark results to:
```
results/lab01_metrics.json
```

The JSON will include:

- CSV read times
- Parquet read times
- File sizes
- Speedup ratio (CSV time / Parquet time)
- Your reflection (3 lines of text)

### G. (Optional) Out-of-Memory Test

**⚠️ Warning:** This is an optional advanced section that will intentionally try to crash your Python kernel!

You'll create a function that allocates increasingly large arrays until your system runs out of memory. This teaches you:

- What happens when data doesn't fit in RAM
- How to recognize OOM (Out of Memory) errors
- Why understanding memory limits is crucial in Big Data

**What you'll see:**

- On some systems: A clean `MemoryError` exception
- On others: The kernel will crash silently (OS kills the process to protect the system)

This is **completely safe** - it only affects the notebook kernel, not your computer. Just restart the kernel after the test.

---

## 🐛 Common Errors and Fixes

### 1. `ModuleNotFoundError: No module named 'pandas'`

**Problem:** Dependencies not installed.

**Fix:**
```bash
# Make sure you're in the project folder
cd ~/Documents/bigdata

# Sync dependencies
uv sync

# Always run Python commands with 'uv run'
uv run python -c "import pandas; print('OK')"
```

### 2. Kernel not found in VS Code

**Problem:** VS Code can't find the `.venv` interpreter.

**Fix:**

1. Close and reopen VS Code
2. Press `Ctrl+Shift+P` → **"Python: Select Interpreter"**
3. If `.venv` doesn't appear, manually enter the path: `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows)

### 3. `FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/...'`

**Problem:** Folders don't exist yet.

**Fix:** The notebook should create them automatically with `ensure_dir()`. Make sure you run all cells in order from top to bottom.

### 4. Notebook cells take forever to run

**Problem:** The dataset might be too large or your system is slow.

**Fix:** In the notebook, reduce `n_rows` from 200,000 to 50,000 or 100,000.

### 5. `uv: command not found`

**Problem:** `uv` is not installed or not in your PATH.

**Fix:**

```bash
# Install uv using pip
pip install uv

# Verify it works
uv --version
```

If you still have issues, close and reopen your terminal, then try again.

---

## 📦 What to Submit

Submit **exactly these two files** via your course platform (e.g., Moodle, email, etc.):

1. **`notebooks/lab01_setup_io.ipynb`** — Your completed notebook with all cells executed
2. **`results/lab01_metrics.json`** — The JSON file generated by the notebook

**Do NOT submit:**

- The CSV or Parquet files
- The entire repository
- Screenshots (unless explicitly requested)

---

## 🎓 Reflections

At the end of the notebook, you'll be asked to write a short reflection (3 lines) answering:

- What surprised you about the performance difference?
- Why do you think Parquet is faster/smaller?

This reflection will be saved in your `lab01_metrics.json` file.

---

## 🚀 Next Steps

After completing this lab:

1. Make sure both deliverable files are ready
2. Check your `results/lab01_metrics.json` contains all expected fields
3. Review your reflection — does it make sense?
4. Submit the files!

---

**Questions?** Ask your instructor. Happy coding! 🎉

# Lab 09: Tips & Reference Guide

Complete templates, YAML snippets, and quick-reference material for setting up your final project repository.

See the **[Instructions](lab09_instructions.md)** for the step-by-step walkthrough.

---

## Git Quick Reference

### Essential commands for this lab

```bash
# Initialise a new repo on an existing directory
git init -b main

# Connect to the remote on GitHub
git remote add origin https://github.com/USER/REPO.git

# Stage specific files and commit
git add <file1> <file2>
git commit -m "Descriptive message in imperative mood"

# Stage all tracked modifications and commit
git add -u
git commit -m "..."

# Push and set the upstream tracking branch
git push -u origin main

# Push subsequent commits (upstream already set)
git push

# View history (one line per commit)
git log --oneline

# See what is staged vs unstaged
git status
git diff --staged
```

### Writing good commit messages

Follow the **imperative mood** convention used by Git itself and most open-source projects:

| ✅ Good | ❌ Bad |
|---|---|
| `Add pyproject.toml and src/weather package` | `added files` |
| `Fix ruff F401 unused import in utils.py` | `fix stuff` |
| `Deploy MkDocs to GitHub Pages via Actions` | `github actions` |

A commit message answers: *"If applied, this commit will ______."*

---

## `.gitignore` Template

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
.eggs/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
env/
.uv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data files — never commit raw data
data/*
!data/.gitkeep
*.csv
*.parquet
*.nc
*.grib
*.grib2
*.h5
*.hdf5

# Secrets & environment variables
.env
.env.*
secrets.json
credentials.json

# MkDocs build output
site/

# Coverage
.coverage
coverage.xml
htmlcov/

# IDEs
.vscode/
.idea/
*.swp

# macOS
.DS_Store

# Ruff cache
.ruff_cache/
```

!!! warning "The `data/` rule is critical"
    ERA5 or AEMET datasets can be tens of gigabytes. A single accidental `git add data/` can create a commit so large it makes the repository unusable. The rule `data/*` + `!data/.gitkeep` blocks all content while keeping the directory tracked.

---

## `pyproject.toml` Template

```toml
[project]
name = "proyecto-meteorologia"
version = "0.0.1"
description = "Análisis de datos meteorológicos — Big Data"
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Your Name" }]

dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "requests>=2.31",
    # add more as your analysis requires:
    # "polars>=1.0",
    # "dask>=2024.1",
    # "meteostat>=1.6",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.8",
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
# E/W: pycodestyle  F: pyflakes  I: isort  UP: pyupgrade
select = ["E", "F", "W", "I", "UP"]
ignore = ["E501"]  # line-too-long — handled by the formatter

[tool.ruff.format]
quote-style = "double"
```

### Version discipline

`pyproject.toml` is the **only** place where the version number lives. Every time you bump `version = "X.Y.Z"` and push to `main`, the `release.yml` workflow will automatically create a git tag and a GitHub Release. Never hard-code the version anywhere else.

---

## `mkdocs.yml` Template

```yaml
site_name: Análisis de Datos Meteorológicos
site_description: Proyecto final Big Data — Análisis de datos meteorológicos
site_author: Your Name
repo_url: https://github.com/YOUR_USERNAME/proyecto-meteorologia
repo_name: proyecto-meteorologia

theme:
  name: material
  language: es
  palette:
    - scheme: default
      primary: blue
      accent: cyan
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: blue
      accent: cyan
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - content.code.copy
    - navigation.indexes
    - navigation.top

nav:
  - Home: index.md
  - Analysis: analysis.md

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - tables
  - toc:
      permalink: true

plugins:
  - search
```

### Useful admonition types

```markdown
!!! note
    A blue informational box.

!!! tip
    A green tip box.

!!! warning
    An orange warning box.

!!! danger
    A red danger box.
```

### Adding pages

Extend the `nav` section as your documentation grows:

```yaml
nav:
  - Home: index.md
  - Data Sources: data_sources.md
  - Methodology: methodology.md
  - Results: results.md
  - Conclusions: conclusions.md
```

---

## GitHub Actions Workflows

### `ci.yml`

Runs on every push and pull request to `main`. Lints, formats-check, and tests with coverage.

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    name: Lint and Test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Check version bump
        run: |
          # Fetch all tags
          git fetch --tags
          VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
          if git rev-parse "v$VERSION" >/dev/null 2>&1; then
            echo "Error: Version v$VERSION already exists as a tag. Please bump the version in pyproject.toml."
            exit 1
          fi

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: "pyproject.toml"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --group dev

      - name: Ruff lint
        run: uv run ruff check .

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Run tests with coverage
        run: uv run pytest --cov=src --cov-report=xml --cov-report=term-missing

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
        continue-on-error: true
```

### `docs.yml`

Builds the docs on every push/PR. Deploys to GitHub Pages only on pushes to `main`.

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: pages-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    name: Build Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: "pyproject.toml"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --group dev

      - name: Build docs
        run: uv run mkdocs build --strict

  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: "pyproject.toml"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --group dev

      - name: Deploy to GitHub Pages
        run: uv run mkdocs gh-deploy --force
```

### `release.yml`

Creates a versioned GitHub Release whenever the version in `pyproject.toml` changes on `main`.

```yaml
name: Release

on:
  push:
    branches: [main]

jobs:
  tag-and-release:
    name: Tag version and create release
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Extract version from pyproject.toml
        id: version
        run: |
          VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          echo "tag=v$VERSION"    >> $GITHUB_OUTPUT

      - name: Check if tag already exists
        id: tag_check
        run: |
          if git rev-parse "v${{ steps.version.outputs.version }}" >/dev/null 2>&1; then
            echo "exists=true"  >> $GITHUB_OUTPUT
          else
            echo "exists=false" >> $GITHUB_OUTPUT
          fi

      - name: Create GitHub Release
        if: steps.tag_check.outputs.exists == 'false'
        uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ steps.version.outputs.tag }}
          name: Release ${{ steps.version.outputs.tag }}
          generate_release_notes: true
```

### Debugging a failing workflow

1. Go to **Actions** tab on GitHub → click the red ❌ run.
2. Expand the failing step to read the full error log.
3. Common issues:

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: weather` | `pythonpath` missing in `pytest` config | Add `pythonpath = ["src"]` to `[tool.pytest.ini_options]` |
| `ruff: error: ... invalid value` | Old ruff syntax | Check ruff changelog; `select`/`ignore` moved under `[tool.ruff.lint]` |
| `mkdocs: WARNING` treated as error | `--strict` flag | Fix the warning (usually a broken link or missing page) |
| Docs deploy fails with permissions error | Pages not configured | Enable GitHub Pages → Source: GitHub Actions |

---

## README Template

```markdown
# Análisis de Datos Meteorológicos

> Proyecto final — Big Data · Grado en Matemáticas · UNIE Universidad

[![CI](https://github.com/USER/proyecto-meteorologia/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/proyecto-meteorologia/actions/workflows/ci.yml)
[![Docs](https://github.com/USER/proyecto-meteorologia/actions/workflows/docs.yml/badge.svg)](https://USER.github.io/proyecto-meteorologia/)
[![Coverage](https://codecov.io/gh/USER/proyecto-meteorologia/graph/badge.svg)](https://codecov.io/gh/USER/proyecto-meteorologia)
[![Version](https://img.shields.io/github/v/release/USER/proyecto-meteorologia)](https://github.com/USER/proyecto-meteorologia/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Description

*(Replace with your line-of-work description.)*

## Documentation

Full documentation at **[USER.github.io/proyecto-meteorologia](https://USER.github.io/proyecto-meteorologia/)**

## Installation

  ```bash
  git clone https://github.com/USER/proyecto-meteorologia.git
  cd proyecto-meteorologia
  pip install uv
  uv sync --group dev
  ```

## Data Download

Data is not included in the repository. To download:

  ```bash
  # TODO: add your data download instructions
  ```

## Usage

  ```bash
  uv run pytest                          # run tests
  uv run pytest --cov=src -v     # tests with coverage
  uv run ruff check .                    # lint
  uv run ruff format .                   # format
  uv run mkdocs serve                    # preview docs at localhost:8000
  ```

## Project Structure

  ```
  proyecto-meteorologia/
  ├── .github/workflows/   # CI/CD pipelines
  ├── data/                # Data files (not committed — see .gitignore)
  ├── docs/                # MkDocs documentation sources
  ├── notebooks/           # Exploratory notebooks
  ├── src/weather/         # Source package
  ├── tests/               # Unit and integration tests
  ├── mkdocs.yml
  ├── pyproject.toml
  └── README.md
  ```

## Author

**Your Name** · [github.com/USER](https://github.com/USER)

## Professor
**Álvaro Diez** · [github.com/alvarodiez20](https://github.com/alvarodiez20)

---

*Big Data · 4º Grado en Matemáticas · UNIE Universidad · 2025–2026*
```

### Badge anatomy

Badges are just images linked to a URL:

```markdown
[![ALT TEXT](IMAGE_URL)](LINK_URL)
```

- **CI badge**: the image URL comes from GitHub Actions directly — it updates in real time.
- **Shields.io badges**: static badges with customisable label/value/colour. Pattern:
  `https://img.shields.io/badge/LABEL-VALUE-COLOR`

---

## Common Pitfalls

| Problem | Fix |
|---|---|
| Coverage badge says "unknown" or doesn't load | Log in to [codecov.io](https://codecov.io) with GitHub, authorize the repository, and wait for the next CI run. |
| Docs link gives a 404 Not Found error | Go to your repo **Settings** → **Pages** and ensure Source is "GitHub Actions". It takes 2–3 minutes after the first deploy. |
| `git push` rejected with "repository not found" | Check the remote URL: `git remote -v`. Re-add with correct username/repo name. |
| `uv sync` fails with "no field `version`" | Ensure `pyproject.toml` has `[project]` section with `version = "..."`. |
| Tests fail with `ImportError: No module named 'weather'` | Set `pythonpath = ["src"]` in `[tool.pytest.ini_options]`. |
| `ruff format --check` fails in CI but passes locally | Run `ruff format .` locally, commit the changes, and push again. |
| MkDocs `--strict` fails with "Doc file ... contains a link ..." | Fix broken internal links in `.md` files. |
| GitHub Pages shows a 404 | Wait 2–3 minutes after the first deploy; also check Settings → Pages is set to *GitHub Actions*. |
| Release not created | Verify the `release.yml` workflow ran. Check that the version tag does not already exist locally: `git tag`. |

---

## Project Structure Explained

```
proyecto-meteorologia/
│
├── .github/
│   └── workflows/
│       ├── ci.yml        ← lint + test on every push/PR
│       ├── docs.yml      ← build + deploy docs on main push
│       └── release.yml   ← auto-release when version changes
│
├── data/
│   └── .gitkeep          ← placeholder so the folder is tracked
│                            (actual data files are gitignored)
│
├── docs/
│   ├── index.md          ← home page of your docs site
│   └── analysis.md       ← your analysis documentation
│
├── notebooks/            ← exploratory Jupyter notebooks (optional)
│
├── src/
│   └── weather/
│       ├── __init__.py   ← makes `weather` importable; sets __version__
│       └── utils.py      ← your utility functions
│
├── tests/
│   ├── __init__.py
│   └── test_utils.py     ← pytest tests
│
├── .gitignore
├── mkdocs.yml
├── pyproject.toml        ← single source of truth for version + config
└── README.md
```

The **`src/` layout** ensures that when you run `pytest`, Python imports from the *installed* package (via `pythonpath = ["src"]`) rather than from the local directory. This catches import errors that a flat layout would silently mask.

---

## Useful Links

- [uv documentation](https://docs.astral.sh/uv/)
- [ruff documentation](https://docs.astral.sh/ruff/)
- [pytest documentation](https://docs.pytest.org/)
- [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Shields.io badge builder](https://shields.io/)
- [Codecov](https://codecov.io/) — free coverage hosting for public repos

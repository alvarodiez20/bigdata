# Lab 09: Final Project Setup — Instructions

In this lab you will build the skeleton of your final project repository from scratch, following professional software engineering practices. You will create the repository on GitHub and evolve it **commit by commit on `main`** — no branches — so you can watch a real project take shape: package layout, tests, linting, documentation, and CI/CD pipelines.

By the end of the session your repository will already have most of the scaffolding required for the final grade. Your remaining work over the coming weeks is to fill it with meteorological data and analysis.

## Additional Resources

- **[Tips & Reference Guide](lab09_guide.md)** — `pyproject.toml` templates, GitHub Actions syntax, MkDocs configuration, git cheatsheet, and badge reference.

## Pre-flight Checklist

1. **Git installed**: `git --version` (should print `2.x` or higher).
2. **Python 3.10+**: `python --version`.
3. **uv installed**: `pip install uv` (or follow [docs.astral.sh/uv](https://docs.astral.sh/uv)).
4. **GitHub account**: if you don't have one, create it at [github.com](https://github.com) — choose a professional username.
5. **GitHub CLI (optional but recommended)**: `gh --version`. Install from [cli.github.com](https://cli.github.com).

---

## Step 0 — Create the GitHub Repository

1. Go to [github.com](https://github.com) → **+** → **New repository**.
2. Name it **`proyecto-meteorologia`** (or something similar).
3. Set visibility to **Public** (required for free GitHub Pages hosting).
4. **Do not** initialise with README, `.gitignore`, or licence — we create everything from scratch.
5. Click **Create repository**.

Then initialise your local copy:

```bash
mkdir -p ~/repos/proyecto-meteorologia
cd ~/repos/proyecto-meteorologia
git init -b main
git remote add origin https://github.com/YOUR_USERNAME/proyecto-meteorologia.git
```

Configure your identity if you have not done so globally:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

---

## Commit 1 — Skeleton and `.gitignore`

!!! objective
    Create the full directory structure and a `.gitignore` that prevents data files, virtual environments, and generated artefacts from ever reaching version control.

### 1.1 — Create the directory tree

```bash
mkdir -p .github/workflows data docs notebooks src/weather tests
touch data/.gitkeep   # keeps the data/ folder tracked even though its contents are ignored
```

### 1.2 — Create `.gitignore`

Create a file `.gitignore` at the root of the project with the following content (see the full template in the [guide](lab09_guide.md#gitignore-template)):

- Python bytecode: `__pycache__/`, `*.pyc`, `*.egg-info/`
- Virtual environments: `.venv/`, `.uv/`
- Jupyter checkpoints: `.ipynb_checkpoints/`
- **Data files** — the most important section: `data/*` with `!data/.gitkeep`
- Secrets: `.env`, `credentials.json`
- Build output: `site/`, `dist/`, `build/`
- Coverage: `.coverage`, `coverage.xml`, `htmlcov/`

### 1.3 — Add a README stub

Create `README.md` with just a title and your name — you will complete it in Commit 7.

### 1.4 — Commit

```bash
git add .gitignore README.md data/.gitkeep
git commit -m "Initial commit: project skeleton and .gitignore"
```

---

## Commit 2 — `pyproject.toml` and Python Package

!!! objective
    Set up `pyproject.toml` as the single source of truth for project metadata, dependencies, and tool configuration. Create the `src/weather/` package with a couple of utility functions.

### 2.1 — `pyproject.toml`

Create `pyproject.toml` at the root. The key sections are:

- `[project]` — name, **version** (`0.0.1`), description, `requires-python = ">=3.10"`, runtime `dependencies`. We will follow [SemVer](https://semver.org/).
- `[dependency-groups] dev` — `pytest`, `pytest-cov`, `ruff`, `mkdocs`, `mkdocs-material`.
- `[tool.pytest.ini_options]` — `testpaths = ["tests"]` and `pythonpath = ["src"]` (so tests can `import weather`).
- `[tool.ruff]` — `line-length = 88`, `target-version = "py310"`.
- `[tool.ruff.lint]` — `select = ["E", "F", "W", "I", "UP"]`.
- `[build-system]` — `setuptools`.

See the complete template in the [guide](lab09_guide.md#pyprojecttoml-template).

### 2.2 — `src/weather/__init__.py`

```python
"""Meteorological data analysis package."""

```

### 2.3 — `src/weather/utils.py`

Add at least one utility function relevant to your chosen line of work. As a starting point, include:

- `celsius_to_fahrenheit(temp_c)` — trivial conversion, useful to verify the test pipeline works.

```python
def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit.

    Args:
        temp_c: The temperature in degrees Celsius.

    Returns:
        The temperature in degrees Fahrenheit.
    """
    return (temp_c * 9 / 5) + 32
```

Each function must have a **docstring** with `Args:` and `Returns:` sections, and **type hints**. Following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### 2.4 — Install the dependencies

```bash
uv sync --group dev
```

### 2.5 — Commit

```bash
git add pyproject.toml src/
git commit -m "Add pyproject.toml and src/weather package skeleton"
```

---

## Commit 3 — Tests with `pytest` and Coverage

!!! objective
    Write a first test suite with enough coverage to validate that the CI pipeline will pass. Aim for **≥70%** coverage from the start — it is much easier than retrofitting tests later.

### 3.1 — `tests/__init__.py`

Empty file — marks the directory as a Python package so `pytest` can discover tests.

### 3.2 — `tests/test_utils.py`

Write tests for every function in `src/weather/utils.py`. Organise them in classes (`TestCelsiusToFahrenheit`, etc.). For each function test:

- The normal (happy path) case.
- At least one edge case or boundary value.
- Any exception it is supposed to raise (`pytest.raises`).

Example structure:

```python
import pytest
import pandas as pd
from weather.utils import celsius_to_fahrenheit

class TestCelsiusToFahrenheit:
    def test_freezing_point(self):
        assert celsius_to_fahrenheit(0) == 32.0

    def test_boiling_point(self):
        assert celsius_to_fahrenheit(100) == 212.0

    def test_negative_forty_is_equal_in_both_scales(self):
        assert celsius_to_fahrenheit(-40) == -40.0
```

### 3.3 — Run tests and check coverage

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing -v
```

You should see a coverage table. Fix any gaps before committing.

### 3.4 — Commit

```bash
git add tests/
git commit -m "Add pytest test suite with coverage for weather.utils"
```

---

## Commit 4 — Code Quality with `ruff`

!!! objective
    Lint and auto-format every Python file in the project. From this point forward, all new code must pass `ruff check` and `ruff format --check` before being committed.

### 4.1 — Lint

```bash
uv run ruff check src/ tests/
```

Fix any reported issues. Most are auto-fixable:

```bash
uv run ruff check --fix src/ tests/
```

### 4.2 — Format

```bash
uv run ruff format src/ tests/
```

### 4.3 — Verify CI will pass

```bash
uv run ruff check src/ tests/       # must exit with code 0
uv run ruff format --check src/ tests/  # must exit with code 0
```

### 4.4 — Commit

```bash
git add src/ tests/
git commit -m "Apply ruff linting and formatting"
```

If ruff made no changes (your code was already clean), use:

```bash
git commit --allow-empty -m "Verify ruff passes (no changes needed)"
```

### 4.5 — VS Code Integration (Optional)

To run Ruff automatically every time you save a Python file in Visual Studio Code:

1. Install the **Ruff** extension from the VS Code Marketplace (publisher: `charliermarsh`).
2. Create a folder named `.vscode` in the root of your project.
3. Inside that folder, create a file named `settings.json` and add the following configuration:

```json
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    }
}
```

---

## Commit 5 — MkDocs Documentation with Material Theme

!!! objective
    Create a documentation site with MkDocs and the Material theme. Verify it builds without errors locally before it goes into CI.

### 5.1 — `mkdocs.yml`

Create `mkdocs.yml` at the root. Key settings:

- `site_name`, `site_author`, `repo_url` — fill in your details.
- `theme.name: material`, `theme.language: es`.
- Light/dark palette toggle.
- Features: `content.code.copy`, `navigation.indexes`, `navigation.top`.
- Minimal nav: `Home: index.md` + `Analysis: analysis.md`.

See the complete template in the [guide](lab09_guide.md#mkdocsyml-template).

### 5.2 — `docs/index.md`

Include: project title, brief description of your chosen line of work, data source, installation instructions, and your name.

### 5.3 — `docs/analysis.md`

A placeholder page with section headings for: Data Loading & Cleaning, Exploratory Analysis, Results, Conclusions. You will fill these in as you develop your project.

### 5.4 — Build locally

```bash
uv run mkdocs build --strict
```

`--strict` turns warnings into errors — the same flag used in CI. Fix any warnings before committing.

To preview in the browser:

```bash
uv run mkdocs serve
# open http://127.0.0.1:8000
```

### 5.5 — Commit

```bash
git add mkdocs.yml docs/
git commit -m "Add MkDocs documentation site with Material theme"
```

---

## Commit 6 — GitHub Actions CI/CD

!!! objective
    Add three automated workflows: CI (lint + test on every push), documentation deployment to GitHub Pages, and automatic versioned releases.

### Before creating the files — enable GitHub Pages

1. Go to your repository on GitHub → **Settings → Pages**.
2. Under *Build and deployment* → **Source: GitHub Actions**.
3. Save. The first deployment will trigger after your next push.

### 6.1 — `.github/workflows/ci.yml`

Runs on every push and PR to `main`. Steps:

1. Checkout (with `fetch-depth: 0`)
2. Check version bump
3. Set up Python (via project file) + `uv`
4. `uv sync --group dev`
5. `ruff check .`
6. `ruff format --check .`
7. `pytest --cov=src --cov-report=xml --cov-report=term-missing`
8. Upload `coverage.xml` to Codecov (optional — `continue-on-error: true`)

See the complete YAML in the [guide](lab09_guide.md#ciyml).

### 6.2 — `.github/workflows/docs.yml`

Runs on push to `main` (and PRs for the build step only). Two jobs:

- **build**: `mkdocs build --strict`
- **deploy** (main only, `permissions: contents: write`): `mkdocs gh-deploy --force`

See the complete YAML in the [guide](lab09_guide.md#docsyml).

### 6.3 — `.github/workflows/release.yml`

Runs on push to `main`. Reads the version from `pyproject.toml` and creates a GitHub Release + tag if the tag does not already exist.

See the complete YAML in the [guide](lab09_guide.md#releaseyml).

### 6.4 — Commit and push

```bash
git add .github/
git commit -m "Add GitHub Actions: CI, docs deploy, and auto-release"
git push -u origin main
```

Watch the **Actions** tab on GitHub. All three workflows should turn green within a few minutes. If any workflow fails, read the error log, fix the issue, and push again.

---

## Commit 7 — Complete README with Badges

!!! objective
    Replace the README stub with a complete project README that includes live status badges, installation instructions, and a project structure overview.

### 7.1 — Badge URLs

Add these badges below the title (replace `YOUR_USERNAME` with your GitHub username):

| Badge | URL pattern |
|---|---|
| CI status | `https://github.com/YOUR_USERNAME/proyecto-meteorologia/actions/workflows/ci.yml/badge.svg` |
| Docs status | `https://github.com/YOUR_USERNAME/proyecto-meteorologia/actions/workflows/docs.yml/badge.svg` |
| Coverage | `https://codecov.io/gh/YOUR_USERNAME/proyecto-meteorologia/graph/badge.svg` |
| Version | `https://img.shields.io/github/v/release/YOUR_USERNAME/proyecto-meteorologia` |
| Python version | `https://img.shields.io/badge/python-3.10%2B-blue` |
| Ruff | `https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json` |

Embed badges as Markdown image links:

```markdown
[![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)](URL_TO_WORKFLOW)
```

### 7.2 — README sections

Include at minimum:

1. **Title and badges**
2. **Description** — your line of work, data source, main objective
3. **Documentation link** — link to your GitHub Pages site
4. **Installation** — `git clone`, `uv sync --group dev`
5. **Data download** — how to get the data (script or instructions)
6. **Usage** — how to run tests, serve docs, lint
7. **Project structure** — directory tree
8. **Author**

See the full README template in the [guide](lab09_guide.md#readme-template).

### 7.3 — Commit and push

```bash
git add README.md
git commit -m "Complete README with CI/docs badges"
git push
```

---

## Final Verification

After all seven commits are pushed, confirm the following:

```bash
git log --oneline
```

You should see exactly seven commits on `main`, with descriptive messages.

Then check on GitHub:

- [ ] **Actions tab** — all three workflows show ✅ on the latest commit.
- [ ] **GitHub Pages** — `https://YOUR_USERNAME.github.io/proyecto-meteorologia/` loads your docs site.
- [ ] **Releases** — a release `v0.0.1` has been created automatically.
- [ ] **Repository root** — the README renders with live green badges.

**Note:** Probably the coverage badge will display "unknown". I let you guys figure out how to fix it. Probably you will need to login into codecov.io and follow the instructions.

---

## What to Submit

Submit the **URL of your GitHub repository** through the course platform:
`https://github.com/YOUR_USERNAME/proyecto-meteorologia`

Before submitting verify:

- [ ] The repository is **public**.
- [ ] The CI workflow shows ✅ on the last commit.
- [ ] The coverage badge shows the correct percentage.
- [ ] The docs site is deployed and accessible.
- [ ] GitHub Release `v0.0.1` exists.
- [ ] All seven commits are on `main` with descriptive messages.

!!! warning "This repository is your final project"
    Keep committing to it regularly as you develop your analysis. The professor will evaluate the GitHub repository as it stands at the time of the oral presentation on **28 April 2026**.

---

**Questions?** Check the [Tips & Reference Guide](lab09_guide.md) or ask your instructor.

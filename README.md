# Big Data - Práctica 1: Setup & I/O Benchmark

Setup del entorno y primeros experimentos de rendimiento para la asignatura de Big Data (UNIE).

## 🚀 Inicio Rápido

Este proyecto usa **uv** para una instalación rápida y reproducible.

### 1. Instalación
```bash
# Instala dependencies y crea entorno virtual
uv sync
```

### 2. Ejecutar Benchmark
Genera datos sintéticos (si no existen) y compara CSV vs Parquet:
```bash
uv run python -m src.io_benchmark
```
Resultados en `results/p1_metrics.json`.

### 3. Ver Documentación
La guía completa de la práctica está en la documentación.
```bash
# Levantar servidor local
uv run mkdocs serve
```
Abre http://127.0.0.1:8000

## 📚 Estructura
- `src/`: Scripts Python (benchmark, utils).
- `notebooks/`: Jupyter Notebooks para la clase.
- `docs/`: Fuentes de la documentación MkDocs.
- `results/`: Salida de los experimentos (JSON/Markdown).

## ☁️ GitHub Pages
Este repositorio publica automáticamente la documentación en:
https://alvarodiez20.github.io/bigdata/
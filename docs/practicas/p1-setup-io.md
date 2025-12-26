# Práctica 1: Setup y Benchmark I/O

Esta práctica tiene como objetivo configurar tu entorno de desarrollo profesional y realizar experimentos de rendimiento ("benchmark") comparando formatos de datos.

## Objetivos de Aprendizaje
- **Ingeniería de Datos**: Comprender la diferencia entre serialización por filas (CSV) y columnar (Parquet).
- **Herramientas Modernas**: Uso de `uv` para gestión de dependencias.
- **Calidad de Código**: Uso de logs, tipado estático y testing básico.
- **Gestión de Recursos**: Monitorización de memoria RAM y entendimiento de errores OOM (Out of Memory).

---

## 1. Instalación de Herramientas (uv)

Usaremos `uv`, un gestor de paquetes Python moderno y extremadamente rápido.

### Pasos
1.  **Instalar uv** (si no lo tienes):
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clonar y Sincronizar**:
    ```bash
    git clone https://github.com/alvarodiez20/bigdata.git
    cd bigdata
    uv sync
    ```

---

## 2. Teoría: Columnar vs Filas

Antes de medir, entendamos qué esperar.

### CSV (Row-Oriented)
- Texto plano, legible por humanos.
- Para leer la columna 100, debes leer (y parsear) las 99 anteriores de cada fila.
- **Lento** para analítica, **Gigante** en disco.

### Parquet (Column-Oriented)
- Binario, comprimido (Snappy, GZIP).
- Almacena datos columna por columna.
- **Column Pruning**: Si solo pides 3 columnas, solo lee esos bloques de bytes.
- **Predicate Pushdown**: Puede filtrar datos antes de cargarlos a memoria.

---

## 3. Ejecución del Benchmark

### A. Vía Terminal (CLI)
Nuestro script `src/io_benchmark.py` ahora usa **Logging** profesional. Puedes controlar el nivel de detalle:

```bash
# Ejecución estándar (INFO)
uv run python -m src.io_benchmark

# Ejecución detallada (DEBUG) - Verás cada run individual
uv run python -m src.io_benchmark --log-level DEBUG
```

### B. Vía Notebook (Jupyter)
Explora `notebooks/p1_setup_io_benchmark.ipynb`. Incluye:
1.  **Generación de datos**: 1M de filas.
2.  **Comparativa**: CSV vs Parquet (Tiempo y Espacio).
3.  **Monitorización de RAM**: Una demo interactiva que satura la memoria para que veas qué ocurre cuando los datos no caben en RAM ("Out of Memory").

---

## 4. Tests Unitarios

Hemos incluido tests para asegurar la calidad del código. Ejecútalos con:
```bash
uv run python -m unittest discover tests
```

---

## Troubleshooting

### ¿Dónde están los logs?
Por defecto salen por consola (stderr). Si quieres guardarlos a archivo, puedes redirigir la salida:
```bash
uv run python -m src.io_benchmark 2> benchmark.log
```

### Error: "Values not in list" (OOM Demo)
Si el notebook crashea, es normal en la sección 4. Hemos forzado un `MemoryError`. Si tu kernel muere silenciosamente, es que el sistema operativo mató el proceso (OOM Killer) para proteger el sistema.

### mkdocs serve no funciona
Asegúrate de estar en la carpeta raíz `bigdata/` y haber hecho `uv sync`.

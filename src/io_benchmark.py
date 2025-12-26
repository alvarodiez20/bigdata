import argparse
import time
import statistics
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from src.utils import measure_rss_mb, save_metrics, save_summary_md, generate_synthetic_data

# Initialize logger
logger = logging.getLogger(__name__)

def run_benchmark(input_path: Path, output_json: Path, output_md: Path, n_runs: int = 3) -> None:
    """Runs the IO benchmark comparing CSV and Parquet.

    Args:
        input_path: Path to the input CSV file.
        output_json: Path to save the JSON metrics.
        output_md: Path to save the Markdown summary.
        n_runs: Number of repetitions for each experiment.
    """
    
    # 0. Setup
    results: Dict[str, Any] = {
        "dataset": {},
        "experiments": []
    }
    
    if not input_path.exists():
        generate_synthetic_data(input_path)
    
    file_stats = input_path.stat()
    
    # Read once to get shape without timing
    logger.info(f"Inspecting input file: {input_path}")
    df_preview = pd.read_csv(input_path)
    results["dataset"] = {
        "path": str(input_path),
        "rows": len(df_preview),
        "columns": len(df_preview.columns),
        "size_bytes": file_stats.st_size
    }
    del df_preview
    
    # --- Experiment A: Read CSV Full ---
    logger.info(f"Experiment A: Reading CSV Full ({n_runs} runs)...")
    runs: List[float] = []
    rss_before = measure_rss_mb()
    
    for i in range(n_runs):
        t0 = time.perf_counter()
        _ = pd.read_csv(input_path)
        t1 = time.perf_counter()
        duration = t1 - t0
        runs.append(duration)
        logger.debug(f"  Run {i+1}: {duration:.4f}s")
        
    rss_after = measure_rss_mb()
    results["experiments"].append({
        "name": "read_csv_full",
        "runs_sec": runs,
        "median_sec": statistics.median(runs),
        "rss_before": rss_before,
        "rss_after": rss_after
    })
    
    # --- Experiment B: Convert to Parquet & Read ---
    logger.info("Experiment B: CSV -> Parquet conversion...")
    parquet_path = input_path.with_suffix('.parquet')
    
    # Measure Write Time
    df = pd.read_csv(input_path)
    t0 = time.perf_counter()
    df.to_parquet(parquet_path)
    write_time = time.perf_counter() - t0
    del df
    
    logger.info(f"  Conversion took {write_time:.4f}s")
    
    # Read Parquet Full
    logger.info(f"  Reading Parquet Full ({n_runs} runs)...")
    runs_pq: List[float] = []
    rss_before_pq = measure_rss_mb()
    
    for i in range(n_runs):
        t0 = time.perf_counter()
        _ = pd.read_parquet(parquet_path)
        t1 = time.perf_counter()
        duration = t1 - t0
        runs_pq.append(duration)
        logger.debug(f"  Run {i+1}: {duration:.4f}s")
        
    rss_after_pq = measure_rss_mb()
    results["experiments"].append({
        "name": "read_parquet_full",
        "runs_sec": runs_pq,
        "median_sec": statistics.median(runs_pq),
        "rss_before": rss_before_pq,
        "rss_after": rss_after_pq,
        "file_size_bytes": parquet_path.stat().st_size
    })
    
    # --- Experiment C: Column Pruning (Read 3 cols) ---
    logger.info("Experiment C: Column Pruning (3 cols)...")
    cols_to_read = ['timestamp', 'user_id', 'value']
    
    # CSV Pruned
    runs_csv_pruned: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = pd.read_csv(input_path, usecols=cols_to_read)
        t1 = time.perf_counter()
        runs_csv_pruned.append(t1 - t0)
        
    results["experiments"].append({
        "name": "read_csv_pruned_3cols",
        "median_sec": statistics.median(runs_csv_pruned),
        "runs_sec": runs_csv_pruned,
        "rss_before": -1, # Ignored
        "rss_after": -1
    })
    
    # Parquet Pruned
    runs_pq_pruned: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = pd.read_parquet(parquet_path, columns=cols_to_read)
        t1 = time.perf_counter()
        runs_pq_pruned.append(t1 - t0)
        
    results["experiments"].append({
        "name": "read_parquet_pruned_3cols",
        "median_sec": statistics.median(runs_pq_pruned),
        "runs_sec": runs_pq_pruned,
        "rss_before": -1,
        "rss_after": -1
    })

    # Save results
    save_metrics(results, output_json)
    save_summary_md(results, output_md)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Big Data P1: IO Benchmark")
    parser.add_argument("--input", type=Path, default=Path("data/raw/synthetic.csv"), help="Path to input CSV")
    parser.add_argument("--out", type=Path, default=Path("results/p1_metrics.json"), help="Path to output JSON")
    parser.add_argument("--summary", type=Path, default=Path("results/p1_summary.md"), help="Path to output Summary MD")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of rows for synthetic data")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level")
    
    args = parser.parse_args()
    
    # Configure logging based on CLI arg
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if not args.input.exists():
        generate_synthetic_data(args.input, n_rows=args.rows)
        
    run_benchmark(args.input, args.out, args.summary)

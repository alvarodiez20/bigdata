import logging
import json
import os
import psutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def measure_rss_mb() -> float:
    """Returns the current Resident Set Size (RSS) in MB.
    
    Returns:
        float: memory usage in Megabytes.
    """
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024)
    logger.debug(f"Current RSS: {rss:.2f} MB")
    return rss

def generate_synthetic_data(
    output_path: Path, 
    n_rows: int = 1_000_000
) -> None:
    """Generates a synthetic CSV dataset if it doesn't exist.
    
    Args:
        output_path: Path where the CSV will be saved.
        n_rows: Number of rows to generate. Defaults to 1_000_000.
    """
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return

    logger.info(f"Generating synthetic dataset with {n_rows} rows at {output_path}...")
    
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rng = np.random.default_rng(42)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='s'),
            'user_id': rng.integers(1, 100000, size=n_rows),
            'item_id': rng.integers(1, 1000, size=n_rows),
            'value': rng.normal(50, 20, size=n_rows),
            'category': rng.choice(['A', 'B', 'C', 'D'], size=n_rows),
            'region': rng.choice(['North', 'South', 'East', 'West'], size=n_rows)
        })
        
        df.to_csv(output_path, index=False)
        logger.info("Generation complete.")
        
    except Exception as e:
        logger.error(f"Failed to generate data: {e}")
        raise

def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Saves metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary containing benchmark results.
        output_path: Destination path for the JSON file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving metrics to {output_path}: {e}")

def save_summary_md(metrics: Dict[str, Any], output_path: Path) -> None:
    """Saves a human-readable summary of the metrics to a Markdown file.

    Args:
        metrics: Dictionary containing benchmark results.
        output_path: Destination path for the Markdown file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_content = "# Benchmark Summary\n\n"
        md_content += f"**Dataset:** `{metrics['dataset']['path']}`\n"
        md_content += f"- Rows: {metrics['dataset']['rows']:,}\n"
        md_content += f"- Size: {metrics['dataset']['size_bytes'] / (1024*1024):.2f} MB\n\n"
        
        md_content += "## Experiments\n\n"
        md_content += "| Experiment | Median Time (s) | RSS Delta (MB) |\n"
        md_content += "|------------|-----------------|----------------|\n"
        
        for exp in metrics['experiments']:
            # Handle cases where RSS might not be tracked (e.g. pruned reads in older code)
            rss_delta = exp.get('rss_after', 0) - exp.get('rss_before', 0)
            median_sec = exp.get('median_sec', 0.0)
            md_content += f"| {exp['name']} | {median_sec:.4f} | {rss_delta:.2f} |\n"
            
        with open(output_path, 'w') as f:
            f.write(md_content)
        logger.info(f"Summary saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving summary to {output_path}: {e}")

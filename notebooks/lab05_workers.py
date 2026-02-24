"""
Lab 05: Worker functions for multiprocessing.

These functions are in a separate module so they can be pickled and sent
to child processes by ProcessPoolExecutor. On macOS, Python uses 'spawn'
to create new processes, which requires that mapped functions be importable
from a regular module (not from a Jupyter notebook's __main__).
"""

import numpy as np
import pandas as pd


def heavy_process(filepath):
    """Process a partition: read, transform, aggregate."""
    df = pd.read_parquet(filepath)
    df['score'] = np.sqrt(df['price']) * np.log1p(df['quantity'])
    return df.groupby('category')['score'].agg(['mean', 'sum', 'count'])


def process_partition(filepath):
    """
    Process a partition: read, transform, aggregate.

    Args:
        filepath: Path to the Parquet partition file

    Returns:
        Aggregated DataFrame with revenue statistics by category and price_bin.
    """
    df = pd.read_parquet(filepath)

    df['revenue'] = df['price'] * df['quantity']
    df['price_bin'] = pd.cut(
        df['price'],
        bins=[0, 50, 200, 500, 1000],
        labels=['low', 'mid', 'high', 'premium']
    )

    result = df.groupby(['category', 'price_bin'], observed=True).agg({
        'revenue': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    })
    return result

"""
Lab 07: Polars & PySuricata

STUDENT NAME: [Your Name Here]

STUDENT REFLECTION:
(TODO 15: After generating the PySuricata report, answer these questions here:
 - Which streaming algorithms from Lab 06 can you identify in the PySuricata report?
 - How does PySuricata handle datasets larger than memory?
 - What advantage does using a LazyFrame give PySuricata compared to an eager DataFrame?
)
"""

import os
from urllib.request import urlretrieve

import polars as pl
from pysuricata import profile


# ---------------------------------------------------------------------------
# Dataset: NYC Yellow Taxi Trip Records (January 2024)
# ~3 million rows, ~45 MB parquet file
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# ---------------------------------------------------------------------------
TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
TAXI_FILE = "yellow_tripdata_2024-01.parquet"


def download_dataset() -> str:
    """
    Downloads the NYC Taxi dataset if not already present.

    Returns:
        str: Path to the local parquet file.
    """
    if not os.path.exists(TAXI_FILE):
        print(f"Downloading {TAXI_FILE} (~45 MB)...")
        urlretrieve(TAXI_URL, TAXI_FILE)
        print("Download complete.")
    else:
        print(f"Dataset already exists: {TAXI_FILE}")
    return TAXI_FILE


# ---------------------------------------------------------------------------
# Exercise 4: Introduction to Polars
# ---------------------------------------------------------------------------

def load_taxi_eager(path: str) -> pl.DataFrame:
    """
    Loads the NYC Taxi dataset eagerly into a Polars DataFrame.

    Uses pl.read_parquet() which reads and parses the entire file immediately,
    loading all ~3 million rows into memory at once — similar to
    pandas.read_parquet().

    Args:
        path (str): Path to the local parquet file.

    Returns:
        pl.DataFrame: The full taxi dataset.

    Examples:
        >>> df = load_taxi_eager("yellow_tripdata_2024-01.parquet")
        >>> df.shape[0] > 1_000_000
        True
    """
    raise NotImplementedError(
        "TODO 10: Use pl.read_parquet(path) to load the dataset"
    )


def load_taxi_lazy(path: str) -> pl.LazyFrame:
    """
    Creates a lazy query plan for the NYC Taxi dataset.

    Uses pl.scan_parquet() which does NOT load data yet — it only creates a
    query plan. Data is only read when you call .collect().

    This is like Spark's lazy evaluation: you describe WHAT you want,
    and Polars optimizes HOW to do it.

    Args:
        path (str): Path to the local parquet file.

    Returns:
        pl.LazyFrame: A lazy query plan for the taxi dataset.
    """
    raise NotImplementedError(
        "TODO 11: Use pl.scan_parquet(path) to create a LazyFrame"
    )


def filter_and_group(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters trips longer than 2 miles and computes mean fare by pickup zone.

    Steps:
    1. Filter rows where trip_distance > 2.0 using df.filter(pl.col("trip_distance") > 2.0)
    2. Group by "PULocationID" using .group_by("PULocationID")
    3. Aggregate the mean of "fare_amount" using .agg(pl.col("fare_amount").mean())

    Args:
        df (pl.DataFrame): The taxi dataset.

    Returns:
        pl.DataFrame: A DataFrame with columns ["PULocationID", "fare_amount"].

    Examples:
        >>> df = load_taxi_eager("yellow_tripdata_2024-01.parquet")
        >>> result = filter_and_group(df)
        >>> "PULocationID" in result.columns
        True
    """
    raise NotImplementedError(
        "TODO 12: Filter trip_distance > 2.0, group by PULocationID, aggregate mean fare_amount"
    )


def add_computed_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a "tip_percentage" column computed as (tip_amount / total_amount) * 100.

    Uses df.with_columns() and pl.col() expressions to create a new
    column without modifying the original. Uses .alias() to name the result.

    Handles division by zero: when total_amount is 0, tip_percentage should be 0.

    Args:
        df (pl.DataFrame): The taxi dataset.

    Returns:
        pl.DataFrame: The dataset with an additional "tip_percentage" column.

    Examples:
        >>> df = load_taxi_eager("yellow_tripdata_2024-01.parquet")
        >>> result = add_computed_column(df)
        >>> "tip_percentage" in result.columns
        True
    """
    raise NotImplementedError(
        "TODO 13: Use with_columns to add tip_percentage = (tip_amount / total_amount) * 100"
    )


# ---------------------------------------------------------------------------
# Exercise 5: PySuricata with Polars
# ---------------------------------------------------------------------------

def generate_report(lf: pl.LazyFrame) -> None:
    """
    Generates a PySuricata profiling report from a LazyFrame.

    Steps:
    1. Call profile(lf) to create a streaming profile of the data.
    2. Save the report as "taxi_report.html".

    PySuricata processes data in chunks using the same streaming algorithms
    from Lab 06: Welford's for statistics, reservoir sampling, KMV sketches
    for distinct counts, and more.

    With ~3 million rows, you should notice PySuricata processing the data
    in multiple chunks — this is streaming in action!

    Args:
        lf (pl.LazyFrame): A lazy query plan for the dataset.
    """
    raise NotImplementedError(
        "TODO 14: Call profile(lf), then save_html('taxi_report.html')"
    )


# ---------------------------------------------------------------------------
# Main — run this script to test your implementations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 07 — Part B: Polars & PySuricata")
    print("=" * 60)

    # Download dataset
    path = download_dataset()

    # --- Exercise 4: Polars basics ---
    print("\n--- Exercise 4a: Eager loading ---")
    try:
        df = load_taxi_eager(path)
        print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns (eager)")
        print(df.head(3))
    except NotImplementedError as e:
        print(f"  ⏳ {e}")

    print("\n--- Exercise 4b: Lazy loading ---")
    try:
        lf = load_taxi_lazy(path)
        print("LazyFrame created (no data loaded yet)")
        print("Collecting first 5 rows...")
        print(lf.head(5).collect())
    except NotImplementedError as e:
        print(f"  ⏳ {e}")

    print("\n--- Exercise 4c: Filter and group ---")
    try:
        df = load_taxi_eager(path)
        result = filter_and_group(df)
        print(f"Mean fare by pickup zone (trips > 2 miles): {result.shape[0]} zones")
        print(result.sort("fare_amount", descending=True).head(5))
    except NotImplementedError as e:
        print(f"  ⏳ {e}")

    print("\n--- Exercise 4d: Computed column ---")
    try:
        df = load_taxi_eager(path)
        result = add_computed_column(df)
        print("First 5 rows with tip_percentage:")
        print(
            result.select(["fare_amount", "tip_amount", "total_amount", "tip_percentage"])
            .head(5)
        )
    except NotImplementedError as e:
        print(f"  ⏳ {e}")

    # --- Exercise 5: PySuricata ---
    print("\n--- Exercise 5: PySuricata report ---")
    try:
        lf = load_taxi_lazy(path)
        generate_report(lf)
        print("✅ Report saved to taxi_report.html")
        print("   Open it in your browser: open taxi_report.html")
    except NotImplementedError as e:
        print(f"  ⏳ {e}")

    print("\n" + "=" * 60)
    print("Done! See instructions for reflection questions (TODO 15).")
    print("=" * 60)

import os
import tempfile

import polars as pl
import pytest

from src.lab07_polars import (
    add_computed_column,
    filter_and_group,
    load_taxi_eager,
    load_taxi_lazy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "trip_distance": [1.0, 3.0, 5.0, 0.5, 2.5],
    "PULocationID":  [10,  10,  20,  20,  30],
    "fare_amount":   [5.0, 15.0, 25.0, 4.0, 12.0],
    "tip_amount":    [1.0, 3.0,  5.0,  0.0, 0.0],
    "total_amount":  [6.0, 18.0, 30.0, 4.0, 0.0],
}


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(SAMPLE_DATA)


@pytest.fixture()
def parquet_file(tmp_path):
    """Write the sample DataFrame to a temp parquet file and return its path."""
    path = str(tmp_path / "test_taxi.parquet")
    pl.DataFrame(SAMPLE_DATA).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Exercise 4a: load_taxi_eager
# ---------------------------------------------------------------------------

def test_load_taxi_eager_returns_dataframe(parquet_file):
    """load_taxi_eager returns a pl.DataFrame."""
    df = load_taxi_eager(parquet_file)
    assert isinstance(df, pl.DataFrame)


def test_load_taxi_eager_row_count(parquet_file):
    """load_taxi_eager loads all rows from the parquet file."""
    df = load_taxi_eager(parquet_file)
    assert df.shape[0] == len(SAMPLE_DATA["trip_distance"])


def test_load_taxi_eager_columns(parquet_file):
    """load_taxi_eager preserves all expected columns."""
    df = load_taxi_eager(parquet_file)
    for col in SAMPLE_DATA:
        assert col in df.columns, f"Expected column '{col}' missing from DataFrame"


# ---------------------------------------------------------------------------
# Exercise 4b: load_taxi_lazy
# ---------------------------------------------------------------------------

def test_load_taxi_lazy_returns_lazyframe(parquet_file):
    """load_taxi_lazy returns a pl.LazyFrame (not yet evaluated)."""
    lf = load_taxi_lazy(parquet_file)
    assert isinstance(lf, pl.LazyFrame)


def test_load_taxi_lazy_collects_correctly(parquet_file):
    """Collecting the LazyFrame produces the same data as eager loading."""
    eager_df = load_taxi_eager(parquet_file)
    lazy_df  = load_taxi_lazy(parquet_file).collect()
    assert eager_df.shape == lazy_df.shape


# ---------------------------------------------------------------------------
# Exercise 4c: filter_and_group
# ---------------------------------------------------------------------------

def test_filter_and_group_returns_dataframe(sample_df):
    """filter_and_group returns a pl.DataFrame."""
    result = filter_and_group(sample_df)
    assert isinstance(result, pl.DataFrame)


def test_filter_and_group_columns(sample_df):
    """Result has exactly PULocationID and fare_amount columns."""
    result = filter_and_group(sample_df)
    assert set(result.columns) == {"PULocationID", "fare_amount"}


def test_filter_and_group_filters_short_trips(sample_df):
    """Only zones that have trips > 2 miles appear in the result."""
    # Zones 10, 20, 30 each have at least one trip > 2 miles (3.0, 5.0, 2.5).
    # Zone 20 also has a 0.5-mile trip that should be excluded from the mean.
    result = filter_and_group(sample_df)
    location_ids = set(result["PULocationID"].to_list())
    # Zone 10: 3.0 miles → included; Zone 20: 5.0 miles → included; Zone 30: 2.5 → included
    assert 10 in location_ids
    assert 20 in location_ids
    assert 30 in location_ids


def test_filter_and_group_correct_mean(sample_df):
    """Mean fare for zone 10 is the average of the trips > 2 miles only."""
    result = filter_and_group(sample_df)
    zone10 = result.filter(pl.col("PULocationID") == 10)
    assert zone10.shape[0] == 1
    # Only the 3.0-mile trip (fare=15.0) passes the filter for zone 10
    assert zone10["fare_amount"][0] == pytest.approx(15.0)


def test_filter_and_group_excludes_short_trips(sample_df):
    """Zone 20's mean fare uses only the 5-mile trip, not the 0.5-mile one."""
    result = filter_and_group(sample_df)
    zone20 = result.filter(pl.col("PULocationID") == 20)
    assert zone20.shape[0] == 1
    # Only the 5.0-mile trip (fare=25.0) passes for zone 20
    assert zone20["fare_amount"][0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Exercise 4d: add_computed_column
# ---------------------------------------------------------------------------

def test_add_computed_column_returns_dataframe(sample_df):
    """add_computed_column returns a pl.DataFrame."""
    result = add_computed_column(sample_df)
    assert isinstance(result, pl.DataFrame)


def test_add_computed_column_has_tip_percentage(sample_df):
    """Result contains a 'tip_percentage' column."""
    result = add_computed_column(sample_df)
    assert "tip_percentage" in result.columns


def test_add_computed_column_preserves_rows(sample_df):
    """Row count is unchanged after adding the computed column."""
    result = add_computed_column(sample_df)
    assert result.shape[0] == sample_df.shape[0]


def test_add_computed_column_values(sample_df):
    """tip_percentage = (tip_amount / total_amount) * 100, with 0 when total is 0."""
    result = add_computed_column(sample_df)
    tips = result["tip_percentage"].to_list()

    # Row 0: 1.0 / 6.0 * 100 ≈ 16.67
    assert tips[0] == pytest.approx(1.0 / 6.0 * 100, rel=1e-3)
    # Row 1: 3.0 / 18.0 * 100 ≈ 16.67
    assert tips[1] == pytest.approx(3.0 / 18.0 * 100, rel=1e-3)
    # Row 2: 5.0 / 30.0 * 100 ≈ 16.67
    assert tips[2] == pytest.approx(5.0 / 30.0 * 100, rel=1e-3)
    # Row 3: 0.0 / 4.0 * 100 = 0.0
    assert tips[3] == pytest.approx(0.0)


def test_add_computed_column_zero_division(sample_df):
    """When total_amount is 0, tip_percentage should be 0 (no division-by-zero error)."""
    result = add_computed_column(sample_df)
    # Row 4 has total_amount=0.0
    tip_pct = result["tip_percentage"].to_list()[4]
    assert tip_pct == pytest.approx(0.0), (
        f"Expected 0.0 when total_amount is 0, got {tip_pct}"
    )


def test_add_computed_column_does_not_modify_original(sample_df):
    """add_computed_column must not mutate the input DataFrame."""
    original_cols = sample_df.columns[:]
    add_computed_column(sample_df)
    assert sample_df.columns == original_cols

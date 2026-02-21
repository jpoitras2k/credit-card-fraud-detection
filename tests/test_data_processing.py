# tests/test_dataframe_utils.py
# Pytest tests for the provided pandas utilities.
#
# Usage:
#   pytest -q
#
# IMPORTANT:
# - Update the import below to match your module name (file where functions live).
#   Example: from my_package.dataframe_utils import ...
#   or:      import dataframe_utils as m

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

# 🔧 TODO: change this import to your actual module path
from ml_framework_project.data_analyzer.data_preprocessing import (  # noqa: F401
    drop_missing_values,
    fill_missing_values,
    fill_missing_values_with_mean,
    fill_missing_values_with_median,
    fill_missing_values_with_mode,
    standardize_column,
    normalize_column,
    normalize_columns,
    shuffle_dataframe,
    encode_categorical_column,
    sample_dataframe,
)


@pytest.fixture
def df_missing():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, np.nan, 40.0],
            "c": ["x", "y", "z", None],
        }
    )


@pytest.fixture
def df_numeric():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})


@pytest.fixture
def df_categorical():
    return pd.DataFrame({"color": ["red", "blue", "red"], "val": [1, 2, 3]})


# -------------------------
# drop_missing_values
# -------------------------
def test_drop_missing_values_all_columns(df_missing):
    out = drop_missing_values(df_missing.copy())
    # rows with any NA across all columns removed; only first row is complete
    expected = df_missing.iloc[[0]].copy()
    assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_drop_missing_values_subset_columns(df_missing):
    out = drop_missing_values(df_missing.copy(), columns=["a", "b"])
    # keep rows where a and b are both not NA:
    # row0 ok, row1 a NA, row2 b NA, row3 ok
    expected = df_missing.iloc[[0, 3]].copy()
    assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


# -------------------------
# fill_missing_values
# -------------------------


def test_fill_missing_values_fills_only_target_column(df_missing):
    df = df_missing.copy()
    out = fill_missing_values(df, column="a", value=999)

    assert out is df  # in-place behavior
    assert out.loc[1, "a"] == 999

    # other missing values elsewhere remain
    assert pd.isna(out.loc[2, "b"])
    # pandas may store missing object values as NaN instead of None
    assert pd.isna(out.loc[3, "c"])


# -------------------------
# fill_missing_values_with_mean / median / mode
# -------------------------
def test_fill_missing_values_with_mean(df_missing):
    df = df_missing.copy()
    mean_a = df["a"].mean()  # mean of [1,3,4] = 8/3
    out = fill_missing_values_with_mean(df, "a")

    assert out is df
    assert np.isclose(out.loc[1, "a"], mean_a)


def test_fill_missing_values_with_median(df_missing):
    df = df_missing.copy()
    median_b = df["b"].median()  # median of [10,20,40] = 20
    out = fill_missing_values_with_median(df, "b")

    assert out is df
    assert np.isclose(out.loc[2, "b"], median_b)


def test_fill_missing_values_with_mode(df_missing):
    df = df_missing.copy()
    # Make a mode: repeat "x"
    df.loc[1, "c"] = "x"
    df.loc[3, "c"] = None
    mode_c = df["c"].mode()[0]
    out = fill_missing_values_with_mode(df, "c")

    assert out is df
    assert out.loc[3, "c"] == mode_c


# -------------------------
# standardize_column
# -------------------------
def test_standardize_column_mean_approx_zero_and_std_approx_one(df_numeric):
    df = df_numeric.copy()
    out = standardize_column(df, "x")

    assert out is df
    # pandas std default is ddof=1, so check with same
    assert np.isclose(out["x"].mean(), 0.0, atol=1e-12)
    assert np.isclose(out["x"].std(ddof=1), 1.0, atol=1e-12)


def test_standardize_column_preserves_other_columns(df_numeric):
    df = df_numeric.copy()
    y_before = df["y"].copy()
    out = standardize_column(df, "x")
    assert_series_equal(out["y"], y_before)


def test_standardize_column_constant_values_produces_nan():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    out = standardize_column(df, "x")
    assert out["x"].isna().all()


# -------------------------
# normalize_column
# -------------------------
def test_normalize_column_range_0_1(df_numeric):
    df = df_numeric.copy()
    out = normalize_column(df, "x")

    assert out is df
    assert np.isclose(out["x"].min(), 0.0)
    assert np.isclose(out["x"].max(), 1.0)
    # Monotonic increasing for original increasing data
    assert out["x"].is_monotonic_increasing


def test_normalize_column_constant_values_produces_nan():
    df = pd.DataFrame({"x": [2.0, 2.0]})
    out = normalize_column(df, "x")
    assert out["x"].isna().all()


# -------------------------
# normalize_columns
# -------------------------
def test_normalize_columns_all_if_none(df_numeric):
    df = df_numeric.copy()
    out = normalize_columns(df)  # should normalize both columns

    assert out is df
    for col in ["x", "y"]:
        assert np.isclose(out[col].min(), 0.0)
        assert np.isclose(out[col].max(), 1.0)


def test_normalize_columns_subset(df_numeric):
    df = df_numeric.copy()
    y_before = df["y"].copy()

    out = normalize_columns(df, columns=["x"])

    assert out is df
    assert np.isclose(out["x"].min(), 0.0)
    assert np.isclose(out["x"].max(), 1.0)
    # y unchanged
    assert_series_equal(out["y"], y_before)


# -------------------------
# shuffle_dataframe
# -------------------------
def test_shuffle_dataframe_reproducible_with_random_state():
    df = pd.DataFrame({"a": list(range(20))})
    out1 = shuffle_dataframe(df, random_state=123)
    out2 = shuffle_dataframe(df, random_state=123)
    assert_frame_equal(out1, out2)


def test_shuffle_dataframe_same_rows_different_order():
    df = pd.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
    out = shuffle_dataframe(df, random_state=42)

    assert len(out) == len(df)
    # same set of rows (order-independent)
    assert set(map(tuple, out.values.tolist())) == set(map(tuple, df.values.tolist()))
    # index reset
    assert list(out.index) == list(range(len(df)))


# -------------------------
# encode_categorical_column
# -------------------------
def test_encode_categorical_column_creates_dummies_and_drops_original(df_categorical):
    df = df_categorical.copy()
    out = encode_categorical_column(df, "color")

    assert "color" not in out.columns
    assert "color_red" in out.columns
    assert "color_blue" in out.columns
    assert "val" in out.columns

    expected = pd.DataFrame(
        {
            "val": [1, 2, 3],
            "color_blue": [0, 1, 0],
            "color_red": [1, 0, 1],
        }
    )

    # pd.get_dummies may output bool dtype; compare values without enforcing dtype
    assert_frame_equal(
        out.sort_index(axis=1),
        expected.sort_index(axis=1),
        check_dtype=False,
    )


def test_encode_categorical_column_handles_nan_category():
    df = pd.DataFrame({"color": ["red", None, "blue"]})
    out = encode_categorical_column(df, "color")
    # By default, get_dummies does not create a column for NaN unless dummy_na=True
    assert "color" not in out.columns
    assert "color_red" in out.columns
    assert "color_blue" in out.columns
    # row with None should be all zeros across dummy cols
    dummy_cols = [c for c in out.columns if c.startswith("color_")]
    assert (out.loc[1, dummy_cols] == 0).all()


# -------------------------
# sample_dataframe
# -------------------------
def test_sample_dataframe_reproducible_with_random_state():
    df = pd.DataFrame({"a": list(range(50))})
    out1 = sample_dataframe(df, n=10, random_state=999)
    out2 = sample_dataframe(df, n=10, random_state=999)
    assert_frame_equal(out1, out2)


def test_sample_dataframe_returns_n_rows_and_resets_index():
    df = pd.DataFrame({"a": list(range(10))})
    out = sample_dataframe(df, n=5, random_state=0)

    assert len(out) == 5
    assert list(out.index) == [0, 1, 2, 3, 4]


def test_sample_dataframe_raises_if_n_too_large():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        sample_dataframe(df, n=10, random_state=0)

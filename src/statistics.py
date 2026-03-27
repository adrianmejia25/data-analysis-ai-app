"""
statistics.py
Descriptive and inferential statistics over a loaded dataset.

Expects input in the standard data contract format produced by data_loader.
"""

import pandas as pd


def descriptive_stats(data: dict) -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).

    Returns
    -------
    pd.DataFrame
        Rows: count, mean, std, min, 25%, 50%, 75%, max per numeric column.
    """
    raise NotImplementedError


def correlation_matrix(data: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    columns : list[str] or None
        Subset of columns to include. None uses all numeric columns.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.
    """
    raise NotImplementedError


def null_report(data: dict) -> pd.DataFrame:
    """
    Return a report of null counts and percentages per column.

    Parameters
    ----------
    data : dict
        Data contract dict with keys 'df' and 'nulls'.

    Returns
    -------
    pd.DataFrame
        Columns: column_name, null_count, null_pct.
    """
    raise NotImplementedError


def outlier_detection(data: dict, column: str, method: str = "iqr") -> pd.Series:
    """
    Detect outliers in a single numeric column.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    column : str
        Name of the numeric column to analyse.
    method : str
        Detection method: 'iqr' (interquartile range) or 'zscore'.

    Returns
    -------
    pd.Series
        Boolean Series — True where a row is an outlier.
    """
    raise NotImplementedError

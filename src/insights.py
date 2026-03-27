"""
insights.py
High-level analytical insights derived from statistics and ML results.

Consumes the standard data contract dict from data_loader and results
from statistics / ml_models to produce human-readable summaries.
"""

import pandas as pd
from sklearn.base import BaseEstimator


def top_correlated_features(data: dict, target_column: str, top_n: int = 5) -> pd.DataFrame:
    """
    Return the top N features most correlated with a target column.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    target_column : str
        Column to measure correlation against.
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Columns: feature (str), correlation (float). Sorted descending by |correlation|.
    """
    raise NotImplementedError


def data_quality_report(data: dict) -> dict:
    """
    Produce a data quality summary for the dataset.

    Parameters
    ----------
    data : dict
        Data contract dict with keys df, dtypes, nulls, shape.

    Returns
    -------
    dict
        Keys:
        - total_rows      (int)
        - total_columns   (int)
        - columns_with_nulls (list[str])
        - null_pct_overall   (float)
        - numeric_columns    (list[str])
        - categorical_columns (list[str])
    """
    raise NotImplementedError


def model_insight_summary(metrics: dict, model_type: str) -> str:
    """
    Generate a plain-text insight summary from model evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Output of ml_models.evaluate_model — keys vary by model_type.
    model_type : str
        One of: 'linear_regression', 'random_forest', 'logistic_regression'.

    Returns
    -------
    str
        Human-readable paragraph summarising model performance.
    """
    raise NotImplementedError


def detect_data_anomalies(data: dict, numeric_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Identify columns with potential anomalies (outliers, skew, constant values).

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    numeric_columns : list[str] or None
        Columns to inspect. None inspects all numeric columns.

    Returns
    -------
    pd.DataFrame
        Columns: column (str), anomaly_type (str), detail (str).
    """
    raise NotImplementedError

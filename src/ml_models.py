"""
ml_models.py
Machine learning model training, evaluation, and prediction helpers.

Expects input in the standard data contract format produced by data_loader.
"""

import pandas as pd
from sklearn.base import BaseEstimator


def split_data(
    data: dict,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    target_column : str
        Name of the column to use as the prediction target.
    test_size : float
        Fraction of data to reserve for testing (0 < test_size < 1).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) as DataFrames / Series.
    """
    raise NotImplementedError


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "linear_regression",
) -> BaseEstimator:
    """
    Train a scikit-learn model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Target vector for training.
    model_type : str
        One of: 'linear_regression', 'random_forest', 'logistic_regression'.

    Returns
    -------
    sklearn.base.BaseEstimator
        Fitted model instance.
    """
    raise NotImplementedError


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted model on the test set.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn model.
    X_test : pd.DataFrame
        Feature matrix for testing.
    y_test : pd.Series
        True target values.

    Returns
    -------
    dict
        Keys depend on task type:
        - Regression:     mse (float), rmse (float), r2 (float)
        - Classification: accuracy (float), report (str), confusion_matrix (ndarray)
    """
    raise NotImplementedError


def predict(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    """
    Generate predictions for new data.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn model.
    X : pd.DataFrame
        Feature matrix to predict on.

    Returns
    -------
    pd.Series
        Predicted values indexed like X.
    """
    raise NotImplementedError

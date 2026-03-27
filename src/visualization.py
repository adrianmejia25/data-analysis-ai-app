"""
visualization.py
Chart generation using matplotlib / seaborn.

All functions return a matplotlib Figure so callers (including Streamlit)
can render or save the figure as needed.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_distribution(data: dict, column: str) -> plt.Figure:
    """
    Plot the distribution of a single numeric column (histogram + KDE).

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    column : str
        Name of the numeric column to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_correlation_heatmap(data: dict, columns: list[str] | None = None) -> plt.Figure:
    """
    Plot a heatmap of the Pearson correlation matrix.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    columns : list[str] or None
        Subset of columns to include. None uses all numeric columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_scatter(data: dict, x_col: str, y_col: str, hue_col: str | None = None) -> plt.Figure:
    """
    Plot a scatter chart between two numeric columns.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    x_col : str
        Column for the x-axis.
    y_col : str
        Column for the y-axis.
    hue_col : str or None
        Optional categorical column used to colour points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_bar(data: dict, x_col: str, y_col: str) -> plt.Figure:
    """
    Plot a bar chart.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    x_col : str
        Categorical column for the x-axis.
    y_col : str
        Numeric column for bar heights.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_model_results(y_test: pd.Series, y_pred: pd.Series) -> plt.Figure:
    """
    Plot actual vs. predicted values for a regression model.

    Parameters
    ----------
    y_test : pd.Series
        True target values.
    y_pred : pd.Series
        Predicted values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError

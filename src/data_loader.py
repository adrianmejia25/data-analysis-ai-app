"""
data_loader.py
Handles loading and initial inspection of data files.

Data contract: all public functions that return data return a dict with:
    - df     (pd.DataFrame): the loaded dataset
    - dtypes (dict):         column name -> dtype string
    - nulls  (dict):         column name -> null count
    - shape  (tuple):        (rows, cols)
"""

import pandas as pd


def load_csv(filepath: str) -> dict:
    """
    Load a CSV file and return the standard data contract dict.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .csv file.

    Returns
    -------
    dict
        Keys: df (DataFrame), dtypes (dict), nulls (dict), shape (tuple).
    """
    raise NotImplementedError


def load_excel(filepath: str, sheet_name: str = 0) -> dict:
    """
    Load an Excel file and return the standard data contract dict.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .xlsx file.
    sheet_name : str or int
        Sheet to read. Defaults to the first sheet (0).

    Returns
    -------
    dict
        Keys: df (DataFrame), dtypes (dict), nulls (dict), shape (tuple).
    """
    raise NotImplementedError


def summarize(data: dict) -> dict:
    """
    Generate a basic summary from a data contract dict.

    Parameters
    ----------
    data : dict
        A data contract dict as returned by load_csv or load_excel.

    Returns
    -------
    dict
        Keys: shape (tuple), dtypes (dict), nulls (dict),
              columns (list[str]).
    """
    raise NotImplementedError

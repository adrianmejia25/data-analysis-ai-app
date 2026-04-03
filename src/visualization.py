"""
visualization.py
Generación de gráficos usando matplotlib y seaborn.

Todas las funciones retornan un objeto matplotlib Figure para que el
llamador (incluyendo Streamlit) pueda renderizarlo o guardarlo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    df = data["df"]

    # Validar que la columna existe y es numérica
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La columna '{column}' no es numérica.")

    # Eliminar nulos para el gráfico
    serie = df[column].dropna()

    if serie.empty:
        raise ValueError(f"La columna '{column}' no tiene datos válidos para graficar.")

    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histograma con KDE superpuesto usando seaborn
    sns.histplot(serie, kde=True, ax=ax, color="steelblue", edgecolor="white")

    # Línea vertical en la media
    ax.axvline(serie.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Media: {serie.mean():.2f}")
    ax.axvline(serie.median(), color="orange", linestyle="--", linewidth=1.2, label=f"Mediana: {serie.median():.2f}")

    ax.set_title(f"Distribución de '{column}'", fontsize=14)
    ax.set_xlabel(column)
    ax.set_ylabel("Densidad")
    ax.set_ylim(bottom=0)  # Evitar que el eje Y empiece por encima de 0
    ax.legend()

    plt.tight_layout()
    return fig


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
    df = data["df"]

    # Seleccionar solo columnas numéricas
    df_numerico = df.select_dtypes(include=np.number)

    # Filtrar por subconjunto si se especifica
    if columns is not None:
        df_numerico = df_numerico[columns]

    # Se necesitan al menos 2 columnas para calcular correlación
    if df_numerico.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 columnas numéricas para la matriz de correlación.")

    # Calcular la matriz de correlación
    corr = df_numerico.corr().round(2)

    # Crear figura; tamaño dinámico según cantidad de columnas
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    # Mapa de calor con anotaciones de valores
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        square=True
    )

    ax.set_title("Matriz de Correlación (Pearson)", fontsize=14)

    plt.tight_layout()
    return fig


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
    df = data["df"]

    # Validar que las columnas requeridas existen
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Validar que hue_col existe si se especifica
    if hue_col is not None and hue_col not in df.columns:
        raise ValueError(f"La columna de color '{hue_col}' no existe en el DataFrame.")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter con o sin variable de color categórica
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        ax=ax,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5
    )

    titulo = f"Dispersión: '{x_col}' vs '{y_col}'"
    if hue_col:
        titulo += f"  (color: {hue_col})"
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    plt.tight_layout()
    return fig


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
    df = data["df"]

    # Validar que las columnas existen
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Validar que y_col es numérica
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError(f"La columna '{y_col}' debe ser numérica para el eje Y.")

    # Si hay muchas categorías únicas, agrupar por media
    df_plot = df[[x_col, y_col]].dropna()

    fig, ax = plt.subplots(figsize=(max(7, len(df_plot[x_col].unique()) * 0.8), 5))

    # Gráfico de barras; si x_col tiene pocos valores únicos se usa directamente,
    # si no, se agrupa por media para evitar barras superpuestas
    if df_plot[x_col].nunique() <= len(df_plot):
        sns.barplot(data=df_plot, x=x_col, y=y_col, ax=ax, color="steelblue", errorbar=None)
    else:
        agrupado = df_plot.groupby(x_col)[y_col].mean().reset_index()
        sns.barplot(data=agrupado, x=x_col, y=y_col, ax=ax, color="steelblue", errorbar=None)

    ax.set_title(f"'{y_col}' por '{x_col}'", fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # Rotar etiquetas del eje X si son largas
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


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
    # Convertir a arrays numpy para evitar problemas de alineación de índices en pandas
    y_test_arr = np.asarray(y_test).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter: valores reales vs predichos — usar arrays para asegurar que se grafican todos los puntos
    ax.scatter(y_test_arr, y_pred_arr, alpha=0.6, color="steelblue", edgecolor="white", linewidth=0.5)

    # Línea de referencia perfecta (y = x)
    minimo = min(y_test_arr.min(), y_pred_arr.min())
    maximo = max(y_test_arr.max(), y_pred_arr.max())
    ax.plot([minimo, maximo], [minimo, maximo], color="red", linestyle="--", linewidth=1.5, label="Predicción perfecta")

    ax.set_title("Valores Reales vs. Predichos", fontsize=14)
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Predicho")
    ax.legend()

    plt.tight_layout()
    return fig

"""
insights.py
High-level analytical insights derived from statistics and ML results.

Consumes the standard data contract dict from data_loader and results
from statistics / ml_models to produce human-readable summaries.
"""

import numpy as np
import pandas as pd


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
    df = data["df"]

    # Validar que la columna objetivo existe
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame.")

    # Seleccionar solo columnas numéricas
    df_numerico = df.select_dtypes(include=np.number)

    # Verificar que hay columnas numéricas suficientes para correlacionar
    if df_numerico.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "correlation"])

    # Validar que la columna objetivo es numérica
    if target_column not in df_numerico.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no es numérica.")

    # Calcular correlaciones de todas las columnas contra la columna objetivo
    correlaciones = df_numerico.corr()[target_column]

    # Excluir la correlación de la columna consigo misma
    correlaciones = correlaciones.drop(labels=[target_column], errors="ignore")

    if correlaciones.empty:
        return pd.DataFrame(columns=["feature", "correlation"])

    # Ordenar por valor absoluto descendente y tomar los top_n
    correlaciones = correlaciones.reindex(correlaciones.abs().sort_values(ascending=False).index)
    correlaciones = correlaciones.head(top_n).round(4)

    return pd.DataFrame({
        "feature": correlaciones.index,
        "correlation": correlaciones.values
    }).reset_index(drop=True)


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
        - total_rows         (int)
        - total_columns      (int)
        - columns_with_nulls (list[str])
        - null_pct_overall   (float)
        - numeric_columns    (list[str])
        - categorical_columns (list[str])
    """
    df = data["df"]

    total_filas, total_columnas = df.shape

    # Detectar columnas con al menos un valor nulo
    nulos_por_columna = df.isnull().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0].index.tolist()

    # Porcentaje global de celdas nulas sobre el total de celdas
    total_celdas = total_filas * total_columnas
    total_nulos = int(nulos_por_columna.sum())
    null_pct_global = round((total_nulos / total_celdas * 100), 2) if total_celdas > 0 else 0.0

    # Separar columnas numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = df.select_dtypes(exclude=np.number).columns.tolist()

    # Detectar filas duplicadas
    filas_duplicadas = int(df.duplicated().sum())

    # Detectar columnas con un único valor (sin variabilidad)
    columnas_constantes = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    return {
        "total_rows": total_filas,
        "total_columns": total_columnas,
        "columns_with_nulls": columnas_con_nulos,
        "null_pct_overall": null_pct_global,
        "numeric_columns": columnas_numericas,
        "categorical_columns": columnas_categoricas,
        "duplicate_rows": filas_duplicadas,
        "constant_columns": columnas_constantes,
    }


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
    nombre = model_type.replace("_", " ").title()

    # Rama de regresión: detectada por presencia de la clave 'r2' en las métricas
    if "r2" in metrics:
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")

        lineas = [f"Modelo: {nombre}"]

        if r2 > 0.85:
            calidad = "excelente"
        elif r2 > 0.70:
            calidad = "bueno"
        elif r2 > 0.50:
            calidad = "aceptable"
        else:
            calidad = "débil"
        lineas.append(f"R² = {r2:.4f} — ajuste {calidad}.")

        if rmse is not None:
            lineas.append(f"RMSE = {rmse:.4f} (error cuadrático medio).")
        if mae is not None:
            lineas.append(f"MAE  = {mae:.4f} (error absoluto medio).")
        if r2 < 0.5:
            lineas.append("El modelo explica menos del 50% de la varianza. Considerar más variables o un modelo más complejo.")

        return "\n".join(lineas)

    # Rama de clasificación: detectada por presencia de la clave 'accuracy' en las métricas
    if "accuracy" in metrics:
        accuracy = metrics.get("accuracy")
        lineas = [f"Modelo: {nombre}"]

        pct = round(accuracy * 100, 2)
        if accuracy >= 0.9:
            calidad = "muy alta"
        elif accuracy >= 0.75:
            calidad = "buena"
        elif accuracy >= 0.6:
            calidad = "moderada"
        else:
            calidad = "baja"
        lineas.append(f"Exactitud (accuracy) = {pct}% — precision {calidad}.")

        for metrica in ("precision", "recall", "f1"):
            if metrica in metrics:
                lineas.append(f"{metrica.capitalize()} = {metrics[metrica]:.4f}.")

        if accuracy < 0.6:
            lineas.append("La exactitud es baja. Revisar el balance de clases o probar otro clasificador.")

        return "\n".join(lineas)

    else:
        # Tipo de modelo desconocido: mostrar las métricas tal cual
        lineas = [f"Modelo: {model_type}", "Métricas:"]
        for clave, valor in metrics.items():
            lineas.append(f"  {clave}: {valor}")
        return "\n".join(lineas)


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
    df = data["df"]

    # Determinar qué columnas inspeccionar
    if numeric_columns is not None:
        columnas = numeric_columns
    else:
        columnas = df.select_dtypes(include=np.number).columns.tolist()

    if not columnas:
        return pd.DataFrame(columns=["column", "anomaly_type", "detail"])

    anomalias = []

    for col in columnas:
        serie = df[col].dropna()

        if serie.empty:
            continue

        # --- Columna constante (sin variabilidad) ---
        if serie.nunique() <= 1:
            anomalias.append({
                "column": col,
                "anomaly_type": "valor_constante",
                "detail": f"La columna tiene un único valor: {serie.iloc[0]}"
            })
            continue  # No tiene sentido seguir analizando una columna constante

        # --- Alto porcentaje de nulos ---
        total_filas = len(df)
        nulos = df[col].isnull().sum()
        pct_nulos = round(nulos / total_filas * 100, 2)
        if pct_nulos > 20:
            anomalias.append({
                "column": col,
                "anomaly_type": "alto_porcentaje_nulos",
                "detail": f"{pct_nulos}% de valores nulos ({nulos} de {total_filas} filas)"
            })

        # --- Outliers por IQR ---
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            limite_inf = q1 - 1.5 * iqr
            limite_sup = q3 + 1.5 * iqr
            n_outliers = int(((serie < limite_inf) | (serie > limite_sup)).sum())

            if n_outliers > 0:
                pct_outliers = round(n_outliers / total_filas * 100, 2)
                anomalias.append({
                    "column": col,
                    "anomaly_type": "outliers_iqr",
                    "detail": (
                        f"{n_outliers} valores atípicos ({pct_outliers}%) "
                        f"fuera del rango [{limite_inf:.2f}, {limite_sup:.2f}]"
                    )
                })

        # --- Alta asimetría (skewness) ---
        asimetria = serie.skew()
        if abs(asimetria) > 2:
            direccion = "positiva" if asimetria > 0 else "negativa"
            anomalias.append({
                "column": col,
                "anomaly_type": "alta_asimetria",
                "detail": f"Asimetría {direccion}: {asimetria:.2f} (umbral: ±2)"
            })

    if not anomalias:
        return pd.DataFrame(columns=["column", "anomaly_type", "detail"])

    return pd.DataFrame(anomalias).reset_index(drop=True)

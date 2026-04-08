"""
insights.py
High-level analytical insights derived from statistics and ML results.

Consumes the standard data contract dict from data_loader and results
from statistics / ml_models to produce human-readable summaries.
"""

import re

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

    # Porcentaje de nulos por columna (sólo columnas con al menos un nulo)
    null_pct_per_column = {
        col: round(nulos_por_columna[col] / total_filas * 100)
        for col in columnas_con_nulos
    } if total_filas > 0 else {}

    return {
        "total_rows": total_filas,
        "total_columns": total_columnas,
        "columns_with_nulls": columnas_con_nulos,
        "null_pct_overall": null_pct_global,
        "null_pct_per_column": null_pct_per_column,
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


# ---------------------------------------------------------------------------
# Funciones de resumen en lenguaje natural (español)
# ---------------------------------------------------------------------------

def resumir_calidad(reporte: dict) -> str:
    """
    Genera un párrafo en español resumiendo la calidad del dataset.

    Parameters
    ----------
    reporte : dict
        Salida de data_quality_report().

    Returns
    -------
    str
        Párrafo con frases legibles para usuarios no técnicos.
    """
    lineas = []

    total_rows = reporte.get("total_rows", 0)
    total_columns = reporte.get("total_columns", 0)
    lineas.append(f"El dataset tiene {total_rows} registros y {total_columns} columnas.")

    duplicate_rows = reporte.get("duplicate_rows", 0)
    if total_rows > 0 and duplicate_rows > 0:
        pct = round(duplicate_rows / total_rows * 100)
        lineas.append(f"Se encontraron {duplicate_rows} filas duplicadas ({pct}% del total).")
    else:
        lineas.append("No se encontraron filas duplicadas.")

    null_pct_per_column = reporte.get("null_pct_per_column", {})
    cols_con_nulos = {col: pct for col, pct in null_pct_per_column.items() if pct > 0}
    if cols_con_nulos:
        top_cols = sorted(cols_con_nulos.items(), key=lambda x: x[1], reverse=True)[:3]
        partes = [f"{col} ({pct}%)" for col, pct in top_cols]
        lineas.append(f"Las columnas con más valores faltantes son: {', '.join(partes)}.")

    null_pct_overall = reporte.get("null_pct_overall", 0)
    if null_pct_overall < 10:
        lineas.append("El dataset está en buenas condiciones para el análisis.")

    return "\n".join(lineas)


def resumir_anomalias(anomalias_df: pd.DataFrame) -> str:
    """
    Genera frases en español describiendo cada anomalía detectada.

    Parameters
    ----------
    anomalias_df : pd.DataFrame
        Salida de detect_data_anomalies(). Columnas: column, anomaly_type, detail.

    Returns
    -------
    str
        Párrafo con una oración por anomalía, o mensaje de ausencia.
    """
    if anomalias_df.empty:
        return "No se detectaron anomalías en las columnas numéricas."

    lineas = []
    for _, fila in anomalias_df.iterrows():
        col = fila["column"]
        tipo = fila["anomaly_type"]
        detalle = fila["detail"]

        if tipo == "outliers_iqr":
            m = re.search(r'\((\d+\.?\d*)%\)', detalle)
            if m:
                pct = round(float(m.group(1)))
                lineas.append(f"El {pct}% de los registros tienen valores atípicos en {col}.")
            else:
                lineas.append(f"Se detectaron valores atípicos en la columna {col}.")

        elif tipo == "alta_asimetria":
            m = re.search(r'Asimetría (\w+): (-?\d+\.?\d*)', detalle)
            if m:
                direccion = m.group(1)
                valor = float(m.group(2))
                lineas.append(
                    f"La columna {col} presenta alta asimetría {direccion} ({valor:.2f})."
                )
            else:
                lineas.append(f"La columna {col} presenta alta asimetría en su distribución.")

        elif tipo == "alto_porcentaje_nulos":
            m = re.search(r'(\d+\.?\d*)%', detalle)
            if m:
                pct = round(float(m.group(1)))
                lineas.append(f"La columna {col} tiene un {pct}% de valores faltantes.")
            else:
                lineas.append(f"La columna {col} tiene un alto porcentaje de valores faltantes.")

        elif tipo == "valor_constante":
            lineas.append(f"La columna {col} tiene un único valor en todos los registros (sin variabilidad).")

    return "\n".join(lineas) if lineas else "No se detectaron anomalías en las columnas numéricas."


def resumir_correlaciones(corr_matrix: pd.DataFrame) -> list[str]:
    """
    Genera una lista de frases en español, una por cada par correlacionado.

    Etiquetas de intensidad:
        - |r| > 0.5 → "fuerte"
        - |r| > 0.3 → "moderada"
        - |r| ≤ 0.3 → ignorado

    Etiquetas de dirección: "positiva" (r > 0) / "negativa" (r < 0).
    Excluye autocorrelaciones (r = 1.0) y duplicados recorriendo el triángulo
    superior de la matriz.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Matriz de correlación cuadrada (salida de correlation_matrix()).

    Returns
    -------
    list[str]
        Una cadena por par relevante. Si no se encuentra ninguno, lista con un
        único mensaje de ausencia.
    """
    if corr_matrix.empty or corr_matrix.shape[1] < 2:
        return ["No se encontraron correlaciones relevantes entre las variables numéricas."]

    mensajes = []
    cols = corr_matrix.columns.tolist()

    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            if col_a not in corr_matrix.index or col_b not in corr_matrix.columns:
                continue
            r = corr_matrix.loc[col_a, col_b]
            if pd.isna(r) or abs(r) >= 1.0:
                continue
            abs_r = abs(r)
            if abs_r > 0.5:
                intensidad = "fuerte"
            elif abs_r > 0.3:
                intensidad = "moderada"
            else:
                continue
            direccion = "positiva" if r > 0 else "negativa"
            mensajes.append(
                f"Existe una correlación {intensidad} {direccion} (r={r:.2f}) entre {col_a} y {col_b}."
            )

    if not mensajes:
        return ["No se encontraron correlaciones relevantes entre las variables numéricas."]

    return mensajes


def resumir_outliers(data: dict, column: str, method: str, mascara: pd.Series) -> str:
    """
    Genera una frase en español describiendo los outliers detectados en una columna.

    Parameters
    ----------
    data : dict
        Contrato de datos estándar con clave 'df'.
    column : str
        Columna analizada.
    method : str
        Método utilizado: 'iqr', 'zscore' o 'isolation_forest'.
    mascara : pd.Series
        Serie booleana donde True indica un outlier (salida de outlier_detection()).

    Returns
    -------
    str
        Oración descriptiva del resultado.
    """
    df = data["df"]
    total = len(df)
    n_outliers = int(mascara.sum())

    metodos_display = {
        "iqr": "IQR",
        "zscore": "Z-Score",
        "isolation_forest": "Isolation Forest",
    }
    metodo_nombre = metodos_display.get(method, method.upper())

    if n_outliers == 0:
        return f"No se detectaron valores atípicos en {column} con el método {metodo_nombre}."

    pct = round(n_outliers / total * 100)
    pct_str = f"{pct}%" if pct > 0 else "<1%"
    return (
        f"El {pct_str} de los registros ({n_outliers} de {total}) tienen valores atípicos "
        f"en {column} según el método {metodo_nombre}."
    )


def resumir_clusters(df_clusters: pd.DataFrame, n_clusters: int) -> str:
    """
    Genera un resumen en español de las características de cada cluster.

    Parameters
    ----------
    df_clusters : pd.DataFrame
        DataFrame con todas las columnas originales más la columna 'cluster'.
    n_clusters : int
        Número de clusters solicitados.

    Returns
    -------
    str
        Párrafo con una oración por cluster.
    """
    if df_clusters.empty:
        return "No hay datos de clustering disponibles."

    lineas = [f"Se detectaron {n_clusters} agrupaciones principales."]

    cols_num = df_clusters.select_dtypes(include=np.number).columns.tolist()
    cols_features = [c for c in cols_num if c != "cluster"]
    cols_mostrar = cols_features[:2]

    clusters_unicos = sorted(df_clusters["cluster"].dropna().unique())

    for cluster_id in clusters_unicos:
        mascara = df_clusters["cluster"] == cluster_id
        n_registros = int(mascara.sum())

        if not cols_mostrar:
            lineas.append(f"El grupo {int(cluster_id)} tiene {n_registros} registros.")
            continue

        partes = []
        for col in cols_mostrar:
            mean_val = df_clusters.loc[mascara, col].mean()
            partes.append(f"{col} promedio de {mean_val:.1f}")

        if len(partes) == 1:
            desc = partes[0]
        else:
            desc = f"{partes[0]} y {partes[1]}"

        lineas.append(f"El grupo {int(cluster_id)} tiene {desc} ({n_registros} registros).")

    return "\n".join(lineas)


def resumir_modelo(metricas: dict, model_type: str, target_col: str) -> str:
    """
    Genera un resumen en español del desempeño del modelo entrenado.

    Parameters
    ----------
    metricas : dict
        Salida de evaluate_model(). Claves varían según tipo de modelo.
    model_type : str
        Tipo de modelo: 'random_forest', 'linear_regression', etc.
    target_col : str
        Nombre de la columna objetivo predicha por el modelo.

    Returns
    -------
    str
        Oración o párrafo legible para usuarios no técnicos.
    """
    def _calidad(score: float) -> str:
        if score > 0.85:
            return "excelente"
        elif score > 0.70:
            return "bueno"
        elif score > 0.50:
            return "aceptable"
        return "débil"

    if "accuracy" in metricas:
        accuracy = metricas.get("accuracy", 0)
        pct = round(accuracy * 100)
        calidad = _calidad(accuracy)
        return (
            f"El modelo predice con {pct}% de exactitud el valor de {target_col}. "
            f"Rendimiento: {calidad}."
        )

    if "r2" in metricas:
        r2 = metricas.get("r2", 0)
        rmse = metricas.get("rmse")
        pct = round(r2 * 100)
        calidad = _calidad(r2)
        partes = [f"R²={r2:.2f}"]
        if rmse is not None:
            partes.append(f"RMSE={rmse:.1f}")
        metricas_str = ", ".join(partes)
        return (
            f"El modelo explica el {pct}% de la variación en {target_col} "
            f"({metricas_str}). Rendimiento: {calidad}."
        )

    return "No se pudo generar un resumen para el tipo de modelo especificado."


def resumir_distribucion(column: str, mean: float, median: float, skewness: float) -> str:
    """
    Genera una frase en español interpretando la forma de la distribución de una columna.

    Parameters
    ----------
    column : str
        Nombre de la columna analizada.
    mean : float
        Media aritmética de los valores no nulos.
    median : float
        Mediana de los valores no nulos.
    skewness : float
        Coeficiente de asimetría (skew) de los valores no nulos.

    Returns
    -------
    str
        Oración descriptiva de la distribución.
    """
    relative_diff = abs(mean - median) / (abs(mean) + 1e-9)

    if relative_diff < 0.05 and abs(skewness) < 0.5:
        return (
            f"La distribución de {column} es aproximadamente simétrica "
            f"(media={mean:.2f}, mediana={median:.2f})."
        )

    if mean > median and skewness > 0.5:
        return (
            f"La distribución de {column} tiene asimetría positiva — "
            f"hay valores altos que elevan la media ({mean:.2f}) "
            f"por encima de la mediana ({median:.2f})."
        )

    if mean < median and skewness < -0.5:
        return (
            f"La distribución de {column} tiene asimetría negativa — "
            f"hay valores bajos que arrastran la media ({mean:.2f}) "
            f"por debajo de la mediana ({median:.2f})."
        )

    # Caso intermedio: diferencia apreciable pero no encaja en los criterios anteriores
    direccion = "positiva" if skewness > 0 else "negativa"
    return (
        f"La distribución de {column} muestra una leve asimetría {direccion} "
        f"(media={mean:.2f}, mediana={median:.2f}, asimetría={skewness:.2f})."
    )

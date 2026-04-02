import pandas as pd
import numpy as np


def _obtener_columnas_numericas(df: pd.DataFrame) -> list:
    """
    Devuelve una lista con los nombres de las columnas numéricas.
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Incluye conteo, media, desviación estándar, mínimo, cuartiles y máximo.
    """

    df_numerico = df.select_dtypes(include=np.number)

    if df_numerico.empty:
        return pd.DataFrame()

    return df_numerico.describe().round(2)


def estadisticas_adicionales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas adicionales:
    mediana, varianza, asimetría y curtosis.
    """
    df_numerico = df.select_dtypes(include=np.number)

    if df_numerico.empty:
        return pd.DataFrame()

    estadisticas = pd.DataFrame({
        "mediana": df_numerico.median(),
        "varianza": df_numerico.var(),
        "asimetria": df_numerico.skew(),
        "curtosis": df_numerico.kurt()
    })

    return estadisticas.round(2)


def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre columnas numéricas.
    """
    df_numerico = df.select_dtypes(include=np.number)

    # Se necesitan al menos 2 columnas para correlación
    if df_numerico.shape[1] < 2:
        return pd.DataFrame()

    #Método de pandas que calcula la correlación entre todas las columnas.
    return df_numerico.corr().round(2)


def detectar_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    Detecta valores atípicos usando el método IQR.
    Devuelve un diccionario por columna con:
    - cantidad_outliers
    - porcentaje_outliers
    - limite_inferior
    - limite_superior
    """
    resultados = {}
    columnas_numericas = _obtener_columnas_numericas(df)

    total_filas = len(df)

    for columna in columnas_numericas:
        serie = df[columna].dropna()

        if serie.empty:
            resultados[columna] = {
                "cantidad_outliers": 0,
                "porcentaje_outliers": 0.0,
                "limite_inferior": None,
                "limite_superior": None
            }
            continue

        #IQR es la diferencia entre cuartil 3 (o percentil 75) y cuartil 1 (o percentil 25)
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        es_outlier = (serie < limite_inferior) | (serie > limite_superior)

        cantidad_outliers = int(es_outlier.sum())
        porcentaje_outliers = round((cantidad_outliers / total_filas) * 100, 2) if total_filas > 0 else 0.0

        resultados[columna] = {
            "cantidad_outliers": cantidad_outliers,
            "porcentaje_outliers": porcentaje_outliers,
            "limite_inferior": round(limite_inferior, 2),
            "limite_superior": round(limite_superior, 2)
        }

    return resultados


def resumen_outliers(outliers: dict) -> pd.DataFrame:
    """
    Convierte el diccionario de outliers en un DataFrame para mostrarlo fácilmente.
    """
    if not outliers:
        return pd.DataFrame()

    filas = []
    for columna, info in outliers.items():
        filas.append({
            "columna": columna,
            "cantidad_outliers": info["cantidad_outliers"],
            "porcentaje_outliers": info["porcentaje_outliers"],
            "limite_inferior": info["limite_inferior"],
            "limite_superior": info["limite_superior"]
        })

    return pd.DataFrame(filas)


def ejecutar_estadisticas(data: dict) -> dict:
    """
    Función principal del módulo.
    Recibe el diccionario producido por data_loader.
    """
    if "df" not in data:
        raise KeyError("El diccionario de entrada no contiene la clave 'df'.")

    df = data["df"]
    columnas_numericas = _obtener_columnas_numericas(df)

    resultados = {
        "shape": df.shape,
        "columnas_numericas": columnas_numericas,
        "estadisticas_descriptivas": estadisticas_descriptivas(df),
        "estadisticas_adicionales": estadisticas_adicionales(df),
        "matriz_correlacion": matriz_correlacion(df),
        "outliers": detectar_outliers_iqr(df)
    }

    return resultados


# Alias en inglés (IMPORTANTE para compatibilidad con el equipo)
get_descriptive_stats = estadisticas_descriptivas
get_additional_stats = estadisticas_adicionales
get_correlation_matrix = matriz_correlacion
get_outliers_iqr = detectar_outliers_iqr
summarize_outliers_table = resumen_outliers
analyze_statistics = ejecutar_estadisticas
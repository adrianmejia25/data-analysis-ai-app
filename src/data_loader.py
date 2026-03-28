"""
data_loader.py
Handles loading and initial inspection of data files.

Data contract: all public functions that return data return a dict with:
    - df     (pd.DataFrame): the loaded dataset
    - dtypes (dict):         column name -> dtype string
    - nulls  (dict):         column name -> null count
    - shape  (tuple):        (rows, cols)
"""

import os
import pandas as pd


def _detectar_tipos(df: pd.DataFrame) -> dict:
    # Clasificar cada columna como 'numeric', 'temporal' o 'categorical'
    tipos = {}
    for columna in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[columna]):
            tipos[columna] = "temporal"
        elif pd.api.types.is_numeric_dtype(df[columna]):
            tipos[columna] = "numeric"
        else:
            # Intentar convertir a fecha para detectar columnas de texto con fechas
            try:
                pd.to_datetime(df[columna], errors="raise")
                tipos[columna] = "temporal"
            except (ValueError, TypeError):
                # Si no es numérica ni temporal, es categórica
                tipos[columna] = "categorical"
    return tipos


def _limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminar filas completamente vacías (todas sus celdas son NaN)
    df = df.dropna(how="all")
    # Eliminar filas duplicadas (mismo valor en todas las columnas)
    df = df.drop_duplicates()
    # Quitar espacios en blanco en los nombres de las columnas
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    return df


def cargar_csv(ruta: str) -> dict:
    # Verificar que el archivo existe antes de intentar abrirlo
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    _, ext = os.path.splitext(ruta)
    if ext.lower() != ".csv":
        raise ValueError(f"Esta función solo acepta archivos .csv, se recibió: {ext}")

    try:
        # Cargar el archivo CSV con separador de coma (estándar)
        data_frame = pd.read_csv(ruta)
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo CSV está vacío: {ruta}")
    except Exception:
        # Algunos CSV usan punto y coma como separador, se intenta de nuevo
        try:
            data_frame = pd.read_csv(ruta, sep=";")
        except Exception as e:
            raise ValueError(f"No se pudo leer el archivo CSV '{ruta}': {e}")

    data_frame = _limpiar_dataframe(data_frame)

    return {
        "df": data_frame,
        "dtypes": _detectar_tipos(data_frame),
        "nulls": data_frame.isnull().sum().to_dict(),
        "shape": data_frame.shape
    }


def cargar_excel(ruta: str, hoja: str = 0) -> dict:
    # Verificar que el archivo existe
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    _, ext = os.path.splitext(ruta)
    if ext.lower() not in (".xlsx", ".xls"):
        raise ValueError(f"Esta función solo acepta archivos .xlsx o .xls, se recibió: {ext}")

    try:
        # Cargar la hoja indicada del archivo Excel
        data_frame = pd.read_excel(ruta, sheet_name=hoja)
    except ValueError as e:
        # Ocurre cuando el nombre de la hoja no existe en el archivo
        raise ValueError(f"La hoja no existe en '{ruta}': {e}")
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo Excel '{ruta}': {e}")

    data_frame = _limpiar_dataframe(data_frame)

    return {
        "df": data_frame,
        "dtypes": _detectar_tipos(data_frame),
        "nulls": data_frame.isnull().sum().to_dict(),
        "shape": data_frame.shape
    }


def cargar_archivo(ruta: str, hoja: str = 0) -> dict:
    # Verificar que el archivo existe
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    _, ext = os.path.splitext(ruta)
    ext = ext.lower()

    # Delegar a la función correspondiente según la extensión detectada
    if ext == ".csv":
        return cargar_csv(ruta)
    elif ext in (".xlsx", ".xls"):
        return cargar_excel(ruta, hoja=hoja)
    else:
        raise ValueError(f"Formato '{ext}' no soportado. Use .csv, .xlsx o .xls")


def resumir(data: dict) -> dict:
    df = data["df"]
    total_filas = df.shape[0]

    # Calcular el porcentaje de nulos por columna para facilitar el análisis
    null_pct = {
        col: round((conteo / total_filas) * 100, 2) if total_filas > 0 else 0.0
        for col, conteo in data["nulls"].items()
    }

    return {
        "shape": data["shape"],
        "dtypes": data["dtypes"],
        "nulls": data["nulls"],
        "null_pct": null_pct,
        "columns": list(df.columns)
    }

# Aliases en inglés para compatibilidad con el resto del proyecto
load_csv = cargar_csv
load_excel = cargar_excel
load_file = cargar_archivo
summarize = resumir
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


# ---------------------------------------------------------------------------
# Funciones privadas de limpieza y detección de tipos
# ---------------------------------------------------------------------------

def _limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza básica al DataFrame recibido."""
    # Eliminar filas completamente vacías (todas sus celdas son NaN)
    df = df.dropna(how="all")

    # Quitar espacios en blanco al inicio y final de los nombres de columna
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

    return df


def _detectar_tipos(df: pd.DataFrame) -> dict:
    """
    Detecta automáticamente el tipo semántico de cada columna:
    'numeric', 'temporal' o 'categorical'.

    Retorna un dict {nombre_columna: tipo_string}.
    """
    tipos = {}
    for col in df.columns:
        # Intentar convertir a fecha si no es ya de tipo datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            tipos[col] = "temporal"
        elif pd.api.types.is_numeric_dtype(df[col]):
            tipos[col] = "numeric"
        else:
            # Intentar parsear como fecha (p. ej. columnas con strings de fecha)
            try:
                pd.to_datetime(df[col], errors="raise")
                tipos[col] = "temporal"
            except (ValueError, TypeError):
                # Si no es numérica ni temporal, es categórica
                tipos[col] = "categorical"
    return tipos


def _calcular_nulos(df: pd.DataFrame) -> dict:
    """
    Calcula el porcentaje de valores nulos por columna.

    Retorna un dict {nombre_columna: porcentaje_nulos (float)}.
    Nota: el contrato original pide 'null count'; aquí devolvemos el conteo
    entero para respetar el contrato, y agregamos el porcentaje internamente.
    """
    # Conteo de nulos por columna (entero, para el contrato estándar)
    return df.isnull().sum().to_dict()


def _construir_resultado(df: pd.DataFrame) -> dict:
    """Construye el dict estándar a partir de un DataFrame ya limpio."""
    return {
        "df": df,
        "dtypes": _detectar_tipos(df),
        "nulls": _calcular_nulos(df),
        "shape": df.shape,
    }


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

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

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si la extensión no corresponde a un CSV o el archivo está corrupto.
    """
    # Verificar que el archivo existe antes de intentar cargarlo
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    # Verificar que la extensión sea .csv
    _, ext = os.path.splitext(filepath)
    if ext.lower() != ".csv":
        raise ValueError(
            f"Formato no soportado '{ext}'. Esta función solo acepta archivos .csv"
        )

    try:
        # Intentar cargar con separador estándar (coma)
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo CSV está vacío: {filepath}")
    except pd.errors.ParserError as e:
        # Algunos CSV usan punto y coma; intentar de nuevo con ese separador
        try:
            df = pd.read_csv(filepath, sep=";")
        except Exception:
            raise ValueError(f"No se pudo parsear el archivo CSV: {filepath}. Detalle: {e}")
    except Exception as e:
        raise ValueError(f"Error inesperado al leer el CSV '{filepath}': {e}")

    # Aplicar limpieza básica
    df = _limpiar_dataframe(df)

    # Construir y retornar el diccionario estándar
    return _construir_resultado(df)


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

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si la extensión no corresponde a Excel o el archivo está corrupto.
    """
    # Verificar que el archivo existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    # Verificar que la extensión sea .xlsx o .xls
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in (".xlsx", ".xls"):
        raise ValueError(
            f"Formato no soportado '{ext}'. Esta función solo acepta archivos .xlsx o .xls"
        )

    try:
        # Cargar la hoja indicada (por nombre o índice)
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    except ValueError as e:
        # Ocurre cuando sheet_name no existe en el libro
        raise ValueError(f"Hoja no encontrada en '{filepath}': {e}")
    except Exception as e:
        raise ValueError(f"Error inesperado al leer el Excel '{filepath}': {e}")

    # Aplicar limpieza básica
    df = _limpiar_dataframe(df)

    # Construir y retornar el diccionario estándar
    return _construir_resultado(df)


def load_file(filepath: str, sheet_name: str = 0) -> dict:
    """
    Detecta automáticamente el tipo de archivo por su extensión y lo carga.
    Admite .csv, .xlsx y .xls.

    Parameters
    ----------
    filepath : str
        Ruta al archivo de datos.
    sheet_name : str or int
        Solo relevante para Excel. Hoja a leer (por defecto la primera).

    Returns
    -------
    dict
        Keys: df (DataFrame), dtypes (dict), nulls (dict), shape (tuple).
    """
    # Verificar que el archivo existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    # Delegar a la función correspondiente según la extensión detectada
    if ext == ".csv":
        return load_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return load_excel(filepath, sheet_name=sheet_name)
    else:
        raise ValueError(
            f"Formato '{ext}' no soportado. Use archivos .csv, .xlsx o .xls"
        )


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
    # Extraer el DataFrame del diccionario estándar
    df: pd.DataFrame = data["df"]

    # Calcular porcentaje de nulos para dar más contexto en el resumen
    total_filas = df.shape[0]
    null_pct = {
        col: round((count / total_filas) * 100, 2) if total_filas > 0 else 0.0
        for col, count in data["nulls"].items()
    }

    return {
        "shape": data["shape"],
        "dtypes": data["dtypes"],
        "nulls": data["nulls"],          # conteo absoluto (contrato estándar)
        "null_pct": null_pct,            # porcentaje adicional para visualización
        "columns": list(df.columns),
    }

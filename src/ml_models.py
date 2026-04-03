"""
ml_models.py
Entrenamiento, evaluación y predicción de modelos de machine learning.

Espera el formato de contrato estándar producido por data_loader.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


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
    df = data["df"]

    # Validar que la columna objetivo existe
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame.")

    total_filas = len(df)

    # --- Features numéricas (excluir columna objetivo y columnas con >40% nulos) ---
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    features_num = [
        col for col in columnas_numericas
        if col != target_column and df[col].isnull().sum() / total_filas <= 0.40
    ]

    # --- Features categóricas con baja cardinalidad (≤10 valores únicos) ---
    # Se codifican con label encoding para capturar variables como 'sex', 'embarked', etc.
    columnas_cat = df.select_dtypes(exclude=np.number).columns.tolist()
    features_cat = [
        col for col in columnas_cat
        if col != target_column
        and df[col].nunique(dropna=True) <= 10
        and df[col].isnull().sum() / total_filas <= 0.40
    ]

    features = features_num + features_cat

    if not features:
        raise ValueError("No hay features disponibles tras filtrar columnas con >40% nulos.")

    # Eliminar solo filas donde el target es nulo (las features se imputarán después)
    df_work = df[features + [target_column]].dropna(subset=[target_column]).copy()

    # Validar que hay suficientes filas para dividir
    if len(df_work) < 4:
        raise ValueError(f"El dataset tiene solo {len(df_work)} filas válidas — se necesitan al menos 4.")

    # Codificar columnas categóricas como códigos enteros (label encoding)
    for col in features_cat:
        df_work[col] = df_work[col].astype("category").cat.codes.replace(-1, np.nan)

    X = df_work[features]
    y = df_work[target_column]

    # Dividir ANTES de imputar para evitar data leakage del conjunto de prueba al de entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Imputar nulos restantes con la mediana de X_train — aplicar a ambos conjuntos
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=features, index=X_test.index
    )

    return X_train, X_test, y_train, y_test


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
    # Seleccionar el modelo según el tipo solicitado
    if model_type == "linear_regression":
        modelo = LinearRegression()

    elif model_type == "random_forest":
        # Auto-detectar si el target es clasificación o regresión:
        # si tiene 20 o menos valores únicos enteros se trata como clasificación
        valores_unicos = y_train.dropna().unique()
        es_clasificacion = (
            len(valores_unicos) <= 20
            and all(float(v) == int(float(v)) for v in valores_unicos)
        )
        if es_clasificacion:
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    elif model_type == "logistic_regression":
        # max_iter=1000 para asegurar convergencia en datasets variados
        modelo = LogisticRegression(max_iter=1000, random_state=42)

    else:
        raise ValueError(
            f"Tipo de modelo desconocido: '{model_type}'. "
            "Use 'linear_regression', 'random_forest' o 'logistic_regression'."
        )

    # Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    return modelo


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
        - Regression:     mse (float), rmse (float), r2 (float), mae (float)
        - Classification: accuracy (float), precision (float), recall (float),
                          f1 (float), report (str), confusion_matrix (ndarray)
    """
    # Generar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Detectar automáticamente si el modelo es clasificador o regresor
    if is_classifier(model):
        # --- Métricas de clasificación ---
        # Probabilidades de la clase positiva para el scatter plot (si el modelo las soporta)
        proba = None
        if hasattr(model, "predict_proba"):
            proba_matrix = model.predict_proba(X_test)
            # Para clasificación binaria tomar columna de la clase positiva; para multiclase, None
            if proba_matrix.shape[1] == 2:
                proba = proba_matrix[:, 1].tolist()

        # average='weighted' maneja correctamente clases desbalanceadas
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "proba": proba,
            "report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    else:
        # --- Métricas de regresión ---
        mse = mean_squared_error(y_test, y_pred)
        return {
            "mse": round(mse, 4),
            "rmse": round(np.sqrt(mse), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        }


def predict(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    """
    Generate predictions for new data.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn model.
    X : pd.DataFrame
        Feature matrix to predict on. Accepts a dict of {column: value}
        which will be converted to a single-row DataFrame internally.

    Returns
    -------
    pd.Series
        Predicted values indexed like X.
    """
    # Convertir dict de entrada a DataFrame de una fila
    if isinstance(X, dict):
        X = pd.DataFrame([X])

    # Generar y retornar predicciones como Series
    predicciones = model.predict(X)
    return pd.Series(predicciones, index=X.index)


# ---------------------------------------------------------------------------
# Funciones adicionales: clustering K-Means (requerido por la rúbrica)
# ---------------------------------------------------------------------------

def train_kmeans(
    data: dict,
    n_clusters: int = 3,
    numeric_columns: list[str] | None = None,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit a KMeans clustering model on numeric columns.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    n_clusters : int
        Number of clusters to form.
    numeric_columns : list[str] or None
        Columns to use for clustering. None uses all numeric columns.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (kmeans_model, cluster_labels) — fitted KMeans and integer label array.
    """
    df = data["df"]

    # Seleccionar columnas para clustering
    if numeric_columns is not None:
        df_cluster = df[numeric_columns].select_dtypes(include=np.number)
    else:
        df_cluster = df.select_dtypes(include=np.number)

    if df_cluster.empty:
        raise ValueError("No hay columnas numéricas disponibles para clustering.")

    # Excluir columnas con 50% o más de nulos (evita descartar demasiadas filas)
    umbral_nulos = 0.5 * len(df_cluster)
    df_cluster = df_cluster.loc[:, df_cluster.isnull().sum() < umbral_nulos]

    if df_cluster.empty:
        raise ValueError("Todas las columnas tienen más del 50% de nulos.")

    # Rellenar nulos restantes con la media de cada columna
    df_cluster = df_cluster.fillna(df_cluster.mean(numeric_only=True))

    if len(df_cluster) < n_clusters:
        raise ValueError(
            f"El dataset tiene {len(df_cluster)} filas, pero se pidieron {n_clusters} clusters. "
            "Se necesitan al menos tantas filas como clusters."
        )

    # Entrenar el modelo KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    etiquetas = kmeans.fit_predict(df_cluster)

    return kmeans, etiquetas


def get_cluster_labels(
    data: dict,
    kmeans_model: KMeans,
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Assign cluster labels to every row and return the full DataFrame with a
    new 'cluster' column.

    Parameters
    ----------
    data : dict
        Data contract dict with key 'df' (DataFrame).
    kmeans_model : KMeans
        A fitted KMeans model (output of train_kmeans).
    numeric_columns : list[str] or None
        Columns used when the model was trained. None uses all numeric columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added 'cluster' (int) column.
    """
    df = data["df"].copy()

    # Seleccionar las mismas columnas usadas durante el entrenamiento
    if numeric_columns is not None:
        df_cluster = df[numeric_columns].select_dtypes(include=np.number)
    else:
        df_cluster = df.select_dtypes(include=np.number)

    # Usar solo columnas con menos del 50% de nulos (igual que en train_kmeans)
    umbral_nulos = 0.5 * len(df_cluster)
    df_cluster = df_cluster.loc[:, df_cluster.isnull().sum() < umbral_nulos]

    # Rellenar nulos restantes con la media para poder predecir en todas las filas
    df_cluster = df_cluster.fillna(df_cluster.mean(numeric_only=True))

    # Predecir etiqueta de cluster para todas las filas
    etiquetas = np.full(len(df), np.nan)
    etiquetas[:] = kmeans_model.predict(df_cluster)

    df["cluster"] = etiquetas
    return df

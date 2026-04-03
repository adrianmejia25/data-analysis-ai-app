"""
app.py
Punto de entrada de Streamlit para el panel de análisis de datos.

Ejecutar con:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.insights import (
    data_quality_report,
    detect_data_anomalies,
    model_insight_summary,
    top_correlated_features,
)
from src.ml_models import (
    evaluate_model,
    get_cluster_labels,
    split_data,
    train_kmeans,
    train_model,
)
from src.stats import (
    correlation_matrix,
    descriptive_stats,
    null_report,
    outlier_detection,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_distribution,
    plot_model_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas vacías y duplicadas, limpia nombres de columnas."""
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    return df


def _detectar_tipos(df: pd.DataFrame) -> dict:
    """Clasifica cada columna como 'numeric', 'temporal' o 'categorical'."""
    tipos = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            tipos[col] = "temporal"
        elif pd.api.types.is_numeric_dtype(df[col]):
            tipos[col] = "numeric"
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                tipos[col] = "temporal"
            except (ValueError, TypeError):
                tipos[col] = "categorical"
    return tipos


def cargar_desde_upload(archivo) -> dict:
    """
    Convierte un UploadedFile de Streamlit al contrato de datos estándar.
    Acepta CSV y Excel. Retorna el mismo dict que data_loader.
    """
    nombre = archivo.name.lower()

    if nombre.endswith(".csv"):
        try:
            df = pd.read_csv(archivo)
        except Exception:
            # Algunos CSV usan punto y coma como separador
            archivo.seek(0)
            df = pd.read_csv(archivo, sep=";")
    elif nombre.endswith((".xlsx", ".xls")):
        df = pd.read_excel(archivo)
    else:
        raise ValueError(f"Formato no soportado: '{archivo.name}'. Use .csv, .xlsx o .xls")

    df = _limpiar_dataframe(df)

    return {
        "df": df,
        "dtypes": _detectar_tipos(df),
        "nulls": df.isnull().sum().to_dict(),
        "shape": df.shape,
    }


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------

def seccion_vista_general(data: dict) -> None:
    """Sección 1: Vista General y Calidad del dataset."""
    df = data["df"]
    filas, cols = data["shape"]

    st.header("Vista General y Calidad")

    # --- Información básica ---
    st.subheader("Información del Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", filas)
    col2.metric("Columnas", cols)
    col3.metric("Columnas numéricas", len(df.select_dtypes(include=np.number).columns))

    # Tabla de tipos de columna
    tipos_df = pd.DataFrame(
        [(col, tipo) for col, tipo in data["dtypes"].items()],
        columns=["Columna", "Tipo"]
    )
    st.dataframe(tipos_df, use_container_width=True)

    # --- Reporte de calidad ---
    st.subheader("Reporte de Calidad")
    try:
        reporte = data_quality_report(data)
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas duplicadas", reporte["duplicate_rows"])
        col2.metric("% nulos global", f"{reporte['null_pct_overall']}%")
        col3.metric("Columnas constantes", len(reporte["constant_columns"]))

        if reporte["columns_with_nulls"]:
            st.warning(f"Columnas con nulos: {', '.join(reporte['columns_with_nulls'])}")
        else:
            st.success("No se encontraron valores nulos.")

        if reporte["constant_columns"]:
            st.warning(f"Columnas sin variabilidad: {', '.join(reporte['constant_columns'])}")
    except Exception as e:
        st.error(f"Error al generar reporte de calidad: {e}")

    # --- Reporte de nulos ---
    st.subheader("Detalle de Nulos por Columna")
    try:
        df_nulos = null_report(data)
        st.dataframe(df_nulos, use_container_width=True)
    except Exception as e:
        st.error(f"Error al calcular reporte de nulos: {e}")

    # --- Detección de anomalías ---
    st.subheader("Anomalías Detectadas")
    try:
        anomalias = detect_data_anomalies(data)
        if anomalias.empty:
            st.success("No se detectaron anomalías en las columnas numéricas.")
        else:
            st.dataframe(anomalias, use_container_width=True)
    except Exception as e:
        st.error(f"Error al detectar anomalías: {e}")

    # --- DataFrame crudo ---
    st.subheader("Datos Cargados")
    st.dataframe(df, use_container_width=True)


def seccion_estadisticas(data: dict) -> None:
    """Sección 2: Análisis Estadístico descriptivo y correlaciones."""
    df = data["df"]
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()

    st.header("Análisis Estadístico")

    if not columnas_numericas:
        st.warning("El dataset no tiene columnas numéricas para analizar.")
        return

    # --- Estadísticas descriptivas ---
    st.subheader("Estadísticas Descriptivas")
    try:
        stats_df = descriptive_stats(data)
        st.dataframe(stats_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error al calcular estadísticas descriptivas: {e}")

    # --- Matriz de correlación (tabla) ---
    st.subheader("Matriz de Correlación")
    if len(columnas_numericas) < 2:
        st.info("Se necesitan al menos 2 columnas numéricas para calcular correlación.")
    else:
        try:
            corr_df = correlation_matrix(data)
            st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
        except Exception as e:
            st.error(f"Error al calcular matriz de correlación: {e}")

        # Heatmap de correlación
        try:
            fig_heatmap = plot_correlation_heatmap(data)
            st.pyplot(fig_heatmap)
        except Exception as e:
            st.error(f"Error al graficar heatmap: {e}")

    # --- Distribución de una columna ---
    st.subheader("Distribución de Columna")
    columna_sel = st.selectbox(
        "Selecciona una columna numérica:",
        columnas_numericas,
        key="dist_col"
    )
    try:
        fig_dist = plot_distribution(data, columna_sel)
        st.pyplot(fig_dist)
    except Exception as e:
        st.error(f"Error al graficar distribución: {e}")

    # --- Correlaciones fuertes automáticas (|r| > 0.7) ---
    if len(columnas_numericas) >= 2:
        st.subheader("Correlaciones Fuertes Detectadas")
        try:
            corr = data["df"][columnas_numericas].corr()
            mensajes = []
            # Recorrer triángulo superior para evitar duplicados
            for i, col_a in enumerate(corr.columns):
                for col_b in corr.columns[i + 1:]:
                    r = corr.loc[col_a, col_b]
                    if abs(r) > 0.7:
                        direccion = "positiva" if r > 0 else "negativa"
                        mensajes.append(f"Existe una fuerte correlación {direccion} (r={r:.2f}) entre **{col_a}** y **{col_b}**.")
            if mensajes:
                for msg in mensajes:
                    st.info(msg)
            else:
                st.write("No se detectaron correlaciones fuertes (|r| > 0.7).")
        except Exception as e:
            st.error(f"Error al detectar correlaciones fuertes: {e}")

    # --- Detección de outliers ---
    st.subheader("Detección de Outliers")
    col1, col2 = st.columns(2)
    col_outlier = col1.selectbox("Columna:", columnas_numericas, key="outlier_col")
    metodo = col2.selectbox("Método:", ["iqr", "zscore", "isolation_forest"], key="outlier_method")

    try:
        mascara = outlier_detection(data, col_outlier, method=metodo)
        n_outliers = int(mascara.sum())
        pct = round(n_outliers / len(df) * 100, 2)

        if n_outliers == 0:
            st.success(f"No se detectaron outliers en '{col_outlier}' con el método {metodo.upper()}.")
        else:
            st.warning(f"Se detectaron {n_outliers} outliers ({pct}%) en '{col_outlier}'.")
            st.dataframe(df[mascara], use_container_width=True)
    except Exception as e:
        st.error(f"Error al detectar outliers: {e}")


def seccion_ml(data: dict) -> None:
    """Sección 3: Entrenamiento y evaluación de modelos de Machine Learning."""
    df = data["df"]
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()

    st.header("Machine Learning")

    if len(columnas_numericas) < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas para entrenar un modelo.")
        return

    # --- Selección de columna objetivo ---
    st.subheader("Regresión con Random Forest")
    target_col = st.selectbox(
        "Selecciona la columna objetivo (target):",
        columnas_numericas,
        key="ml_target"
    )

    if st.button("Entrenar modelo", key="btn_train"):
        try:
            # Dividir los datos — el conjunto de entrenamiento y prueba son estrictamente separados
            X_train, X_test, y_train, y_test = split_data(data, target_column=target_col)

            # Entrenar Random Forest solo sobre X_train
            modelo = train_model(X_train, y_train, model_type="random_forest")

            # Evaluar sobre X_test (datos que el modelo nunca vio)
            metricas_test = evaluate_model(modelo, X_test, y_test)

            # Calcular métricas sobre entrenamiento para comparar y detectar overfitting
            metricas_train = evaluate_model(modelo, X_train, y_train)

            # Guardar en session_state para no reentrenar en cada interacción
            st.session_state["modelo"] = modelo
            st.session_state["metricas"] = metricas_test
            st.session_state["metricas_train"] = metricas_train
            st.session_state["y_test"] = y_test.reset_index(drop=True)
            st.session_state["y_pred"] = pd.Series(
                modelo.predict(X_test), name="y_pred"
            ).reset_index(drop=True)
            st.session_state["target_col"] = target_col

        except Exception as e:
            st.error(f"Error al entrenar el modelo: {e}")

    # Mostrar métricas si el modelo ya fue entrenado para esta columna objetivo
    if "metricas" in st.session_state and st.session_state.get("target_col") == target_col:
        metricas = st.session_state["metricas"]
        metricas_train = st.session_state.get("metricas_train", {})
        y_test_stored = st.session_state["y_test"]
        y_pred_stored = st.session_state["y_pred"]

        # Métricas de prueba (conjunto que el modelo nunca vio)
        st.subheader("Métricas — Conjunto de Prueba (test set)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² (test)", metricas.get("r2", "—"))
        col2.metric("RMSE (test)", metricas.get("rmse", "—"))
        col3.metric("MAE (test)", metricas.get("mae", "—"))
        col4.metric("MSE (test)", metricas.get("mse", "—"))

        # Mostrar R² de entrenamiento junto al de prueba para detectar overfitting
        r2_train = metricas_train.get("r2", None)
        r2_test = metricas.get("r2", None)
        if r2_train is not None and r2_test is not None:
            diff = round(r2_train - r2_test, 4)
            st.caption(f"R² entrenamiento: {r2_train} | R² prueba: {r2_test} | Diferencia: {diff}")
            if diff > 0.1:
                st.warning("El modelo muestra posible overfitting (R² entrenamiento muy superior al de prueba).")

        # Resumen de texto desde insights
        try:
            resumen = model_insight_summary(metricas, "random_forest")
            st.info(resumen)
        except Exception as e:
            st.error(f"Error al generar resumen del modelo: {e}")

        # Gráfico real vs predicho — usar arrays ya almacenados con índices alineados
        st.subheader("Valores Reales vs. Predichos (test set)")
        try:
            fig_pred = plot_model_results(y_test_stored, y_pred_stored)
            st.pyplot(fig_pred)
        except Exception as e:
            st.error(f"Error al graficar predicciones: {e}")

        # Features más correlacionadas con el target
        st.subheader("Features Más Correlacionadas con el Target")
        try:
            top_feat = top_correlated_features(data, target_column=target_col)
            st.dataframe(top_feat, use_container_width=True)
        except Exception as e:
            st.error(f"Error al calcular correlaciones: {e}")

    # --- Clustering K-Means ---
    st.subheader("Clustering K-Means")
    n_clusters = st.slider("Número de clusters:", min_value=2, max_value=10, value=3, key="n_clusters")

    if st.button("Ejecutar clustering", key="btn_kmeans"):
        try:
            # Entrenar KMeans con las columnas numéricas disponibles
            kmeans, _ = train_kmeans(data, n_clusters=n_clusters)

            # Agregar etiquetas de cluster al DataFrame original
            df_clusters = get_cluster_labels(data, kmeans)
            st.session_state["df_clusters"] = df_clusters

        except Exception as e:
            st.error(f"Error al ejecutar clustering: {e}")

    # Mostrar resultados de clustering si están disponibles
    if "df_clusters" in st.session_state:
        df_clusters = st.session_state["df_clusters"]
        cols_num = df_clusters.select_dtypes(include=np.number).columns.tolist()
        # Excluir la columna 'cluster' de las numéricas para análisis
        cols_features = [c for c in cols_num if c != "cluster"]

        conteo = df_clusters["cluster"].value_counts().sort_index()
        st.write("**Distribución de clusters:**")
        st.bar_chart(conteo)

        # Gráfico 2D de clusters usando las 2 primeras columnas numéricas
        if len(cols_features) >= 2:
            st.write("**Visualización 2D de clusters:**")
            try:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                col_x, col_y = cols_features[0], cols_features[1]
                fig_cl, ax_cl = plt.subplots(figsize=(8, 5))
                clusters_unicos = sorted(df_clusters["cluster"].dropna().unique())
                colores = cm.tab10.colors
                for i, cluster_id in enumerate(clusters_unicos):
                    mascara = df_clusters["cluster"] == cluster_id
                    ax_cl.scatter(
                        df_clusters.loc[mascara, col_x],
                        df_clusters.loc[mascara, col_y],
                        label=f"Cluster {int(cluster_id)}",
                        color=colores[i % len(colores)],
                        alpha=0.7,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                ax_cl.set_xlabel(col_x)
                ax_cl.set_ylabel(col_y)
                ax_cl.set_title(f"Clusters — {col_x} vs {col_y}")
                ax_cl.legend()
                plt.tight_layout()
                st.pyplot(fig_cl)
            except Exception as e:
                st.error(f"Error al graficar clusters: {e}")

        # Resumen automático: media de columnas numéricas por cluster
        if cols_features:
            st.write("**Características por cluster (media de columnas numéricas):**")
            try:
                resumen_clusters = (
                    df_clusters.groupby("cluster")[cols_features]
                    .mean()
                    .round(2)
                )
                st.dataframe(resumen_clusters, use_container_width=True)
            except Exception as e:
                st.error(f"Error al calcular resumen de clusters: {e}")

        st.write("**Dataset con etiquetas de cluster:**")
        st.dataframe(df_clusters, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Panel de Análisis de Datos",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Panel de Análisis de Datos")

    # --- Carga de archivo ---
    archivo = st.file_uploader(
        "Carga un archivo CSV o Excel para comenzar",
        type=["csv", "xlsx", "xls"],
    )

    if archivo is not None:
        # Cargar solo si el archivo cambió (evita recargar en cada interacción)
        archivo_id = archivo.name + str(archivo.size)
        if st.session_state.get("archivo_id") != archivo_id:
            try:
                data = cargar_desde_upload(archivo)
                st.session_state["data"] = data
                st.session_state["archivo_id"] = archivo_id
                # Limpiar estado de modelo anterior al cambiar de archivo
                for clave in ("modelo", "metricas", "y_test", "X_test", "target_col", "df_clusters"):
                    st.session_state.pop(clave, None)
                st.success(f"Archivo cargado: **{archivo.name}** — {data['shape'][0]} filas × {data['shape'][1]} columnas")
            except Exception as e:
                st.error(f"No se pudo cargar el archivo: {e}")
                return

    # Si ya hay datos en session_state, mostrar la navegación
    if "data" in st.session_state:
        data = st.session_state["data"]

        # Navegación en la barra lateral
        seccion = st.sidebar.selectbox(
            "Sección",
            [
                "Vista General y Calidad",
                "Análisis Estadístico",
                "Machine Learning",
            ],
        )

        if seccion == "Vista General y Calidad":
            seccion_vista_general(data)
        elif seccion == "Análisis Estadístico":
            seccion_estadisticas(data)
        elif seccion == "Machine Learning":
            seccion_ml(data)

    else:
        # Mensaje de bienvenida cuando no hay archivo cargado
        st.markdown(
            """
            ### Bienvenido al Panel de Análisis de Datos

            Este panel te permite explorar, analizar y aplicar modelos de machine learning
            a tus datasets de forma interactiva.

            **Para comenzar**, carga un archivo CSV o Excel usando el botón de arriba.

            **Módulos disponibles:**
            - **Vista General y Calidad** — resumen del dataset, nulos, anomalías
            - **Análisis Estadístico** — estadísticas descriptivas, correlaciones, distribuciones, outliers
            - **Machine Learning** — regresión con Random Forest, clustering K-Means
            """
        )


if __name__ == "__main__":
    main()

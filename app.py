"""
app.py
Streamlit entry point for the data analysis dashboard.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.data_loader import load_csv, load_excel
from src.stats import descriptive_stats, correlation_matrix, null_report
from src.ml_models import split_data, train_model, evaluate_model
from src.visualization import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_scatter,
    plot_bar,
    plot_model_results,
)
from src.insights import (
    top_correlated_features,
    data_quality_report,
    model_insight_summary,
    detect_data_anomalies,
)


def main():
    st.title("Panel de Análisis de Datos")
    # TODO: implement UI


if __name__ == "__main__":
    main()

# Proyecto de Análisis de Datos

Aplicación interactiva de análisis de datos construida con Python y Streamlit.

---

## Requisitos previos

- Python 3.10 o superior
- pip

---

## Instalación

1. **Clonar el repositorio**

   ```bash
   git clone <url-del-repositorio>
   cd <nombre-del-proyecto>
   ```

2. **Crear el entorno virtual**

   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual**

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

---

## Uso

1. Coloca tus archivos de datos (`.csv` o `.xlsx`) en la carpeta `data/`.

2. Ejecuta la aplicación:

   ```bash
   streamlit run app.py
   ```

3. Abre el navegador en la dirección indicada por Streamlit (por defecto `http://localhost:8501`).

---

## Estructura del proyecto

```
├── data/                  # Archivos de datos (ignorados por git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Carga de archivos CSV y Excel
│   ├── statistics.py      # Estadísticas descriptivas e inferenciales
│   ├── ml_models.py       # Entrenamiento y evaluación de modelos ML
│   ├── visualization.py   # Generación de gráficos
│   └── insights.py        # Resúmenes e insights automáticos
├── app.py                 # Punto de entrada de Streamlit
├── requirements.txt       # Dependencias del proyecto
└── README.md
```

---

## Módulos

| Módulo | Responsabilidad |
|---|---|
| `data_loader` | Carga archivos y devuelve el contrato de datos estándar |
| `statistics` | Estadísticas descriptivas, correlaciones y detección de valores atípicos |
| `ml_models` | División de datos, entrenamiento y evaluación de modelos |
| `visualization` | Gráficos de distribución, correlación, dispersión y barras |
| `insights` | Resúmenes de calidad de datos y rendimiento de modelos |

---

## Contrato de datos

Todas las funciones de `data_loader` devuelven un diccionario con las siguientes claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `df` | `pd.DataFrame` | Dataset cargado |
| `dtypes` | `dict` | Nombre de columna → tipo de dato |
| `nulls` | `dict` | Nombre de columna → cantidad de nulos |
| `shape` | `tuple` | `(filas, columnas)` |

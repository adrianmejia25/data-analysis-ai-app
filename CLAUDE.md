# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

```bash
# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app runs at `http://localhost:8501`. Place CSV or Excel files in the `data/` folder — they are read from there by the data loader.

## Architecture

This is a **Streamlit data analysis dashboard** with a modular pipeline in `src/`:

```
app.py  →  src/data_loader.py  →  src/statistics.py
                                →  src/ml_models.py
                                →  src/visualization.py
                                →  src/insights.py
```

All data-loading functions return a **standardized dict**:
```python
{
    "df": pd.DataFrame,
    "dtypes": dict,   # column → dtype
    "nulls": dict,    # column → null count
    "shape": tuple    # (rows, cols)
}
```

Downstream modules (`statistics`, `ml_models`, `visualization`, `insights`) all consume this dict. Keep this contract intact.

### Module responsibilities

| Module | Purpose |
|---|---|
| `data_loader.py` | Load CSV/Excel, return standard dict |
| `statistics.py` | Descriptive stats, correlation, outlier detection |
| `ml_models.py` | Train/evaluate sklearn regression & classification models |
| `visualization.py` | Generate matplotlib/seaborn figures |
| `insights.py` | Human-readable summaries and anomaly reports |

### Implementation status

All `src/` functions currently raise `NotImplementedError` — the signatures and docstrings define the contracts. The Streamlit UI in `app.py` is also a TODO stub.

## Git workflow

After completing any meaningful unit of work, commit and push to GitHub:

```bash
git add <specific files>
git commit -m "short, descriptive message"
git push
```

Commit frequently — after each feature, bug fix, or completed module — so work is never lost and progress is always recoverable from the remote.

## Key dependencies

- `streamlit` — UI
- `pandas`, `numpy` — data handling
- `scikit-learn`, `scipy` — ML and stats
- `matplotlib`, `seaborn`, `altair` — visualization
- `openpyxl` — Excel support

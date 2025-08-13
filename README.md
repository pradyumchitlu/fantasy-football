## Fantasy Football — Machine Learning Predictor
Using machine learning to predict player fantasy output across positions (QB, RB, TE, WR).

Read the background write-up here: [Fantasy Football Analytics (blog)](https://msjsportsanalytics955224621.wordpress.com/2022/09/06/fantasy-football-analytics/).

## Project contents
- `PredictorV3.ipynb`: end-to-end notebook to train and evaluate models per position
- CSVs used by the notebook (already included):
  - `FantasyProject - 2019+2020 QBs.csv`
  - `FantasyProject - 2019+2020 RBs.csv`
  - `FantasyProject - 2019+2020 TEs.csv`
  - `FantasyProject - 2019+2020 WRs.csv`
  - `FantasyProject - 2021 QBs.csv`
  - `FantasyProject - 2021 RBs.csv`
  - `FantasyProject - 2021 TEs.csv`
  - `FantasyProject - 2021 WRs.csv`

## What the notebook does
- Loads 2019–2020 and 2021 historical features/labels for each position
- Splits data into train/test sets (25% test)
- Tunes and trains gradient-boosted tree models (XGBoost) separately for QB/RB/TE/WR
- Evaluates performance with common regression metrics (e.g., RMSE/MAE/R²) and visualizations

## Requirements
Python 3.8+ is recommended. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn statsmodels scipy jupyter
```

## Quickstart
1) Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (see above).

3) Launch Jupyter and open the notebook:

```bash
jupyter lab  # or: jupyter notebook
```

4) Open `PredictorV3.ipynb` and update the data paths. The original notebook used Colab-style paths like `/content/...`. Replace those with this directory path. Example preamble you can add near the top of the notebook:

```python
from pathlib import Path
DATA_DIR = Path("/Users/pradyumchitlu/nfl/fantasy-football")  # update if your clone lives elsewhere

df2021qb = pd.read_csv(DATA_DIR / "FantasyProject - 2021 QBs.csv")
df2021rb = pd.read_csv(DATA_DIR / "FantasyProject - 2021 RBs.csv")
df2021te = pd.read_csv(DATA_DIR / "FantasyProject - 2021 TEs.csv")
df2021wr = pd.read_csv(DATA_DIR / "FantasyProject - 2021 WRs.csv")

df201920qb = pd.read_csv(DATA_DIR / "FantasyProject - 2019+2020 QBs.csv")
df201920rb = pd.read_csv(DATA_DIR / "FantasyProject - 2019+2020 RBs.csv")
df201920te = pd.read_csv(DATA_DIR / "FantasyProject - 2019+2020 TEs.csv")
df201920wr = pd.read_csv(DATA_DIR / "FantasyProject - 2019+2020 WRs.csv")
```

5) Run the cells to train and evaluate the models for each position.

## Notes
- The notebook uses `RandomizedSearchCV` to tune `XGBRegressor` hyperparameters for each position separately.
- If you want fully reproducible runs, set seeds for NumPy and scikit-learn where applicable; the XGBoost estimators in the notebook are initialized with a fixed seed for tuning.
- XGBoost builds can require OpenMP on macOS. If you hit performance/build warnings, consider `brew install libomp`.

## License
See `LICENSE` in this directory.

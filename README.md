# Chlorophyll-a Forecasting

Academic machine learning project at Instituto Superior Técnico (2026/2027), developed by **Inês Silva Brás and Maria Oliveira Maia — Group 32**.

Both authors worked jointly on data analysis, preprocessing, model development, validation, interpretation and preparation of the submission.

## Goal

Predict daily chlorophyll-a concentration from past chlorophyll observations and current/past environmental measurements. This is a one-day prediction setting with previous-day true observations available, rather than a recursive multi-day forecast. Availability of current environmental measurements is an assumption of the project setting.

## Final model

LASSO regression (`alpha=0.0005`) with 43 features:

- 12 chlorophyll lags;
- current and lagged nitrate (lags 0–3), wind speed (0–4), upwelling (0–2), sea surface temperature (0–5), photosynthetically active radiation (0–4), and mixed-layer depth (0–3);
- annual and semiannual sine/cosine features.

Chlorophyll input lags are upper-clipped at `Q3 + 2 × IQR`, calculated from training targets. Targets are not clipped. Standardisation parameters are learned on training data only, and predictions are constrained to non-negative values. River discharge is not an input to the final model.

## Recorded development results

| Evaluation | R² | RMSE (mg/m³) | MAE (mg/m³) |
| --- | ---: | ---: | ---: |
| Pooled out-of-fold predictions, 2015–2022 | 0.767898 | 0.295428 | 0.177527 |
| Separate 2023 development evaluation | 0.567266 | 0.510798 | 0.194915 |

The 2015–2022 evaluation leaves out one year at a time. Training can use years before and after the held-out year under the project's stationarity assumption: this is **not rolling-origin validation**. The first 18 days after the excluded year are removed from training to prevent its observations entering training lag vectors. Mean annual R² (0.783149) differs from pooled R².

Both 2015–2022 and 2023 were consulted during model development and selection. These scores are **development results, not an untouched independent test estimate**. Hidden 2024 assessment performance is not reported here.

The final configuration was a group decision following comparisons of models, lag structures, seasonal features and outlier treatments; it was not the automatic winner of every exploratory search.

## Repository contents

- `best_model/utils.py`: feature construction and preprocessing.
- `best_model/my_model.py`: prediction interface using a trained artifact.
- `training_code/train.py`: final training entry point.
- `training_code/development/`: the two final-configuration evaluation scripts and their shared support module.

This portfolio extract retains the original selected Python files without changing the modelling logic. Historical search scripts, signed declarations, the original dataset and the fitted pickle are excluded. The training script's historical `--experiment` option only accepts scripts present in this extract.

## Data

The original course dataset is not redistributed. Reproducing the recorded results requires the same externally supplied data; the commands below cannot reproduce those scores without it.

Provide an XLSX file with a `daily_data` sheet containing a consecutive daily sequence for 2015–2023. The final model uses:

`date`, `chlorophyll_a_mg_m3`, `nitrate_umol_l`, `wind_speed_m_s`, `upwelling_index`, `sst_c`, `par_umol_m2_s`, `mixed_layer_depth_m`.

The development support module also constructs historical river-discharge features and therefore requires `river_discharge_index`, although the final predictor does not use these features.

## Run

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and train:

```bash
python -m pip install -r requirements.txt
python training_code/train.py path/to/dataset.xlsx
```

The script checks consistency between batch and daily feature construction, evaluates the fixed configuration on 2023 using 2015–2022 training data, then refits on 2015–2023 and writes `best_model/model.pkl`.

Run both final-configuration development evaluations before refitting:

```bash
python training_code/train.py path/to/dataset.xlsx --reproduce-development
```

The training entry point also accepts CSV; the development evaluation scripts require XLSX. Dependency ranges are provided for setup, not an exact historical environment lock.

## Verification status

The selected Python files have been syntax-checked for this portfolio extract and the training command's help interface checked. The original dataset is not included, so the full training pipeline and recorded metrics have not been rerun for this extract.

## Author

[Maria Oliveira Maia — LinkedIn](https://www.linkedin.com/in/mariaoliveiramaia/)

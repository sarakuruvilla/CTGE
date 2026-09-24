# CTGE-AQI reproducibility repository

This repository contains the  code and frozen analysis artifacts for the manuscript's **Context-aware Temporal Graph-Enhanced (CTGE) representation framework for one-day-ahead AQI forecasting**.

The repository is organized like this: it contains the Rohan Rao/CPCB data-preparation path, optional ERA5 acquisition and matching, CPDM/ERDM construction, AFSA reliability calibration, seven-seed CTGE evaluation, CPCB operational analysis, dependence-aware statistical testing, the same-data external comparator suite (including iTransformer), nested leave-one-city-out (LOCO) transfer evaluation, and figure/table generation.

## 1. Locked manuscript architecture

The reproducible component ladder is:

- **C1** = protected pollutant/AQI state + intraday dynamics.
- **C2** = C1 + integrated, regime-adaptive **CPDM–ERDM** contextual residual correction.
- **C3 / Full CTGE** = C2 + hierarchical **AFSA reliability calibration**.

Seven fixed forecast seeds are used throughout the component analysis:

`11, 22, 33, 44, 55, 66, 77`

Five AFSA optimization seeds are used for search-stability checks:

`101, 202, 303, 404, 505`

The locked development protocol uses 2018 for model fitting and 2019 for development/validation. All fitted transforms, context models, feature selection, gate thresholds, and reliability calibration are learned only from the allowed training/development observations. The one-day-ahead target is `AQI(t+1)`.

The locked reference results are included in `reference_outputs/` so a reviewer can inspect the exact manuscript-facing tables independently of rerunning the full pipeline.

## 2. Data source and expected files

The primary public dataset is **Rohan Rao, “Air Quality Data in India (2015–2020)”**, compiled from CPCB records and released under CC0:

<https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>

Download and place these files in `data/raw/`:

```text
data/raw/
├── city_day.csv
├── city_hour.csv
├── station_day.csv
├── station_hour.csv
└── stations.csv
```

The seven cities used in the manuscript are:

`Bengaluru, Hyderabad, Chennai, Delhi, Jaipur, Lucknow, Gurugram`.


## 3. ERA5 audit trail

Two ERA5 acquisition routes are provided which shows the acquisition/matching/integration workflow:

- `scripts/03_download_era5_cds.py` — official Copernicus CDS route.
- `scripts/03b_download_era5_openmeteo.py` — historical ERA5 API fallback used during development.
- `scripts/04_match_integrate_era5.py` — nearest-grid matching and daily aggregation/integration.

Official ERA5 single-level documentation:

<https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>

**Important scope note.** CPDM's expected-value model (`ctge/cpdm.py`, manuscript Section 3.2) uses six ERA5-derived meteorological covariates — temperature, relative humidity, surface pressure, precipitation, wind speed, and boundary-layer height — alongside calendar/seasonal/diurnal context. `scripts/04_match_integrate_era5.py` must be run (after ERA5 acquisition, `03` or `03b`) before `scripts/05_build_cpdm_erdm.py` for CPDM to receive these covariates; `05_build_cpdm_erdm.py` will load the ERA5-integrated master automatically if present, and otherwise falls back to the non-ERA5 base master with an explicit warning printed at runtime (not a silent substitution), since a run without ERA5 does not reproduce the manuscript's described CPDM configuration.

## 4. Repository map

```text
CTGE_AQI_Reproducibility_Repository/
├── README.md
├── MANIFEST.md
├── config.yaml
├── requirements.txt
├── environment.yml
├── run_all.py
├── configs/
│   ├── cities.csv
│   ├── frozen_c2_config.csv
│   └── frozen_c3_config.csv
├── ctge/
│   ├── data.py
│   ├── features.py
│   ├── cpdm.py
│   ├── cpdm2.py
│   ├── erdm.py
│   ├── models.py
│   ├── models2.py
│   ├── afsa.py
│   ├── itransformer.py
│   ├── stats.py
│   ├── cpcb.py
│   └── loco.py
├── scripts/
│   ├── 00_validate_inputs.py
│   ├── 01_prepare_rohan_rao.py
│   ├── 02_build_intraday.py
│   ├── 03_download_era5_cds.py
│   ├── 03b_download_era5_openmeteo.py
│   ├── 04_match_integrate_era5.py
│   ├── 04_match_integrate_era5_2.py
│   ├── 05_build_cpdm_erdm.py
│   ├── 05_build_cpdm_erdm2.py
│   ├── 06_afsa_reliability.py
│   ├── 07_run_seven_seed_ctge.py
│   ├── 08_cpcb_analysis.py
│   ├── 09_statistical_tests.py
│   ├── 10_external_comparators.py
│   ├── 11_nested_loco.py
│   └── 12_generate_tables_figures.py
├── reference_outputs/
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
└── outputs/
```



## 5. Installation

Recommended: Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

If PyTorch installation on a specific CUDA platform requires a platform-specific wheel, install the appropriate PyTorch build first, then run the remaining requirements.

## 6. Reproduce the core pipeline

After downloading the five Rohan Rao CSVs into `data/raw/`:

```bash
python scripts/00_validate_inputs.py
python scripts/01_prepare_rohan_rao.py
python scripts/02_build_intraday.py
python scripts/05_build_cpdm_erdm.py
python scripts/07_run_seven_seed_ctge.py
python scripts/08_cpcb_analysis.py
python scripts/10_external_comparators.py
python scripts/09_statistical_tests.py
python scripts/11_nested_loco.py
python scripts/12_generate_tables_figures.py
```

Or run the complete non-ERA5 path:

```bash
python run_all.py
```

To also download/match ERA5 before the core analysis:

```bash
python run_all.py --with-era5
```

ERA5 is not consumed by the locked seven-city core unless the analyst deliberately builds an ERA5-augmented experiment.

## 7. Rohan Rao preprocessing and intraday dynamics

`scripts/01_prepare_rohan_rao.py`:

1. validates city/station identifiers;
2. harmonizes Bengaluru/Bangalore and Gurugram/Gurgaon naming;
3. coerces pollutant/AQI columns to numeric values;
4. treats physically invalid negative concentrations as missing;
5. removes duplicate city-date / city-hour / station-date / station-hour records;
6. creates missingness indicators and strictly lagged pollutant/AQI state variables;
7. defines the one-day-ahead `target_AQI_t1`.

`scripts/02_build_intraday.py` converts hourly pollutant/AQI records into same-day intraday descriptors (coverage, moments, quantiles, range/IQR, within-day slope, four day-part means, and peak timing) and merges them with the daily protected state. No future-day value is used to construct an intraday feature.

## 8. CPDM and ERDM construction

### CPDM — Context-conditioned Pollutant Deviation Modeling

`ctge/cpdm.py` estimates an expected pollutant state from leakage-safe context. Ridge and HistGradientBoosting candidate expectation models are compared on an inner chronological pre-validation split. The fitted expected value is used to construct a robust standardized deviation:

```text
z_p(t) = clip[(P_p(t) - P_hat_p(t)) / robust_scale_p, -6, 6]
```

The feature set includes current contextual deviation, deviation momentum, multi-scale trailing deviation means/dispersion, persistence, and system-level abnormality summaries.

### ERDM — Event, Regime, and Relational Dynamics Modeling

`ctge/erdm.py` builds trailing relational graphs over CPDM deviation states. It reports graph strength/density/dispersion/spectral summaries and graph-drift statistics over 7/14/30-day trailing windows. Relational drift is computed only from observations available at the forecast origin. Event/regime columns can be supplied through `data/external/event_calendar.csv`; interactions are constructed without using future AQI.

## 9. C1 → C2 → C3 evaluation

`scripts/07_run_seven_seed_ctge.py` runs all seven fixed seeds and writes:

- `outputs/ctge_7seed_metrics.csv`
- `outputs/ctge_7seed_predictions.csv`
- `outputs/ctge_7seed_summary.csv`

The fixed development/reference probe is ExtraTrees with 300 trees, `max_features=0.7`, and `min_samples_leaf=2`.

C2 fits a contextual residual model on CPDM/ERDM features and applies the frozen per-city gate family/hyperparameters from `configs/frozen_c2_config.csv`.

C3 applies the frozen hierarchical AFSA reliability configuration from `configs/frozen_c3_config.csv`. The protected C1 path is always retained; contextual information only modifies the residual correction.

## 10. AFSA reliability calibration

`scripts/06_afsa_reliability.py` implements the auditable AFSA search used for reliability calibration. `ctge/afsa.py` contains prey, follow, and swarm moves over a finite candidate grid. The search space separates:

- reliability signal family;
- threshold quantile;
- low-regime reliability weight;
- high-regime reliability weight.

Five AFSA seeds are supported. The frozen manuscript-facing solution is kept in `configs/frozen_c3_config.csv`, while a new search is written separately to `outputs/afsa_selected_config_research.csv` so the frozen values are never overwritten automatically.

## 11. Seven-seed and prediction-level statistics

`scripts/09_statistical_tests.py` reports complementary evidence rather than treating seed variation as inference:

- mean ± SD over the seven fixed forecast seeds;
- per-city Diebold–Mariano tests with 7-day Newey–West/HAC variance;
- 5,000-replicate moving-block bootstrap CIs for delta-RMSE with 7-day blocks;
- Holm step-down correction across the seven cities within each comparison;
- rank-biserial effect size based on paired daily absolute-error differences;
- exact one-sided cross-city Wilcoxon signed-rank tests;
- Friedman, Quade, and Nemenyi analyses for the final multi-model comparator suite.

Daily losses are averaged across seeds before prediction-level inference; seed runs are **not** incorrectly treated as independent daily observations.

## 12. CPCB practical / operational analysis

`scripts/08_cpcb_analysis.py` uses official CPCB category thresholds `50/100/200/300/400` and evaluates:

- AQI category accuracy;
- macro-F1;
- quadratic weighted kappa;
- high-risk (`AQI > 300`) F1;
- boundary-proximal RMSE;
- category-transition-day RMSE;
- boundary-width sensitivity at ±10, ±15, ±20, ±25, and ±30 AQI units.

The manuscript's principal operational endpoints are the ±20 boundary-proximal RMSE and observed category-transition-day RMSE; broader classification metrics are retained as heterogeneous context and are not presented as uniformly improving in every city.

## 13. External comparator suite and iTransformer

`scripts/10_external_comparators.py` evaluates the final empirical comparator set on the same city split, forecast horizon, and common evaluation dates:

1. Persistence
2. Ridge
3. HistGradientBoosting
4. Random Forest
5. XGBoost
6. LightGBM
7. iTransformer (input-matched reproduction)
8. Full CTGE

**TimeMixer is intentionally not part of the final empirical suite.**

The iTransformer implementation in `ctge/itransformer.py` is written from scratch using the core published inverted-token principle: each variable's lookback history is embedded as a token and self-attention operates across variable tokens. It does not import code from the authors' repository.

Reference:

Y. Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024.

<https://openreview.net/forum?id=JePfAI8fah>

## 14. Nested zero-shot LOCO

`scripts/11_nested_loco.py` separates within-city multi-city consistency from genuine unseen-city transfer.

For each outer held-out city:

1. the city is removed completely from fitting;
2. only the remaining six source cities are used;
3. a contextual-transfer shrinkage scale is selected by inner source-city validation;
4. the conservative rule chooses the largest candidate scale whose inner source-city folds are all positive;
5. the selected rule is applied zero-shot to the seventh city;
6. the process rotates across all seven cities and all seven forecast seeds.

The locked manuscript reference uses a transfer scale of `0.35`, selected from source-city inner validation only.

## 15. Figure and table generation

`scripts/12_generate_tables_figures.py` converts fresh outputs into:

- C1/C2/C3 mean ± SD tables;
- component RMSE figure;
- external comparator average-rank table;
- Nemenyi pairwise table;
- critical-difference/rank-axis figure;
- CPCB practical table;
- LOCO table;
- component statistical table.



## 16. Frozen reference outputs

`reference_outputs/` contains the exact locked analysis tables used for manuscript integration, including:

- full seven-seed C2/C3 rows and predictions where available;
- component prediction-level statistics;
- CPCB seven-seed/city/statistical summaries;
- frozen C2/C3 configuration tables;
- external comparator rank/statistic summaries;
- iTransformer aggregate comparison summary;
- nested LOCO city/statistical summary.

The key frozen audit points represented by these files are:

- C1→C2: 7/7 cities positive, 49/49 city×seed comparisons positive;
- C2→C3: 7/7 cities positive, 49/49 positive;
- C1→Full CTGE: 7/7 cities positive, 49/49 positive;
- CPCB boundary-proximal RMSE: 7/7 cities and 49/49 seed comparisons positive;
- final eight-model ranking: CTGE average rank 1.00; Friedman and Quade significant; Nemenyi significant vs iTransformer and Persistence;
- nested zero-shot LOCO: positive mean transfer gain in 7/7 held-out cities, 45/49 seed comparisons positive.

## 17. Reproducibility / interpretation boundary

The 2019 C1/C2/C3 component-ladder significance calculations are **development/post-selection evidence**, because the same development period was used to select and freeze C2/C3 settings. The repository preserves that distinction explicitly rather than presenting those p-values as an independent untouched-test confirmation.

The external comparator, CPCB, AFSA-control, and LOCO scripts are separated so a reviewer can inspect each claim and its leakage controls independently.

## 18. Quick verification

For a full rerun:

```bash
python run_all.py
```

For a faster core verification without external iTransformer and LOCO retraining:

```bash
python scripts/00_validate_inputs.py
python scripts/01_prepare_rohan_rao.py
python scripts/02_build_intraday.py
python scripts/05_build_cpdm_erdm.py
python scripts/07_run_seven_seed_ctge.py
python scripts/08_cpcb_analysis.py
python scripts/09_statistical_tests.py
python scripts/12_generate_tables_figures.py
```

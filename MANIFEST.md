# Reproducibility manifest

| Editor-requested item | Repository implementation |
|---|---|
| Data-processing scripts | `scripts/00_validate_inputs.py`, `01_prepare_rohan_rao.py`, `02_build_intraday.py` |
| Rohan Rao preparation/preprocessing | `01_prepare_rohan_rao.py` |
| ERA5 acquisition | `03_download_era5_cds.py` (+ Open-Meteo fallback in comments/README) |
| ERA5 matching/integration | `04_match_integrate_era5.py` |
| CPDM/ERDM construction | `05_build_cpdm_erdm.py`, `ctge/cpdm.py`, `ctge/erdm.py` |
| AFSA / reliability calibration | `06_afsa_reliability.py`, `ctge/afsa.py` |
| Model training / seven-seed evaluation | `07_run_seven_seed_ctge.py` |
| CPCB boundary/transition analysis | `08_cpcb_analysis.py` |
| Statistical testing | `09_statistical_tests.py` |
| External baselines + iTransformer | `10_external_comparators.py`, `ctge/itransformer.py` |
| Nested LOCO | `11_nested_loco.py`, `ctge/loco.py` |
| Figure/table generation | `12_generate_tables_figures.py` |
| End-to-end orchestration | `run_all.py` |
| Frozen per-city configs | `configs/frozen_c2_config.csv`, `configs/frozen_c3_config.csv` |
| Locked reference results | `reference_outputs/` |

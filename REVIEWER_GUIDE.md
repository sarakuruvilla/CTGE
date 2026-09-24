# Reviewer verification guide

This file provides the shortest path through the repository.

1. **Data provenance and preprocessing** — `scripts/00_validate_inputs.py`, `01_prepare_rohan_rao.py`, `02_build_intraday.py`.
2. **ERA5 audit trail** — `03_download_era5_cds.py` or `03b_download_era5_openmeteo.py`, then `04_match_integrate_era5.py`.
3. **CPDM/ERDM** — `05_build_cpdm_erdm.py`, with equations/logic in `ctge/cpdm.py` and `ctge/erdm.py`.
4. **AFSA reliability calibration** — `06_afsa_reliability.py` and `ctge/afsa.py`; frozen configurations are in `configs/`.
5. **Seven-seed CTGE evaluation** — `07_run_seven_seed_ctge.py`.
6. **CPCB practical analysis** — `08_cpcb_analysis.py`.
7. **Statistical validation** — `09_statistical_tests.py`.
8. **External baselines/iTransformer** — `10_external_comparators.py` and `ctge/itransformer.py`.
9. **Nested zero-shot LOCO** — `11_nested_loco.py`.
10. **Tables/figures** — `12_generate_tables_figures.py`.

`reference_outputs/` contains the locked manuscript-facing CSVs so the numerical claims can be checked without retraining every model.

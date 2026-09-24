# Data layout

Place the Rohan Rao / Kaggle **Air Quality Data in India (2015–2020)** files in `data/raw/` using these exact names:

- `city_day.csv`
- `city_hour.csv`
- `station_day.csv`
- `station_hour.csv`
- `stations.csv`

The original dataset is CC0 and was compiled from CPCB records. The repository intentionally does not redistribute the large raw files.

Optional/context files go in `data/external/`:

- `event_calendar.csv` — columns: `Date` plus binary/context columns. If absent, `scripts/01_prepare_rohan_rao.py` creates a minimal seasonal/calendar file.
- ERA5 NetCDF/CSV output produced by `scripts/03_download_era5_cds.py` / `scripts/04_match_integrate_era5.py`.

No script should fit a transform on validation/test observations before the chronological split is fixed.

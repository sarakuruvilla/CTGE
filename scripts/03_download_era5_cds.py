#!/usr/bin/env python
"""Download ERA5 single-level variables for the seven CTGE cities via Copernicus CDS.

Prerequisites:
  1. Register at https://cds.climate.copernicus.eu/
  2. Configure ~/.cdsapirc
  3. pip install cdsapi

The locked seven-city CTGE core does not require ERA5; this script is included to
make the climate-data acquisition/integration audit fully reproducible.
"""
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import cdsapi
from ctge.config import load_config

CITY_BOXES={
 'Bengaluru':(13.25,77.25,12.70,77.90), 'Hyderabad':(17.70,78.15,17.05,78.80),
 'Chennai':(13.40,79.90,12.75,80.45), 'Delhi':(29.05,76.75,28.25,77.65),
 'Jaipur':(27.25,75.45,26.55,76.20), 'Lucknow':(27.15,80.55,26.55,81.30),
 'Gurugram':(28.80,76.70,28.15,77.35),
}
VARIABLES=['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_dewpoint_temperature','boundary_layer_height','surface_pressure','total_precipitation']

def main():
    cfg=load_config(); out=Path(cfg['root'])/'data'/'external'/'era5'; out.mkdir(parents=True,exist_ok=True)
    c=cdsapi.Client()
    for city in cfg['cities']:
        north,west,south,east=CITY_BOXES[city]
        target=out/f'{city.lower()}_era5_2017_2020.nc'
        req={
          'product_type':'reanalysis','variable':VARIABLES,
          'year':['2017','2018','2019','2020'],
          'month':[f'{m:02d}' for m in range(1,13)],
          'day':[f'{d:02d}' for d in range(1,32)],
          'time':[f'{h:02d}:00' for h in range(24)],
          'area':[north,west,south,east], 'format':'netcdf'
        }
        print('Downloading',city,'->',target)
        c.retrieve('reanalysis-era5-single-levels',req,str(target))
if __name__=='__main__': main()

#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from ctge.config import load_config

CITY_CENTERS={
 'Bengaluru':(12.9716,77.5946),'Hyderabad':(17.3850,78.4867),'Chennai':(13.0827,80.2707),
 'Delhi':(28.6139,77.2090),'Jaipur':(26.9124,75.7873),'Lucknow':(26.8467,80.9462),'Gurugram':(28.4595,77.0266)
}

def _find_time(ds):
    for x in ['valid_time','time']:
        if x in ds.coords:return x
    raise KeyError('ERA5 time coordinate not found')

def main():
    cfg=load_config(); root=Path(cfg['root']); era=root/'data'/'external'/'era5'; out=root/'data'/'processed'
    frames=[]
    for city in cfg['cities']:
        f=era/f'{city.lower()}_era5_2017_2020.nc'
        if not f.exists():
            print('skip missing',f); continue
        ds=xr.open_dataset(f); tname=_find_time(ds); lat,lon=CITY_CENTERS[city]
        # nearest grid point: deterministic and documented
        point=ds.sel(latitude=lat,longitude=lon,method='nearest')
        df=point.to_dataframe().reset_index(); df['Date']=pd.to_datetime(df[tname]).dt.floor('D'); df['City']=city
        rename={'t2m':'temperature_2m_k','d2m':'dewpoint_2m_k','u10':'u10_ms','v10':'v10_ms','blh':'boundary_layer_height_m','sp':'surface_pressure_pa','tp':'total_precipitation_m'}
        df=df.rename(columns={k:v for k,v in rename.items() if k in df})
        aggs={}
        for c in ['temperature_2m_k','dewpoint_2m_k','u10_ms','v10_ms','boundary_layer_height_m','surface_pressure_pa']:
            if c in df:aggs[c]='mean'
        if 'total_precipitation_m' in df:aggs['total_precipitation_m']='sum'
        daily=df.groupby(['City','Date'],as_index=False).agg(aggs)
        if {'u10_ms','v10_ms'}.issubset(daily.columns):
            daily['wind_speed_ms']=np.sqrt(daily['u10_ms']**2+daily['v10_ms']**2)
        # Relative humidity is NOT a raw ERA5 single-levels variable and was
        # previously never computed here; only raw temperature and dewpoint
        # were carried through. It is derived from 2m temperature (t2m) and
        # 2m dewpoint (d2m) via the Magnus-Tetens approximation, since the
        # manuscript (Section 3.2) explicitly requires relative humidity as
        # one of CPDM's six meteorological covariates.
        if {'temperature_2m_k','dewpoint_2m_k'}.issubset(daily.columns):
            t_c=daily['temperature_2m_k']-273.15; td_c=daily['dewpoint_2m_k']-273.15
            es_t=6.112*np.exp(17.62*t_c/(243.12+t_c)); es_td=6.112*np.exp(17.62*td_c/(243.12+td_c))
            daily['relative_humidity_pct']=np.clip(100.0*es_td/es_t,0,100)
            daily['temperature_c']=t_c
        if 'surface_pressure_pa' in daily.columns:
            daily['surface_pressure_hpa']=daily['surface_pressure_pa']/100.0
        if 'total_precipitation_m' in daily.columns:
            daily['precipitation_mm']=daily['total_precipitation_m']*1000.0
        frames.append(daily)
    if not frames: raise SystemExit('No ERA5 files found. Run 03_download_era5_cds.py first.')
    era_daily=pd.concat(frames,ignore_index=True); era_daily.to_csv(out/'era5_daily.csv',index=False)
    master=pd.read_csv(out/'ctge_base_master.csv',parse_dates=['Date'])
    merged=master.merge(era_daily,on=['City','Date'],how='left')
    merged.to_csv(out/'ctge_master_with_era5.csv',index=False)
    print('Wrote',out/'era5_daily.csv','and',out/'ctge_master_with_era5.csv')
if __name__=='__main__': main()

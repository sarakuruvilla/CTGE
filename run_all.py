#!/usr/bin/env python
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STAGES=[
 '00_validate_inputs.py','01_prepare_rohan_rao.py','02_build_intraday.py',
 '05_build_cpdm_erdm.py','07_run_seven_seed_ctge.py','08_cpcb_analysis.py',
 '10_external_comparators.py','09_statistical_tests.py','11_nested_loco.py','12_generate_tables_figures.py'
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--with-era5',action='store_true'); ap.add_argument('--skip-loco',action='store_true'); args=ap.parse_args()
    stages=STAGES.copy()
    if args.with_era5: stages[3:3]=['03_download_era5_cds.py','04_match_integrate_era5.py']
    if args.skip_loco: stages=[s for s in stages if s!='11_nested_loco.py']
    for s in stages:
        print('\n===',s,'==='); subprocess.run([sys.executable,str(ROOT/'scripts'/s)],check=True,cwd=ROOT)
if __name__=='__main__': main()

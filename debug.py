import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from meteostat import Stations
import warnings
import shutil

# --- CONFIG ---
os.environ['HERBIE_SAVE_DIR'] = r"D:\WxChallenge\data"
from herbie import Herbie

# Don't hide warnings in debug mode
warnings.filterwarnings("default")

def test_grib_support():
    print("\n[DEBUG] Checking GRIB support...")
    try:
        import cfgrib
        print("      SUCCESS: 'cfgrib' module found.")
    except ImportError:
        print("      FAIL: 'cfgrib' module NOT found.")
        print("      SOLUTION: You need to install eccodes.")
        print("      If using Conda: 'conda install -c conda-forge eccodes cfgrib'")
        print("      If using Pip: It is very hard on Windows. Try using Conda.")
        return False
    return True

def get_inputs():
    # Hardcoded for debugging to save you typing
    print("\n[DEBUG] Using hardcoded inputs for testing...")
    station_id = "KHOU"
    lat = 29.6375
    lon = -95.2825
    
    # Set a target date (Tomorrow)
    target_date = datetime.utcnow().replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=1)
    start_window = target_date
    end_window = start_window + timedelta(hours=6) # Short window for test
    
    print(f"      Target: {station_id}")
    print(f"      Window: {start_window} to {end_window}")
    return station_id, lat, lon, start_window, end_window

def process_model_sequential(model, run_date, lat, lon, fxx_list):
    print(f"\n[DEBUG] Starting {model.upper()} processing...")
    
    # HRRR is 'sfc', others 'pgrb2.0p25'
    product = 'sfc' if model == 'hrrr' else 'pgrb2.0p25'
    if model == 'nam': product = 'awphys'
    
    search_str = ":TMP:2 m" # Keep it simple for test
    
    for fxx in fxx_list:
        print(f"      -> Hour {fxx}: Initializing Herbie...")
        try:
            H = Herbie(run_date, model=model, product=product, fxx=fxx, verbose=True)
            
            print(f"      -> Hour {fxx}: Downloading/Opening xarray...")
            # We remove the try/except here to let it CRASH if it fails
            ds = H.xarray(search=search_str, verbose=True)
            
            print(f"      -> Hour {fxx}: Interpolating...")
            pt = ds.interp(latitude=lat, longitude=lon, method='linear')
            val = (pt['t2m'].values - 273.15) * 9/5 + 32
            print(f"      -> Hour {fxx}: SUCCESS. Temp = {val:.2f} F")
            
            # Clean up
            ds.close()
            
        except Exception as e:
            print(f"      -> Hour {fxx}: FAILED.")
            print(f"      ERROR DETAILS: {e}")
            # If it's an import error, stop immediately
            if "cfgrib" in str(e) or "eccodes" in str(e):
                print("\n[CRITICAL] You are missing the GRIB reader library.")
                sys.exit()

def main():
    print("--- WxChallenge DEBUGGER ---")
    
    # 1. Check Libraries
    if not test_grib_support():
        input("Press Enter to exit...")
        sys.exit()

    # 2. Setup
    station_id, lat, lon, start_window, end_window = get_inputs()
    
    # 3. Get recent run
    now_utc = datetime.utcnow()
    model_run_date = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    # If 12z isn't ready, go back to yesterday 18z
    if now_utc.hour < 16:
        model_run_date = model_run_date - timedelta(days=1)
        model_run_date = model_run_date.replace(hour=18)
        
    print(f"[DEBUG] Model Run: {model_run_date}")
    
    # 4. Test just 1 hour
    fxx_list = [12] 
    
    # 5. Run Sequentially
    process_model_sequential('gfs', model_run_date, lat, lon, fxx_list)
    
    print("\n[DEBUG] Test Complete.")

if __name__ == "__main__":
    main()
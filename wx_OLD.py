import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from meteostat import Point, Daily
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
os.environ['HERBIE_SAVE_DIR'] = r"D:\WxChallenge\data"
from herbie import Herbie

warnings.filterwarnings("ignore")

# --- USER SETTINGS ---
TRAINING_DAYS = 10  
MODELS = ['gfs', 'nam', 'hrrr'] 
MAX_WORKERS = 16      
DELETE_GRIBS = True   

GRID_INDEX_CACHE = {}

# ==========================================
# 1. INPUTS & DATABASE
# ==========================================
def get_inputs():
    print("--- WxChallenge Database Script (Gust Edition) ---")
    try:
        lat = float(input("Enter Latitude (e.g., 32.1313): "))
        lon = float(input("Enter Longitude (e.g., -81.2023): "))
        date_str = input("Enter target forecast date (YYYY-MM-DD): ")
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        return lat, lon, target_date
    except ValueError:
        print("Invalid input. Please restart.")
        sys.exit()

def get_history_filename(lat, lon):
    return f"wx_history_{lat:.4f}_{lon:.4f}.csv"

def load_history(lat, lon):
    filename = get_history_filename(lat, lon)
    if os.path.exists(filename):
        print(f"[Database] Loading existing history from {filename}...")
        df = pd.read_csv(filename, parse_dates=['date'])
        df.set_index('date', inplace=True)
        return df
    return pd.DataFrame()

def save_history(df, lat, lon):
    filename = get_history_filename(lat, lon)
    df.sort_index().to_csv(filename)
    print(f"[Database] History saved to {filename}")

# ==========================================
# 2. GET REAL OBSERVATION DATA
# ==========================================
def get_real_data(lat, lon, start_date, end_date):
    print(f"\n[1/4] Fetching real observation data...")
    location = Point(lat, lon)
    data = Daily(location, start_date, end_date)
    data = data.fetch()
    
    retries = 0
    while data.empty and retries < 5:
        retries += 1
        new_end = end_date - timedelta(days=retries)
        new_start = start_date - timedelta(days=retries)
        data = Daily(location, new_start, new_end)
        data = data.fetch()

    if data.empty: return pd.DataFrame()

    df = pd.DataFrame()
    df['obs_max_f'] = (data['tmax'] * 9/5) + 32
    df['obs_min_f'] = (data['tmin'] * 9/5) + 32
    df['obs_wspd_kt'] = data['wspd'] * 0.539957
    df['obs_prcp_in'] = data['prcp'] * 0.0393701
    
    df['obs_prcp_in'] = df['obs_prcp_in'].fillna(0.0)
    if df['obs_wspd_kt'].isnull().any():
        mean_wind = df['obs_wspd_kt'].mean()
        if pd.isna(mean_wind): mean_wind = 5.0
        df['obs_wspd_kt'] = df['obs_wspd_kt'].fillna(mean_wind)

    df = df.dropna(subset=['obs_max_f', 'obs_min_f'])
    
    conditions = []
    for index, row in df.iterrows():
        if row['obs_prcp_in'] > 0.01: conditions.append('Wet')
        else: conditions.append('Dry')
    df['condition'] = conditions
    return df

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_product_name(model_name):
    if model_name == 'gfs': return 'pgrb2.0p25'
    if model_name == 'nam': return 'awphys'
    if model_name == 'hrrr': return 'sfc'
    return 'pgrb2.0p25'

def get_nearest_point(ds, target_lat, target_lon):
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        lon_lookup = target_lon + 360 if (ds.longitude.max() > 180 and target_lon < 0) else target_lon
        return ds.sel(latitude=target_lat, longitude=lon_lookup, method='nearest')
    else:
        cache_key = (target_lat, target_lon)
        if cache_key in GRID_INDEX_CACHE:
            return ds.isel(**GRID_INDEX_CACHE[cache_key])

        lats = ds['latitude'].values
        lons = ds['longitude'].values
        lons_norm = (lons + 180) % 360 - 180
        target_lon_norm = (target_lon + 180) % 360 - 180
        
        dist = (lats - target_lat)**2 + (lons_norm - target_lon_norm)**2
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        
        dims = ds['latitude'].dims 
        indices = {dims[0]: min_idx[0], dims[1]: min_idx[1]}
        GRID_INDEX_CACHE[cache_key] = indices
        return ds.isel(**indices)

# ==========================================
# 4. DOWNLOAD & PROCESS LOGIC
# ==========================================
def download_worker(model_name, run_date, hour, search_str):
    prod = get_product_name(model_name)
    try:
        H = Herbie(run_date, model=model_name, product=prod, fxx=hour, verbose=False)
        H.download(search=search_str, verbose=False)
        return True
    except Exception:
        return False

def pre_download_data(model_name, run_date, step=3):
    hours = range(24, 49, step)
    search_inst = ":TMP:2 m|:UGRD:10 m|:VGRD:10 m"
    search_pcp = ":APCP"
    search_gust = ":GUST:surface" # NEW: Download Gusts
    
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for h in hours:
            tasks.append(executor.submit(download_worker, model_name, run_date, h, search_inst))
            tasks.append(executor.submit(download_worker, model_name, run_date, h, search_pcp))
            tasks.append(executor.submit(download_worker, model_name, run_date, h, search_gust))
        for future in as_completed(tasks):
            future.result()

def process_model_data(model_name, run_date, lat, lon, step=3):
    prod = get_product_name(model_name)
    hours_to_check = range(24, 49, step)
    
    temps, winds, gusts = [], [], []
    precip_total = 0.0
    success_count = 0

    for hour in hours_to_check:
        try:
            H = Herbie(run_date, model=model_name, product=prod, fxx=hour, verbose=False)
            
            # --- Read Temp/Wind ---
            ds_inst = H.xarray(search=":TMP:2 m|:UGRD:10 m|:VGRD:10 m", verbose=False)
            if isinstance(ds_inst, list): ds_inst = xr.merge(ds_inst, compat='override')
            
            pt = get_nearest_point(ds_inst, lat, lon)
            
            if 't2m' in pt: 
                temps.append((pt['t2m'].values - 273.15) * 9/5 + 32)
            
            if 'u10' in pt and 'v10' in pt:
                winds.append(np.sqrt(pt['u10'].values**2 + pt['v10'].values**2) * 1.94384)
            
            # --- Read Gusts (NEW) ---
            try:
                ds_gust = H.xarray(search=":GUST:surface", verbose=False)
                if isinstance(ds_gust, list): ds_gust = xr.merge(ds_gust, compat='override')
                pt_g = get_nearest_point(ds_gust, lat, lon)
                if 'gust' in pt_g:
                    gusts.append(pt_g['gust'].values * 1.94384) # m/s to knots
            except: pass

            # --- Read Precip ---
            try:
                ds_pcp = H.xarray(search=":APCP", verbose=False)
                if isinstance(ds_pcp, list): ds_pcp = xr.merge(ds_pcp, compat='override')
                pt_p = get_nearest_point(ds_pcp, lat, lon)
                
                val = 0.0
                if 'tp' in pt_p: val = pt_p['tp'].values
                elif 'apcp' in pt_p: val = pt_p['apcp'].values
                
                if hour == 48: precip_total = val * 0.0393701
            except: pass

            success_count += 1
            
            if DELETE_GRIBS:
                try:
                    local_file = H.get_localFilePath(search=":TMP:2 m|:UGRD:10 m|:VGRD:10 m")
                    if local_file.exists(): local_file.unlink()
                    local_file_p = H.get_localFilePath(search=":APCP")
                    if local_file_p.exists(): local_file_p.unlink()
                    local_file_g = H.get_localFilePath(search=":GUST:surface")
                    if local_file_g.exists(): local_file_g.unlink()
                except: pass

        except Exception: continue

    if success_count == 0 or not temps: return None, None, None, None, None
    
    max_t = np.max(temps)
    min_t = np.min(temps)
    max_w = np.max(winds) if winds else 0.0
    max_g = np.max(gusts) if gusts else 0.0
    
    return max_t, min_t, max_w, precip_total, max_g

# ==========================================
# 5. MAIN LOGIC
# ==========================================
def main():
    lat, lon, target_date = get_inputs()
    history_df = load_history(lat, lon)
    
    end_train = target_date - timedelta(days=1)
    start_train = end_train - timedelta(days=TRAINING_DAYS)
    
    obs_df = get_real_data(lat, lon, start_train, end_train)
    if obs_df.empty: 
        print("No observation data found.")
        return

    needed_dates = [d for d in obs_df.index if d not in history_df.index]
    
    print(f"\n[2/4] Database Status:")
    print(f"      Total Days Needed: {len(obs_df)}")
    print(f"      Already in DB:     {len(obs_df) - len(needed_dates)}")
    print(f"      To Download:       {len(needed_dates)}")
    
    if needed_dates:
        print(f"      (Parallel Download -> Sequential Read -> Auto-Delete)")
        new_rows = []
        for date in needed_dates:
            run_date = date - timedelta(days=1)
            print(f"      Processing new day: {run_date.strftime('%Y-%m-%d')}...")
            
            for model in MODELS: pre_download_data(model, run_date, step=3)
            
            day_valid = True
            row_data = {'date': date}
            
            for model in MODELS:
                # Note: We ignore gusts for history training to keep it simple
                mx, mn, w, p, g = process_model_data(model, run_date, lat, lon, step=3)
                if mx is None:
                    day_valid = False
                    break
                row_data[f'{model}_max'] = mx
                row_data[f'{model}_min'] = mn
                row_data[f'{model}_wspd'] = w
                row_data[f'{model}_prcp'] = p
            
            if day_valid:
                new_rows.append(row_data)
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.set_index('date', inplace=True)
            history_df = pd.concat([history_df, new_df])
            history_df = history_df[~history_df.index.duplicated(keep='last')]
            save_history(history_df, lat, lon)

    analysis_df = obs_df.join(history_df, how='inner')
    if analysis_df.empty:
        print("CRITICAL: No overlapping data between Obs and Model History.")
        return

    print("\n[3/4] Calculating Weights...")
    print(f"      Fetching {target_date.strftime('%Y-%m-%d')} raw forecast (Hourly Precision)...")
    run_date_current = target_date - timedelta(days=1)
    for model in MODELS: pre_download_data(model, run_date_current, step=1)
    
    current_forecasts = {}
    avg_prcp_forecast = 0
    
    for model in MODELS:
        mx, mn, w, p, g = process_model_data(model, run_date_current, lat, lon, step=1)
        if mx is None:
            print(f"CRITICAL: Could not find current data for {model}")
            return
        current_forecasts[model] = {'max': mx, 'min': mn, 'wspd': w, 'prcp': p, 'gust': g}
        avg_prcp_forecast += p
    
    avg_prcp_forecast /= len(MODELS)
    forecast_type = 'Wet' if avg_prcp_forecast > 0.01 else 'Dry'
    print(f"      Forecast Type: {forecast_type}")
    
    type_df = analysis_df[analysis_df['condition'] == forecast_type]
    if len(type_df) < 5:
        print(f"      Low sample size ({len(type_df)} days), using all history.")
        type_df = analysis_df

    weights = {'max': {}, 'min': {}, 'wspd': {}, 'prcp': {}}
    for param, obs_col in [('max', 'obs_max_f'), ('min', 'obs_min_f'), ('wspd', 'obs_wspd_kt'), ('prcp', 'obs_prcp_in')]:
        total_inverse_mae = 0
        model_maes = {}
        for model in MODELS:
            if f'{model}_{param}' not in type_df.columns: continue
            mae = np.mean(np.abs(type_df[f'{model}_{param}'] - type_df[obs_col]))
            if mae == 0: mae = 0.1
            model_maes[model] = mae
            total_inverse_mae += (1 / mae)
        for model in MODELS:
            if model in model_maes:
                weights[param][model] = (1 / model_maes[model]) / total_inverse_mae

    print("\n[4/4] Generating Final Consensus Forecast...")
    final_max = sum(current_forecasts[m]['max'] * weights['max'][m] for m in MODELS)
    final_min = sum(current_forecasts[m]['min'] * weights['min'][m] for m in MODELS)
    final_wspd = sum(current_forecasts[m]['wspd'] * weights['wspd'][m] for m in MODELS)
    final_prcp = sum(current_forecasts[m]['prcp'] * weights['prcp'][m] for m in MODELS)
    
    print("\n" + "="*30)
    print(f"OFFICIAL FORECAST FOR {target_date.strftime('%Y-%m-%d')}")
    print(f"Location: {lat}, {lon}")
    print("="*30)
    print(f"Max Temp:   {final_max:.1f} F")
    print(f"Min Temp:   {final_min:.1f} F")
    print(f"Max Wind:   {final_wspd:.1f} kt")
    print(f"Precip:     {final_prcp:.2f} in")
    print("-" * 30)
    print("Raw Model Forecasts (Unweighted):")
    raw_df = pd.DataFrame(current_forecasts).T
    print(raw_df[['max', 'min', 'wspd', 'gust', 'prcp']].round(2))
    print("-" * 30)
    print("Model Weights Used:")
    print(pd.DataFrame(weights))

if __name__ == "__main__":
    main()
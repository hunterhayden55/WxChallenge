import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from meteostat import Point, Hourly, Stations
import warnings
import shutil
import time
import json
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
os.environ['HERBIE_SAVE_DIR'] = r"~/Downloads/WxChallenge/data"
from herbie import Herbie

warnings.filterwarnings("ignore")

# --- USER SETTINGS ---
TRAINING_DAYS = 10  
DELETE_GRIBS = False  # <--- Files will definitely stay now
CACHE_FILE = "wx_model_cache.json"

# ==========================================
# 1. CACHE SYSTEM
# ==========================================
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)

# ==========================================
# 2. INPUTS & CYCLE DETECTION
# ==========================================
def get_inputs():
    print(f"\n--- WxChallenge Forecaster (v9.7 Force Save) ---")
    print(f"    Training Window: Last {TRAINING_DAYS} Days")
    
    print(f"    DEBUG: DELETE_GRIBS is set to {DELETE_GRIBS}")
    if not DELETE_GRIBS:
        print(f"    DEBUG: GRIB files will be PERMANENTLY saved to:\n           {os.environ['HERBIE_SAVE_DIR']}")
    
    station_input = input("Enter Station Identifier (e.g., KHOU): ").upper().strip()
    if len(station_input) == 3: station_input = "K" + station_input
    
    print(f"Looking up {station_input}...")
    stations = Stations()
    stations = stations.region('US')
    df = stations.fetch()
    station = df[df['icao'] == station_input]
    
    if station.empty:
        print(f"Station {station_input} not found. Trying global search...")
        stations = Stations()
        df = stations.fetch()
        station = df[df['icao'] == station_input]
        if station.empty:
            print("Station not found.")
            sys.exit()
        
    lat = station.iloc[0]['latitude']
    lon = station.iloc[0]['longitude']
    name = station.iloc[0]['name']
    
    print(f"Target: {name} ({lat}, {lon})")

    print("\nEnter the START date of the forecast (Day 1).")
    date_str = input("Date (YYYY-MM-DD): ")
    try:
        target_date_start = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date.")
        sys.exit()
        
    start_window = target_date_start.replace(hour=6, minute=0, second=0, microsecond=0)
    end_window = start_window + timedelta(hours=24)
    
    print(f"Forecast Window (UTC): {start_window} to {end_window}")
    
    # --- CYCLE DETECTION ---
    now_utc = datetime.utcnow()
    run_date_base = start_window - timedelta(days=1) 
    
    is_live_run = (now_utc.date() == run_date_base.date())
    
    current_cycle = 12 
    
    if is_live_run:
        if now_utc.hour >= 22 and now_utc.minute >= 30:
            current_cycle = 18
        elif now_utc.hour >= 23:
            current_cycle = 18
        else:
            current_cycle = 12
    else:
        if now_utc > (run_date_base + timedelta(hours=23)):
            current_cycle = 18
        else:
            current_cycle = 12

    print(f"Current Cycle Selection: {current_cycle}Z")
    
    return station_input, lat, lon, start_window, end_window, current_cycle

# ==========================================
# 3. HISTORY (OBSERVATIONS)
# ==========================================
def get_hourly_obs(station_id, lat, lon, end_date, days_back):
    print(f"\n[1/6] Fetching Observation History...")
    start_fetch = end_date - timedelta(days=days_back + 5)
    try:
        loc = Point(lat, lon)
        data = Hourly(loc, start_fetch, end_date)
        df = data.fetch()
        if df.empty: return pd.DataFrame()

        df['temp_f'] = (df['temp'] * 9/5) + 32
        df['wspd_kt'] = df['wspd'] * 0.539957
        df['prcp_in'] = df['prcp'] * 0.0393701
        
        df_shifted = df.shift(-6, freq='H')
        
        daily_stats = df_shifted.resample('D').agg({
            'temp_f': ['max', 'min'],
            'wspd_kt': 'max',
            'prcp_in': 'sum'
        }).dropna()
        
        daily_stats.columns = ['obs_max', 'obs_min', 'obs_wspd', 'obs_prcp']
        print(f"      Found {len(daily_stats)} days of valid history.")
        return daily_stats
    except:
        return pd.DataFrame()

# ==========================================
# 4. MODEL PROCESSING ENGINE
# ==========================================
def get_model_hours(run_date, target_start, target_end):
    diff_start = (target_start - run_date).total_seconds() / 3600
    diff_end = (target_end - run_date).total_seconds() / 3600
    start_fxx = max(0, int(diff_start))
    end_fxx = int(diff_end)
    return list(range(start_fxx, end_fxx + 1))

def robust_interp(ds, target_lat, target_lon):
    if 'gridlat_0' in ds.coords: ds = ds.rename({'gridlat_0': 'latitude', 'gridlon_0': 'longitude'})
    if 'lat' in ds.coords: ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    try:
        model_lons = ds['longitude'].values
        if model_lons.max() > 180 and target_lon < 0:
            target_lon_adj = target_lon + 360
        elif model_lons.min() < 0 and target_lon > 180:
            target_lon_adj = target_lon - 360
        else:
            target_lon_adj = target_lon
    except:
        target_lon_adj = target_lon

    try:
        return ds.interp(latitude=target_lat, longitude=target_lon_adj, method='linear')
    except:
        pass

    try:
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        dist = (lats - target_lat)**2 + (lons - target_lon_adj)**2
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        dims = ds['latitude'].dims
        selector = {dims[0]: min_idx[0], dims[1]: min_idx[1]}
        return ds.isel(**selector)
    except Exception as e:
        raise e

def safe_cleanup(H, search_str):
    """Only deletes files if DELETE_GRIBS is explicitly True"""
    if not DELETE_GRIBS:
        return 
        
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            files = H.get_localFilePath(search=search_str)
        if isinstance(files, list):
            for f in files: 
                if os.path.exists(f): os.remove(f)
        elif os.path.exists(files): os.remove(files)
    except: pass

def fetch_precip_value(H, lat, lon, search_str):
    try:
        # Explicit download first to ensure persistence
        with contextlib.redirect_stdout(io.StringIO()):
            H.download(search=search_str, verbose=False)
            ds_prcp = H.xarray(search=search_str, verbose=False)
        
        if isinstance(ds_prcp, list): ds_prcp = ds_prcp[0]
        pt_p = robust_interp(ds_prcp, lat, lon)
        
        raw_p = 0.0
        if 'tp' in pt_p: raw_p = pt_p['tp'].values
        elif 'apcp' in pt_p: raw_p = pt_p['apcp'].values
        
        ds_prcp.close()
        
        if np.isnan(raw_p): return 0.0
        return raw_p * 0.0393701 
    except:
        return 0.0

def process_model(model, run_date, lat, lon, fxx_list, verbose_prefix="", debug=False):
    product = 'sfc' if model == 'hrrr' else 'pgrb2.0p25'
    if model == 'nam': product = 'awphys'
    
    # Combined search string (Fast)
    search_str = ":TMP:2 m|:UGRD:10 m|:VGRD:10 m|:GUST:surface|:APCP:|:TP:"
    
    temps, winds, gusts = [], [], []
    total_prcp = 0.0
    prev_accum_precip = 0.0

    # --- 1. PREPARE DOWNLOAD LIST ---
    herbie_objs = []
    # Pre-fetch previous hour for precip subtraction
    if fxx_list[0] > 0:
        herbie_objs.append((fxx_list[0]-1, Herbie(run_date, model=model, product=product, fxx=fxx_list[0]-1, verbose=False)))
    
    for fxx in fxx_list:
        herbie_objs.append((fxx, Herbie(run_date, model=model, product=product, fxx=fxx, verbose=False)))

    # --- 2. SAFE PARALLEL DOWNLOAD (4 Threads) ---
    if verbose_prefix:
        print(f"{verbose_prefix} Downloading data...", end="\r")

    def _download_worker(item):
        fxx, h_obj = item
        try:
            h_obj.download(search=search_str, verbose=False)
            return fxx, h_obj
        except:
            return fxx, None

    downloaded_map = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_download_worker, herbie_objs))
        for fxx, h in results:
            downloaded_map[fxx] = h

    # --- 3. HELPER: EXTRACT VARIABLES FROM DATASET LIST ---
    def extract_vars(H_obj):
        """Handles whether Herbie returns a single Dataset or a List of Datasets"""
        try:
            ds_result = H_obj.xarray(search=search_str, verbose=False)
        except:
            return None

        # Standardize to list so we can loop
        if isinstance(ds_result, list):
            ds_list = ds_result
        else:
            ds_list = [ds_result]

        # Placeholders
        t = None
        u, v = None, None
        g = 0.0
        p = 0.0
        
        found_p = False

        for ds in ds_list:
            try:
                pt = robust_interp(ds, lat, lon)
                
                # Check for Temp
                if 't2m' in pt: 
                    t = (float(pt['t2m'].values) - 273.15) * 9/5 + 32
                
                # Check for Wind
                if 'u10' in pt: u = float(pt['u10'].values)
                if 'v10' in pt: v = float(pt['v10'].values)
                
                # Check for Gust
                if 'gust' in pt: g = float(pt['gust'].values) * 1.94384
                
                # Check for Precip (Total Precip or Accumulated Precip)
                if 'tp' in pt: 
                    p = float(pt['tp'].values)
                    found_p = True
                elif 'apcp' in pt: 
                    p = float(pt['apcp'].values)
                    found_p = True
                    
                ds.close()
            except:
                pass
        
        # Calculate Wind Speed Magnitude if we found U and V
        w = 0.0
        if u is not None and v is not None:
            w = np.sqrt(u**2 + v**2) * 1.94384
            
        return {'t': t, 'w': w, 'g': g, 'p': p, 'found_p': found_p}

    # --- 4. GET BASELINE PRECIP ---
    start_fxx = fxx_list[0]
    if start_fxx > 0 and (start_fxx - 1) in downloaded_map:
        data = extract_vars(downloaded_map[start_fxx - 1])
        if data and data['found_p']:
            prev_accum_precip = data['p']

    # --- 5. PROCESS FORECAST HOURS ---
    for fxx in fxx_list:
        if verbose_prefix and not debug:
            print(f"{verbose_prefix} Processing Hour {fxx}...", end="\r")
            
        H = downloaded_map.get(fxx)
        if H is None:
            continue

        data = extract_vars(H)
        
        if data is None or data['t'] is None:
            if debug: print(f"      [MISS] Hour {fxx} data missing.")
            continue

        # Precip Calculation
        curr_accum_mm = data['p']
        curr_accum_in = curr_accum_mm * 0.0393701
        prev_accum_in = prev_accum_precip * 0.0393701
        
        delta = curr_accum_in - prev_accum_in
        if delta < 0: delta = curr_accum_in # Bucket reset
        
        hourly_p = delta
        prev_accum_precip = curr_accum_mm # Update baseline
        
        temps.append(data['t'])
        winds.append(data['w'])
        gusts.append(data['g'])
        total_prcp += hourly_p
        
        if debug:
            print(f"      [{model.upper()}] Hour {fxx}: T={data['t']:.1f} W={data['w']:.1f} P={hourly_p:.3f}")

    if not temps: return None
    
    return {
        'max': float(np.max(temps)),
        'min': float(np.min(temps)),
        'wspd': float(np.max(winds)),
        'gust': float(np.max(gusts)),
        'prcp': float(total_prcp)
    }

# ==========================================
# 5. TRAINING LOOP (DUAL CACHE)
# ==========================================
def train_models(station_id, lat, lon, target_date, history_df, current_cycle):
    print(f"\n[2/6] Training Models (Backtesting last {TRAINING_DAYS} days)...")
    print(f"      Strategy: Caching BOTH 12Z and 18Z. Weighting based on {current_cycle}Z.")
    
    cache = load_cache()
    if station_id not in cache: cache[station_id] = {}
    
    model_errors = {'gfs': [], 'nam': [], 'hrrr': []}
    
    for i in range(1, TRAINING_DAYS + 1):
        past_target_date = target_date - timedelta(days=i)
        lookup_date = past_target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_key = lookup_date.strftime('%Y-%m-%d')
        
        if lookup_date not in history_df.index:
            print(f"      Skipping {date_key} (No Obs found)")
            continue
            
        obs = history_df.loc[lookup_date]
        
        # --- DUAL CYCLE PROCESSING (12Z AND 18Z) ---
        for cycle_check in [12, 18]:
            cycle_str = str(cycle_check)
            
            # 1. Check Cache
            if date_key in cache[station_id] and cycle_str in cache[station_id][date_key]:
                if cycle_check == current_cycle:
                    print(f"      Using Cached {cycle_check}Z Data for {date_key}")
                day_results = cache[station_id][date_key][cycle_str]
            
            # 2. Download if missing
            else:
                past_run_date = past_target_date - timedelta(days=1)
                past_run_date = past_run_date.replace(hour=cycle_check, minute=0, second=0, microsecond=0)
                
                start_win = past_target_date.replace(hour=6)
                end_win = start_win + timedelta(hours=24)
                fxx_list = get_model_hours(past_run_date, start_win, end_win)
                
                print(f"\n      >>> DOWNLOADING TRAINING DATA: {date_key} | Cycle: {cycle_check}Z <<<")
                
                day_results = {}
                for m in ['gfs', 'nam', 'hrrr']:
                    res = process_model(m, past_run_date, lat, lon, fxx_list, verbose_prefix=f"        [{m.upper()}]", debug=True)
                    if res: day_results[m] = res
                
                if date_key not in cache[station_id]: cache[station_id][date_key] = {}
                cache[station_id][date_key][cycle_str] = day_results
                save_cache(cache)

            # 3. Calculate Errors (ONLY for the Current Cycle)
            if cycle_check == current_cycle:
                for m in ['gfs', 'nam', 'hrrr']:
                    if m in day_results:
                        res = day_results[m]
                        err_max = abs(res['max'] - obs['obs_max'])
                        err_min = abs(res['min'] - obs['obs_min'])
                        err_wspd = abs(res['wspd'] - obs['obs_wspd'])
                        err_prcp = abs(res['prcp'] - obs['obs_prcp'])
                        
                        model_errors[m].append({
                            'max': err_max, 'min': err_min, 'wspd': err_wspd, 'prcp': err_prcp
                        })
    
    print("\n      Training Complete.")
    return model_errors

def calculate_weights(model_errors):
    print(f"\n[3/6] Calculating Weights based on MAE (Mean Absolute Error)...")
    
    weights = {'max': {}, 'min': {}, 'wspd': {}, 'prcp': {}}
    stats = {}
    
    for param in ['max', 'min', 'wspd', 'prcp']:
        total_inv_mae = 0
        maes = {}
        
        for m in ['gfs', 'nam', 'hrrr']:
            errors = [e[param] for e in model_errors[m]]
            if not errors:
                mae = 99.9 
            else:
                mae = np.mean(errors)
                if mae < 0.01: mae = 0.01 
            
            maes[m] = mae
            total_inv_mae += (1 / mae)
            
        for m in ['gfs', 'nam', 'hrrr']:
            w = (1 / maes[m]) / total_inv_mae
            weights[param][m] = w
            
        stats[param] = maes
        
    return weights, stats

# ==========================================
# 6. MAIN
# ==========================================
def main():
    station_id, lat, lon, start_window, end_window, current_cycle = get_inputs()
    
    # 1. History
    history_df = get_hourly_obs(station_id, lat, lon, start_window, days_back=TRAINING_DAYS)
    
    # 2. Train (With Dual Cache)
    model_errors = train_models(station_id, lat, lon, start_window, history_df, current_cycle)
    
    # 3. Weights
    weights, maes = calculate_weights(model_errors)
    
    # Print Bias Report
    print("\n" + "="*65)
    print(f"HISTORICAL BIAS REPORT ({current_cycle}Z Runs - MAE - Lower is Better)")
    print("="*65)
    print(f"{'MODEL':<8} {'MAX T':<12} {'MIN T':<12} {'WIND':<12} {'PRCP':<12}")
    print("-" * 65)
    for m in ['gfs', 'nam', 'hrrr']:
        print(f"{m.upper():<8} "
              f"{maes['max'][m]:<5.2f} ({int(weights['max'][m]*100)}%)   "
              f"{maes['min'][m]:<5.2f} ({int(weights['min'][m]*100)}%)   "
              f"{maes['wspd'][m]:<5.2f} ({int(weights['wspd'][m]*100)}%)   "
              f"{maes['prcp'][m]:<5.2f} ({int(weights['prcp'][m]*100)}%)")
    print("="*65)

    # 4. Current Forecast
    print(f"\n[4/6] Generating Current Forecast...")
    
    run_date_base = start_window - timedelta(days=1)
    model_run_date = run_date_base.replace(hour=current_cycle, minute=0, second=0, microsecond=0)
    
    print(f"      Using Run: {model_run_date.strftime('%Y-%m-%d %HZ')}")
    fxx_list = get_model_hours(model_run_date, start_window, end_window)
    
    forecasts = {}
    for m in ['gfs', 'nam', 'hrrr']:
        print(f"\n      >>> PROCESSING {m.upper()} <<<")
        res = process_model(m, model_run_date, lat, lon, fxx_list, verbose_prefix=f"      [{m.upper()}]", debug=True)
        if res: forecasts[m] = res
        else: print(f"      [{m.upper()}] Failed.")

    # 5. Apply Weights
    print(f"\n\n[5/6] Applying Weighted Consensus...")
    
    final_max = 0
    final_min = 0
    final_wspd = 0
    final_prcp = 0 
    
    valid_models = [m for m in forecasts.keys()]
    
    for param in ['max', 'min', 'wspd', 'prcp']:
        total_w = sum(weights[param][m] for m in valid_models)
        for m in valid_models:
            w = weights[param][m] / total_w
            if param == 'max': final_max += forecasts[m]['max'] * w
            if param == 'min': final_min += forecasts[m]['min'] * w
            if param == 'wspd': final_wspd += forecasts[m]['wspd'] * w
            if param == 'prcp': final_prcp += forecasts[m]['prcp'] * w
            
    # 6. Output
    print("\n" + "="*60)
    print(f"OFFICIAL GUIDANCE: {station_id}")
    print(f"Valid: {start_window.strftime('%d/%H')}Z to {end_window.strftime('%d/%H')}Z")
    print(f"Run Cycle Used: {current_cycle}Z")
    print("="*60)
    print(f"{'MODEL':<10} {'MAX':<8} {'MIN':<8} {'WIND':<8} {'GUST':<8} {'PRCP':<8}")
    print("-" * 60)
    for m in valid_models:
        d = forecasts[m]
        print(f"{m.upper():<10} {d['max']:<8.1f} {d['min']:<8.1f} {d['wspd']:<8.1f} {d['gust']:<8.1f} {d['prcp']:<8.2f}")
    print("-" * 60)
    print(f"{'WEIGHTED':<10} {final_max:<8.1f} {final_min:<8.1f} {final_wspd:<8.1f} {'--':<8} {final_prcp:<8.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
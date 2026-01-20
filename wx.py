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

# ==========================================
# CONFIGURATION
# ==========================================
os.environ['HERBIE_SAVE_DIR'] = r"D:\WxChallenge\data"
from herbie import Herbie

warnings.filterwarnings("ignore")

# --- USER SETTINGS ---
TRAINING_DAYS = 10  
DELETE_GRIBS = True
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
# 2. INPUTS
# ==========================================
def get_inputs():
    print(f"\n--- WxChallenge Forecaster (v8.0 Caching Edition) ---")
    print(f"    Training Window: Last {TRAINING_DAYS} Days")
    
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
    return station_input, lat, lon, start_window, end_window

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

def process_model(model, run_date, lat, lon, fxx_list, verbose_prefix=""):
    product = 'sfc' if model == 'hrrr' else 'pgrb2.0p25'
    if model == 'nam': product = 'awphys'
    
    search_main = ":TMP:2 m|:UGRD:10 m|:VGRD:10 m|:GUST:surface"
    search_prcp = ":APCP:.*:0-1" 
    
    temps, winds, gusts = [], [], []
    total_prcp = 0.0

    for fxx in fxx_list:
        if verbose_prefix:
            print(f"{verbose_prefix} Hour {fxx}...", end="\r")
        
        try:
            H = Herbie(run_date, model=model, product=product, fxx=fxx, verbose=False)
            ds_main = H.xarray(search=search_main, verbose=False)
            if isinstance(ds_main, list): ds_list = ds_main
            else: ds_list = [ds_main]

            t_val, u_val, v_val, g_val = None, None, None, 0.0

            for ds in ds_list:
                pt = robust_interp(ds, lat, lon)
                if 't2m' in pt: t_val = (pt['t2m'].values - 273.15) * 9/5 + 32
                if 'u10' in pt: u_val = pt['u10'].values
                if 'v10' in pt: v_val = pt['v10'].values
                if 'gust' in pt: g_val = pt['gust'].values * 1.94384
                ds.close()

            w_val = 0.0
            if u_val is not None and v_val is not None:
                w_val = np.sqrt(u_val**2 + v_val**2) * 1.94384
            
            if t_val is not None:
                temps.append(t_val)
                winds.append(w_val)
                gusts.append(g_val)
            
            if DELETE_GRIBS:
                try:
                    files = H.get_localFilePath(search=search_main)
                    if isinstance(files, list):
                        for f in files: 
                            if os.path.exists(f): os.remove(f)
                    elif os.path.exists(files): os.remove(files)
                except: pass

        except: pass

        try:
            ds_prcp = H.xarray(search=search_prcp, verbose=False)
            if isinstance(ds_prcp, list): ds_prcp = ds_prcp[0]
            pt_p = robust_interp(ds_prcp, lat, lon)
            p_val = 0.0
            if 'tp' in pt_p: p_val = pt_p['tp'].values
            elif 'apcp' in pt_p: p_val = pt_p['apcp'].values
            if not np.isnan(p_val): total_prcp += (p_val * 0.0393701)
            ds_prcp.close()
            if DELETE_GRIBS:
                try:
                    f_p = H.get_localFilePath(search=search_prcp)
                    if os.path.exists(f_p): os.remove(f_p)
                except: pass
        except: pass

    if not temps: return None
    
    return {
        'max': float(np.max(temps)),
        'min': float(np.min(temps)),
        'wspd': float(np.max(winds)),
        'gust': float(np.max(gusts)),
        'prcp': float(total_prcp)
    }

# ==========================================
# 5. TRAINING LOOP (WITH CACHE)
# ==========================================
def train_models(station_id, lat, lon, target_date, history_df):
    print(f"\n[2/6] Training Models (Backtesting last {TRAINING_DAYS} days)...")
    
    # Load Cache
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
        
        # Check Cache First
        if date_key in cache[station_id]:
            print(f"      Using Cached Data for {date_key}")
            day_results = cache[station_id][date_key]
        else:
            # Download if not in cache
            past_run_date = past_target_date - timedelta(days=1)
            past_run_date = past_run_date.replace(hour=12, minute=0, second=0, microsecond=0)
            
            start_win = past_target_date.replace(hour=6)
            end_win = start_win + timedelta(hours=24)
            fxx_list = get_model_hours(past_run_date, start_win, end_win)
            
            print(f"      Downloading {date_key} (Run: {past_run_date.strftime('%d/12Z')})...")
            
            day_results = {}
            for m in ['gfs', 'nam', 'hrrr']:
                res = process_model(m, past_run_date, lat, lon, fxx_list, verbose_prefix=f"        [{m.upper()}]")
                print(f"        [{m.upper()}] Done.                    ", end="\r")
                if res: day_results[m] = res
            
            # Save to cache immediately
            cache[station_id][date_key] = day_results
            save_cache(cache)

        # Calculate Errors
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
    station_id, lat, lon, start_window, end_window = get_inputs()
    
    # 1. History
    history_df = get_hourly_obs(station_id, lat, lon, start_window, days_back=TRAINING_DAYS)
    
    # 2. Train (With Cache)
    model_errors = train_models(station_id, lat, lon, start_window, history_df)
    
    # 3. Weights
    weights, maes = calculate_weights(model_errors)
    
    # Print Bias Report
    print("\n" + "="*65)
    print("HISTORICAL BIAS REPORT (MAE - Lower is Better)")
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
    now_utc = datetime.utcnow()
    run_candidate = now_utc - timedelta(hours=5)
    hour_block = (run_candidate.hour // 6) * 6
    model_run_date = run_candidate.replace(hour=hour_block, minute=0, second=0, microsecond=0)
    
    print(f"      Using Run: {model_run_date.strftime('%Y-%m-%d %HZ')}")
    fxx_list = get_model_hours(model_run_date, start_window, end_window)
    
    forecasts = {}
    for m in ['gfs', 'nam', 'hrrr']:
        res = process_model(m, model_run_date, lat, lon, fxx_list, verbose_prefix=f"      [{m.upper()}]")
        print(f"      [{m.upper()}] Done.                    ", end="\r")
        if res: forecasts[m] = res
        else: print(f"\n      [{m.upper()}] Failed.")

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
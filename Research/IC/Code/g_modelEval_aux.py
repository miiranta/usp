import os
import pandas as pd
import requests
from datetime import datetime
from scipy.interpolate import CubicSpline
import numpy as np

INTERPOLATE_INFLATION_TO_DAILY = False # ARIMA does not like it

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "optimized_results")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "eval_results")

INFLATION_FILE = os.path.join(OUTPUT_FOLDER, "ipca_data.csv")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def download_ipca_data():
    if os.path.exists(INFLATION_FILE):
        print(f"IPCA data already downloaded at: {INFLATION_FILE}")
        df = pd.read_csv(INFLATION_FILE)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} records from existing file.")
        return df
    
    print("Downloading IPCA monthly data from Banco Central API...")
    
    try:
        # BCB API for IPCA monthly - Series 433
        url_ipca = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
        
        response = requests.get(url_ipca, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        records = []
        for item in data:
            records.append({
                'date': item['data'],
                'ipca_monthly': float(item['valor'])
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df = df.sort_values('date').reset_index(drop=True)
        
        # Interpolate to daily if needed (cubic spline)
        if INTERPOLATE_INFLATION_TO_DAILY:
            print("Interpolating IPCA data to daily frequency using CubicSpline...")
            
            # Convert dates to numeric values (days since first date)
            dates = df['date'].values
            values = df['ipca_monthly'].values
            
            # Create date range for daily interpolation
            start_date = df['date'].min()
            end_date = df['date'].max()
            daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Convert to numeric for interpolation (days since start)
            x_original = (dates - dates[0]).astype('timedelta64[D]').astype(int)
            x_daily = (daily_dates - dates[0]).to_numpy().astype('timedelta64[D]').astype(int)
            
            # Apply cubic spline interpolation
            cs = CubicSpline(x_original, values)
            daily_values = cs(x_daily)
            
            # Create new dataframe with daily data
            df = pd.DataFrame({
                'date': daily_dates,
                'ipca_monthly': daily_values
            })
        
        # Save to CSV
        df.to_csv(INFLATION_FILE, index=False)
        print(f"[OK] IPCA data downloaded successfully! Saved to: {INFLATION_FILE}")
        print(f"[OK] Downloaded {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading IPCA data: {e}")
        return None

def create_evaluation_csvs(rank):
    # Load IPCA data
    print("Loading IPCA data...")
    ipca_data = download_ipca_data()
    
    # Read optimization results to get the best result
    opt_results_path = os.path.join(INPUT_FOLDER, 'all_optimization_results.csv')
    print(f"\nReading optimization results from {opt_results_path}...")
    opt_results = pd.read_csv(opt_results_path, sep='|')
    
    # Get the first row (best result)
    best_result = opt_results.iloc[rank]
    run_title = best_result['Run_Title']
    print(f"Best optimization result: {run_title}")
    
    # Extract model names part (before --eq)
    model_names = run_title.split('--')[0]
    
    # Interpolated file path
    interpolated_file = f"model_{model_names}_daily_averages_interpolated.csv"
    interpolated_path = os.path.join(INPUT_FOLDER, 'interpolated', interpolated_file)
    
    # Optimized file path
    optimized_file = f"{run_title}_optimized.csv"
    optimized_path = os.path.join(INPUT_FOLDER, 'optimized', optimized_file)
    
    print(f"\nLoading interpolated data from: {interpolated_file}")
    interpolated_data = pd.read_csv(interpolated_path, sep='|')
    
    print(f"Loading optimized data from: {optimized_file}")
    optimized_data = pd.read_csv(optimized_path, sep='|')
    
    # Rename columns for consistency
    interpolated_data.columns = ['date', 'sentiment']
    optimized_data.columns = ['date', 'sentiment']
    
    # Convert dates to datetime
    ipca_data['date'] = pd.to_datetime(ipca_data['date'], format='%d/%m/%Y')
    interpolated_data['date'] = pd.to_datetime(interpolated_data['date'], format='%d/%m/%Y')
    optimized_data['date'] = pd.to_datetime(optimized_data['date'], format='%d/%m/%Y')
    
    # Rename IPCA column
    ipca_data = ipca_data.rename(columns={'ipca_monthly': 'inflation'})
    
    # 3. Create optimized CSV
    print("\nCreating optimized CSV...")
    optimized_merged = pd.merge(ipca_data, optimized_data, on='date', how='inner')
    optimized_merged = optimized_merged[['date', 'inflation', 'sentiment']].dropna()
    
    # 2. Create interpolated CSV
    print("\nCreating interpolated CSV...")
    interpolated_merged = pd.merge(ipca_data, interpolated_data, on='date', how='inner')
    interpolated_merged = interpolated_merged[['date', 'inflation', 'sentiment']].dropna()
    
    # Find common dates across all datasets (use minimum row count)
    print("\nFinding common dates across all datasets...")
    common_dates = set(optimized_merged['date']) & set(interpolated_merged['date'])
    print(f"Common dates found: {len(common_dates)}")
    
    # Filter all datasets to only include common dates
    optimized_merged = optimized_merged[optimized_merged['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    interpolated_merged = interpolated_merged[interpolated_merged['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    
    # Save optimized CSV
    optimized_output = os.path.join(OUTPUT_FOLDER, 'sentiment_corrected.csv')
    optimized_merged.to_csv(optimized_output, sep='|', index=False)
    print(f"Saved optimized CSV with {len(optimized_merged)} rows to: {optimized_output}")
    
    # Save interpolated CSV
    interpolated_output = os.path.join(OUTPUT_FOLDER, 'sentiment_not_corrected.csv')
    interpolated_merged.to_csv(interpolated_output, sep='|', index=False)
    print(f"Saved interpolated CSV with {len(interpolated_merged)} rows to: {interpolated_output}")
    
    # 1. Create baseline CSV (sentiment = 0) - same dates as other datasets
    print("\nCreating baseline CSV...")
    baseline_df = interpolated_merged[['date', 'inflation']].copy()
    baseline_df['sentiment'] = 0.0
    baseline_output = os.path.join(OUTPUT_FOLDER, 'sentiment_baseline.csv')
    baseline_df.to_csv(baseline_output, sep='|', index=False)
    print(f"Saved baseline CSV with {len(baseline_df)} rows to: {baseline_output}")
    
    print("\n[OK] All 3 evaluation CSVs created successfully!")
    return baseline_df, interpolated_merged, optimized_merged, run_title
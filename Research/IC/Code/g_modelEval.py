import os
import pandas as pd
import requests
from datetime import datetime

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "optimized_results")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "eval_results")

INFLATION_FILE = os.path.join(OUTPUT_FOLDER, "ipca_daily_data.csv")

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
        
        # Save to CSV
        df.to_csv(INFLATION_FILE, index=False)
        print(f"✓ IPCA data downloaded successfully! Saved to: {INFLATION_FILE}")
        print(f"✓ Downloaded {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading IPCA data: {e}")
        return None

# Download IPCA data
ipca_data = download_ipca_data()


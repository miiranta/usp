import os
import requests
import time
import re
from urllib.parse import urlparse
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "atas")

START_URL = 'https://www.bcb.gov.br/publicacoes/atascopom/cronologicos'
KNOWN_URLS = []

def get_download_links():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print('Loading index:', START_URL)
        page.goto(START_URL, timeout=60000)
        page.wait_for_load_state('networkidle', timeout=60000)

        hrefs = page.eval_on_selector_all('a', 'els => els.map(e => e.href)')

        meetings = []
        seen = set()
        for h in hrefs:
            if not h:
                continue
            m = re.search(r"/publicacoes/atascopom/(\d{8})", h)
            if m:
                date = m.group(1)
                url = urljoin('https://www.bcb.gov.br', f'/publicacoes/atascopom/{date}')
                if url not in seen:
                    meetings.append(url)
                    seen.add(url)

        print('Found', len(meetings), 'meeting pages')

        for i, murl in enumerate(meetings, start=1):
            try:
                print(f'[{i}/{len(meetings)}] Loading', murl)
                page.goto(murl, timeout=60000)
                page.wait_for_load_state('networkidle', timeout=60000)
                hrefs2 = page.eval_on_selector_all('a', 'els => els.map(e => e.href)')
                pdf_link = None
                for hh in hrefs2:
                    if hh and '/content/copom/atascopom/' in hh and hh.lower().endswith('.pdf'):
                        pdf_link = hh
                        break
                if pdf_link:
                    print('  PDF:', pdf_link)
                    KNOWN_URLS.append(pdf_link)
                else:
                    print('  No PDF found on', murl)
            except Exception as e:
                print('  Error loading', murl, e)
            time.sleep(0.2)

        browser.close()

    print(KNOWN_URLS)
    print('\nDone.',)

def download_pdfs():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")
    
    total_files = len(KNOWN_URLS)
    
    print(f"Starting download of {total_files} PDF files...")
    
    for i, url in enumerate(KNOWN_URLS, 1):
        try:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            
            if os.path.exists(filepath):
                print(f"[{i}/{total_files}] Skipping {filename} (already exists)")
                continue
            
            print(f"[{i}/{total_files}] Downloading {filename}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"[{i}/{total_files}] ✓ Downloaded {filename}")
            
        except Exception as e:
            print(f"[{i}/{total_files}] ✗ Failed to download {filename}: {e}")
    
    print(f"\nDownloads completed!")

if __name__ == "__main__":
    get_download_links()
    download_pdfs()

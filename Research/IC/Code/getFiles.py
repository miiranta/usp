import os
import re
import time
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, 'atas')

START_URL = 'https://www.bcb.gov.br/publicacoes/atascopom/cronologicos'
TIMEOUT = 60

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def collect_meeting_urls(driver):
    meetings = {}
    
    links = driver.find_elements(By.TAG_NAME, 'a')
    for link in links:
        href = link.get_attribute('href')
        if not href:
            continue
        
        m = re.search(r'/publicacoes/atascopom/(\d{8})', href)
        if m:
            date = m.group(1)  # DDMMYYYY
            if href.startswith('http'):
                url = href.split('?')[0]
            else:
                url = urljoin('https://www.bcb.gov.br', f'/publicacoes/atascopom/{date}')
            meetings[date] = url
        
    print(f'Collected {len(meetings)} meeting URLs.')
    return meetings

def download_meeting_page(driver, wait, url, out_path):
    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Data de publicação')]")))
        except:
            pass
        
        html = driver.page_source
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return len(html)
    except Exception as e:
        raise e

def main():
    chrome_options = Options()
    #chrome_options.add_argument('--headless') 
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    wait = WebDriverWait(driver, TIMEOUT)
    
    try:
        print('Loading index:', START_URL)
        driver.get(START_URL)
        
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(8)

        meetings = collect_meeting_urls(driver)
        dates = sorted(meetings.keys(), reverse=True)

        for i, date in enumerate(dates, start=1):
            url = meetings[date]
            out_path = os.path.join(OUTPUT_FOLDER, f'{date}.html')
            if os.path.exists(out_path):
                print(f'[{i}/{len(dates)}] Skipping {date} (already downloaded)')
                continue
            
            try:
                print(f'[{i}/{len(dates)}] Loading {url}')
                download_meeting_page(driver, wait, url, out_path)
            except Exception as e:
                print(f'Error loading {url}:', e)
            time.sleep(0.25)
    
    finally:
        driver.quit()
    
    print('Done.')

if __name__ == '__main__':
    main()

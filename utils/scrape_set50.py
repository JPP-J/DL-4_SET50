from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import os
from datetime import datetime as dt

def scrape_data(url="https://www.set.or.th/en/market/index/set50", save_to_csv=False):
    """
    Scrape SET50 stock constituents from the SET website.
    Returns a DataFrame with stock information.
    """
    # Set up Chrome options

    options = Options()
    options.binary_location = "/usr/bin/google-chrome"
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(url)

    # Wait for rows to load
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.table-responsive table tbody tr"))
    )

    rows = driver.find_elements(By.CSS_SELECTOR, "div.table-responsive table tbody tr")

    names = []
    opens = []
    highs = []
    lows = []
    lates = []
    changes = []
    percent_changes = []
    bids = []
    offers = []
    volumes = []
    values = []


    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) >= 2 and all(col.text.strip() for col in cols[:10]):
            names.append(cols[0].text.strip())
            opens.append(cols[1].text.strip())
            highs.append(cols[2].text.strip())
            lows.append(cols[3].text.strip())
            lates.append(cols[4].text.strip())
            changes.append(cols[5].text.strip())
            percent_changes.append(cols[6].text.strip())
            bids.append(cols[7].text.strip())
            offers.append(cols[8].text.strip())
            volumes.append(cols[9].text.strip())
            values.append(cols[10].text.strip())

    driver.quit()

    df = pd.DataFrame({'Name': names, 
                    'Open': opens, 
                    'High': highs,
                    'Low': lows,
                    'Last': lates,
                    'Change': changes,
                    'Change %': percent_changes,
                    'Bid': bids,
                    'Offer': offers,
                    'Volume': volumes,
                    'Value(baht)': values})

    print(f"SET50 Constituents at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}: ")
    print(df)



    # Save if needed
    if save_to_csv:
        # Ensure the 'data' directory exists
        os.makedirs("data", exist_ok=True)

        # Save DataFrame to CSV
        save_path = "data/set50_constituents.csv"
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")

    return df

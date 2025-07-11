import requests
from bs4 import BeautifulSoup

def scrape_set50_1():
    """
    Scrape SET50 stock symbols from the SET website.
    Returns a list of stock symbols.
    """
    url = "https://www.set.or.th/en/market/index/set50"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Example: You’ll need to inspect the table class for current SET50
    # This needs tailoring to current HTML structure

# ==================================================================
from selenium import webdriver
from bs4 import BeautifulSoup
import time

def scrape_set50_2():
    """
    Scrape SET50 stock symbols using Selenium.
    Returns a BeautifulSoup object containing the page source.
    """
    # Initialize Selenium WebDriver (make sure you have the correct driver installed)

    driver = webdriver.Chrome()
    driver.get("https://www.set.or.th/en/market/index/set50")

    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Parse table for stock symbols
    driver.quit()

    return soup

if __name__ == "__main__":
    # Example usage
    soup = scrape_set50_2()

    print(soup.prettify()[:50])  # Print first 1000 characters of the soup object
    

import yfinance as yf
import pandas as pd
import os 

def download_stock_data(ticker, start_date, end_date, interval='1d', save_to_csv=False):
    """
    Downloads stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol e.g., PTT.BK, AOT.BK.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    interval (str): Data interval ('1d', '1wk', '1mo', etc.).

    Returns:
    DataFrame: Stock data as a pandas DataFrame.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    
    if save_to_csv:
        os.makedirs('data', exist_ok=True)

        filename = f"data/{ticker.replace('.', '_')}.csv"
        df.to_csv(filename)

        print(f"Data saved to {filename}")
    
    return df




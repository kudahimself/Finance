import pandas as pd
import requests
from typing import List
import numpy as np
import pandas_ta
import pandas_datareader.data as web
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import yfinance as yf
import time
import io
import zipfile
import re
import os


def download_sp500_tickers(save_local: bool) -> List[str]:
    """
    Scrapes the current list of S&P 500 constituent stock tickers from Wikipedia.
    
    Uses requests with a User-Agent to prevent HTTP 403 Forbidden errors, 
    then uses pandas.read_html() to parse the HTML table.

    Returns:
        A list of strings containing the S&P 500 company tickers.
        Returns an empty list [] if fetching or parsing fails.
    """
    WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Define a standard User-Agent header to avoid 403 Forbidden errors
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Attempting to fetch HTML content from: {WIKI_URL}")
        
        # 1. Fetch the HTML content using requests with a User-Agent
        response = requests.get(WIKI_URL, headers=HEADERS, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors (like 403)
        tables = pd.read_html(response.text)
        [print(table) for table in tables]
        sp500_table = tables[0]
        
        if 'Symbol' in sp500_table.columns:
            # 4. Extract and clean the tickers
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean up potential odd entries (like leading/trailing whitespace or footnotes)
            tickers = [ticker.strip().replace('.', '-') for ticker in tickers if isinstance(ticker, str)]
            tickers.append('SPY')
            
            print(f"Successfully retrieved {len(tickers)} tickers.")
            if save_local:
                pd.DataFrame(tickers, columns=['Ticker']).to_csv("data/sp500_tickers.csv", index=False)
                print("Saved S&P 500 tickers to 'data/sp500_tickers.csv'")
            return tickers
        else:
            print("Error: Could not find the 'Symbol' column in the expected table structure.")
            return []
            
    except requests.exceptions.HTTPError as e:
        # Catch 4XX errors specifically
        print(f"Error: HTTP Status {e.response.status_code}. The server rejected the request. Try running again.")
        return []
    except Exception as e:
        print(f"An error occurred while scraping the Wikipedia page: {e}")
        print("This may be due to a network error, a change in the Wikipedia page structure, or missing dependencies.")
        return []


def get_sp500_tickers() -> List[str]:
    """
    Reads the locally saved CSV file containing S&P 500 tickers.

    Returns:
        A list of strings containing the S&P 500 company tickers.
    """
    try:
        df = pd.read_csv("data/sp500_tickers.csv")
        tickers = df['Ticker'].tolist()
        print(f"Loaded {len(tickers)} tickers from local file.")
        return tickers
    except FileNotFoundError:
        print("Error: 'data/sp500_tickers.csv' not found. Please run get_sp500_tickers(save_local=True) first.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the local tickers file: {e}")
        return []

def download_sp500_data(sp500_list: List[str], save_local: bool) -> pd.DataFrame:
    
    # Download historical data for all tickers from 2010-01-04
    data = yf.download(tickers=sp500_list, start='2010-01-04', end='2025-11-01', auto_adjust=False, group_by='ticker', threads=True)

    if not data.empty:
        print('Data fetched successfully for S&P 500 tickers!')
        df = data
        # Clean column names in the MultiIndex
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, col.strip().lower().replace(' ', '_')) for ticker, col in df.columns]
        )
    else:
        print('Failed to fetch data for S&P 500 tickers.')
        df = None

    if save_local and df is not None:
        df.to_csv("data/sp500_data.csv")

    df.columns = df.columns.swaplevel(0, 1)
    df = df.sort_index(axis=1, level=0)
    df = df.stack()
    
    return df

def get_sp500_data(start_date: str = None, end_date: str = None, tickers: List[str] = None) -> pd.DataFrame:
    """
    Reads the locally saved CSV file containing S&P 500 historical data.

    Returns:
        A pandas DataFrame with MultiIndex (Date, Ticker) containing historical stock data.
    """
    try:
        df = pd.read_csv("data/sp500_data.csv", header=[0,1], index_col=0, parse_dates=True)
        df.columns = df.columns.swaplevel(0, 1)
        df = df.sort_index(axis=1, level=0)
        df = df.stack()
        df.index.names = ['date', 'ticker']
        print("Loaded S&P 500 historical data from 'data/sp500_data.csv'")
        if (start_date is not None) and (end_date is not None):
            print('Filtering data between', start_date, 'and', end_date)
            df = df.loc[(df.index.get_level_values('date') >= start_date) & 
                        (df.index.get_level_values('date') <= end_date)]
        if tickers is not None:
            print('Filtering data for specified tickers')
            df = df.loc[df.index.get_level_values('ticker').isin(tickers)]
        return df
    except FileNotFoundError:
        print("Error: 'data/sp500_data.csv' not found. Please run download_sp500_data() first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the local S&P 500 data file: {e}")
        return pd.DataFrame()


def download_famafrench_data(save_local: bool) -> pd.DataFrame:
    """
    Downloads the Fama-French 5-factor model data from the Ken French website.

    Parameters:
        save_local (bool): If True, saves the data to a local CSV file.

    Returns:
        A pandas DataFrame with the factor data, indexed by date.
    """
    
    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                                'famafrench',
                                start='2016')[0].drop('RF', axis=1)
    print(factor_data.tail())
    
    # Clean column names: strip whitespace, convert to lowercase, replace spaces with underscores
    factor_data.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in factor_data.columns]

    factor_data.index = factor_data.index.to_timestamp()

    factor_data = factor_data.resample('ME').last().div(100)
    factor_data.index.name =  'date'
    if save_local:
        factor_data.to_csv("data/famafrench_factors.csv")

    return factor_data

def get_famafrench_data() -> pd.DataFrame:
    try:
        factor_data = pd.read_csv("data/famafrench_factors.csv", index_col=0, parse_dates=True)
        factor_data.index.name = 'date'
        print("Loaded Fama-French factors from 'data/famafrench_factors.csv'")
        return factor_data
    except FileNotFoundError:
        print("Error: 'data/famafrench_factors.csv' not found. Please run download_famafrench_data() first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the local Fama-French data file: {e}")
        return pd.DataFrame()


def download_uk_famafrench_data(save_local: bool) -> pd.DataFrame:
    """
    Downloads the Fama-French 5-factor model data from the Ken French website.

    Parameters:
        save_local (bool): If True, saves the data to a local CSV file.

    Returns:
        A pandas DataFrame with the factor data, indexed by date.
    """
    
    # Try a few known dataset keys via pandas_datareader first (may not support country-specific names)
    candidate_keys = [
        'F-F_Research_Data_5_Factors_2x3',  # US (fallback)
        'Europe_5_Factors',
        'Europe_5_Factors_2x3',
        'UK_5_Factors',
        'United_Kingdom_5_Factors'
    ]

    def _clean_and_format(df: pd.DataFrame) -> pd.DataFrame:
        # Expect the DataFrame to have factor columns including a risk-free column 'RF' possibly
        if 'RF' in df.columns:
            df = df.drop('RF', axis=1)
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        # convert index to timestamps and monthly end frequency
        try:
            df.index = df.index.to_timestamp()
        except Exception:
            pass
        df = df.resample('ME').last().div(100)
        df.index.name = 'date'
        return df

    for key in candidate_keys:
        try:
            factor_data = web.DataReader(key, 'famafrench', start='2010')[0]
            # Basic validation: has common factor names
            cols_lower = [c.upper() for c in factor_data.columns]
            if any(x in cols_lower for x in ['MKT-RF', 'MKT_RF', 'MKT']):
                factor_data = _clean_and_format(factor_data)
                if save_local:
                    factor_data.to_csv("data/uk_famafrench_factors.csv")
                print(f"Loaded Fama-French factors using key '{key}'")
                return factor_data
        except Exception:
            continue

    # Fallback: scrape Ken French Data Library for the United Kingdom link and download
    DATA_LIB = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html'
    try:
        r = requests.get(DATA_LIB, timeout=15)
        r.raise_for_status()
        html = r.text

        # Find the first href after the 'United Kingdom' mention
        idx = html.find('United Kingdom')
        if idx == -1:
            idx = html.find('United_Kingdom')

        href = None
        if idx != -1:
            # search forward for href
            m = re.search(r'href\s*=\s*"([^"]+\.(?:zip|csv))"', html[idx:idx+2000], re.IGNORECASE)
            if m:
                href = m.group(1)

        if not href:
            # As a backup, search whole page for links containing 'United' and 'csv' or 'zip'
            m2 = re.search(r'href\s*=\s*"([^"]*(?:united|uk|united_kingdom)[^"]*\.(?:zip|csv))"', html, re.IGNORECASE)
            if m2:
                href = m2.group(1)

        if not href:
            print('Could not automatically find United Kingdom Fama-French link on Ken French page.')
            raise RuntimeError('UK Fama-French link not found')

        if href.startswith('/'):
            url = 'https://mba.tuck.dartmouth.edu' + href
        elif href.startswith('http'):
            url = href
        else:
            url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/' + href

        print(f'Downloading UK Fama-French data from {url}')
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()

        content = resp.content
        df = None
        if url.lower().endswith('.zip'):
            z = zipfile.ZipFile(io.BytesIO(content))
            # pick the first csv-like file
            csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csv_names:
                raise RuntimeError('No CSV files found inside downloaded zip')
            with z.open(csv_names[0]) as fh:
                try:
                    df_raw = pd.read_csv(fh, header=0, index_col=0, parse_dates=True)
                except Exception:
                    fh.seek(0)
                    df_raw = pd.read_csv(fh, header=1, index_col=0, parse_dates=True)
        else:
            # direct CSV
            try:
                df_raw = pd.read_csv(io.BytesIO(content), header=0, index_col=0, parse_dates=True)
            except Exception:
                df_raw = pd.read_csv(io.BytesIO(content), header=1, index_col=0, parse_dates=True)

        # Try to clean common Ken French formatting where first column is year-month
        try:
            factor_data = _clean_and_format(df_raw)
            if save_local:
                factor_data.to_csv("data/uk_famafrench_factors.csv")
            return factor_data
        except Exception as e:
            print('Failed to parse downloaded UK Fama-French file:', e)
            return pd.DataFrame()

    except Exception as e:
        print(f'Failed to obtain UK Fama-French factors: {e}')
        return pd.DataFrame()

def get_uk_famafrench_data() -> pd.DataFrame:
    try:
        factor_data = pd.read_csv("data/uk_famafrench_factors.csv", index_col=0, parse_dates=True)
        factor_data.index.name = 'date'
        print("Loaded Fama-French factors from 'data/uk_famafrench_factors.csv'")
        return factor_data
    except FileNotFoundError:
        print("Error: 'data/uk_famafrench_factors.csv' not found. Please run download_uk_famafrench_data() first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the local Fama-French data file: {e}")
        return pd.DataFrame()


def download_ftse250_tickers(save_local: bool) -> List[str]:
    """
    Scrapes the current list of FTSE 250 constituent stock tickers from Wikipedia.

    Returns:
        A list of strings containing the FTSE 250 company tickers.
        Returns an empty list [] if fetching or parsing fails.
    """
    WIKI_URL = 'https://en.wikipedia.org/wiki/FTSE_250_Index'
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(WIKI_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        # Find the first table with a 'Ticker' column
        for table in tables:
            if 'Ticker' in table.columns:
                tickers = [str(t).strip().upper() for t in table['Ticker'] if isinstance(t, str) and t.strip()]
                if save_local:
                    pd.DataFrame(tickers, columns=['Ticker']).to_csv("data/ftse250_tickers.csv", index=False)
                    print("Saved FTSE 250 tickers to 'data/ftse250_tickers.csv'")
                print(f"Successfully retrieved {len(tickers)} FTSE 250 tickers.")
                return tickers
        print("Error: Could not find the 'Ticker' column in any table.")
        return []
    except Exception as e:
        print(f"An error occurred while scraping the Wikipedia page: {e}")
        return []



def get_ftse250_tickers() -> List[str]:
    """
    Reads the locally saved CSV file containing FTSE 250 tickers.

    Returns:
        A list of strings containing the FTSE 250 company tickers.
    """
    try:
        df = pd.read_csv("data/ftse250_tickers.csv")
        tickers = df['Ticker'].tolist()
        print(f"Loaded {len(tickers)} tickers from local file.")
        return tickers
    except FileNotFoundError:
        print("Error: 'data/ftse250_tickers.csv' not found. Please run download_ftse250_tickers(save_local=True) first.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the local tickers file: {e}")
        return []


def download_ftse250_data(ftse250_list: List[str], save_local: bool) -> pd.DataFrame:
    """
    Downloads historical data for all FTSE 250 tickers using yfinance.

    Args:
        ftse250_list (List[str]): List of FTSE 250 tickers.
        save_local (bool): If True, saves the data to a local CSV file.

    Returns:
        pd.DataFrame: MultiIndex DataFrame (date, ticker) with historical stock data.
    """
    # Robust batched download from Yahoo Finance using yfinance
    def _to_yahoo_tickers(tickers: List[str]) -> List[str]:
        """Convert simple FTSE tickers (e.g. 'VOD') to Yahoo format (e.g. 'VOD.L')."""
        out = []
        for t in tickers:
            if not isinstance(t, str) or not t:
                continue
            t_up = t.strip().upper()
            # If user already supplied a Yahoo-style ticker, keep it
            if t_up.endswith('.L'):
                out.append(t_up)
            else:
                out.append(f"{t_up}.L")
        return out

    ytickers = _to_yahoo_tickers(ftse250_list)

    # Batch requests to avoid timeouts for very large lists
    batch_size = 40
    retry_attempts = 3
    pause_sec = 1.0

    wide_frames = []
    parts = [ytickers[i:i+batch_size] for i in range(0, len(ytickers), batch_size)]

    for i, chunk in enumerate(parts):
        for attempt in range(1, retry_attempts + 1):
            try:
                print(f"Downloading batch {i+1}/{len(parts)} (tickers: {len(chunk)}) attempt {attempt}")
                data = yf.download(
                    tickers=chunk,
                    start='2015-01-04',
                    end='2025-11-01',
                    auto_adjust=False,
                    group_by='ticker',
                    threads=True,
                    progress=False
                )

                if data is None or data.empty:
                    raise RuntimeError("Empty response from yfinance")

                # Normalize MultiIndex column names to (ticker, field)
                data.columns = pd.MultiIndex.from_tuples(
                    [(ticker, col.strip().lower().replace(' ', '_')) for ticker, col in data.columns]
                )

                wide_frames.append(data)
                time.sleep(pause_sec)
                break
            except Exception as e:
                print(f"Batch {i+1} attempt {attempt} failed: {e}")
                if attempt < retry_attempts:
                    time.sleep(pause_sec * attempt)
                else:
                    print(f"Giving up on batch {i+1} after {retry_attempts} attempts")

    if not wide_frames:
        print('No data downloaded for FTSE 250 tickers.')
        return pd.DataFrame()

    # Concatenate along columns (tickers), align on dates
    df_wide = pd.concat(wide_frames, axis=1)

    # Save the wide MultiIndex CSV to keep compatibility with existing reader
    if save_local and not df_wide.empty:
        df_wide.to_csv("data/ftse250_data.csv")

    # Convert to stacked format (date, ticker) like the previous implementation
    df_wide.columns = df_wide.columns.swaplevel(0, 1)
    df_wide = df_wide.sort_index(axis=1, level=0)
    df = df_wide.stack()
    df.index.names = ['date', 'ticker']
    return df


def get_ftse250_data(start_date: str = None, end_date: str = None, tickers: List[str] = None) -> pd.DataFrame:
    """
    Reads the locally saved CSV file containing FTSE 250 historical data.

    Returns:
        A pandas DataFrame with MultiIndex (Date, Ticker) containing historical stock data.
    """
    try:
        df = pd.read_csv("data/ftse250_data.csv", header=[0,1], index_col=0, parse_dates=True)
        df.columns = df.columns.swaplevel(0, 1)
        df = df.sort_index(axis=1, level=0)
        df = df.stack()
        df.index.names = ['date', 'ticker']
        print("Loaded FTSE 250 historical data from 'data/ftse250_data.csv'")
        if (start_date is not None) and (end_date is not None):
            print('Filtering data between', start_date, 'and', end_date)
            df = df.loc[(df.index.get_level_values('date') >= start_date) & 
                        (df.index.get_level_values('date') <= end_date)]
        if tickers is not None:
            print('Filtering data for specified tickers')
            df = df.loc[df.index.get_level_values('ticker').isin(tickers)]
        return df
    except FileNotFoundError:
        print("Error: 'data/ftse250_data.csv' not found. Please run download_ftse250_data() first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the local FTSE 250 data file: {e}")
        return pd.DataFrame()

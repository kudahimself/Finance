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
        sp500_table = tables[1]
        
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
    
    # Download historical data for all tickers from 2016-01-01
    data = yf.download(tickers=sp500_list, start='2015-01-04', end='2025-11-01', auto_adjust=False, group_by='ticker', threads=True)

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


def download_spy_data(save_local: bool) -> pd.DataFrame:
    """
    Downloads historical data for the SPY ETF from Yahoo Finance.

    Parameters:
        save_local (bool): If True, saves the data to a local CSV file.
    """
    spy = yf.download(tickers='SPY',
                  auto_adjust=False,
                    start='2022-02-01',
                    end='2025-10-31')

    spy_ret = np.log(spy['adj_close']).diff().dropna().rename({'SPY': 'SPY Buy and Hold'}, axis=1)
    if save_local:
        spy_ret.to_csv("data/spy_data.csv")

    return spy_ret

def get_spy_data() -> pd.DataFrame:
    """
    Reads the locally saved CSV file containing SPY historical data.

    Returns:
        A pandas DataFrame with Date index containing SPY returns.
    """
    try:
        spy_ret = pd.read_csv("data/spy_data.csv", index_col=0, parse_dates=True)
        print("Loaded SPY data from 'data/spy_data.csv'")
        return spy_ret
    except FileNotFoundError:
        print("Error: 'data/spy_data.csv' not found. Please run download_spy_data() first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the local SPY data file: {e}")
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
    
# requirements: requests, pandas, yfinance
import requests
import pandas as pd
import yfinance as yf

FMP_KEY = "YOUR_FMP_API_KEY"  # register and get a key

def get_quarterly_eps_fmp(ticker: str, limit=40):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&limit={limit}&apikey={FMP_KEY}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    items = r.json()
    # API returns list of quarter dicts; field names depend on provider version.
    # Many responses include 'eps' or 'epsDiluted' — inspect returned keys.
    rows = []
    for it in items:
        # fallback keys
        eps = it.get("eps") or it.get("epsDiluted") or it.get("epsBasic") or None
        date = it.get("date") or it.get("reportedDate")  # date of quarter end
        if date and eps is not None:
            rows.append({"date": pd.to_datetime(date), "eps": float(eps)})
    return pd.DataFrame(rows).sort_values("date").set_index("date")

def monthly_pe_from_eps(ticker: str, start: str, end: str):
    # 1) month-end prices
    hist = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    month_close = hist['Adj Close'].resample('M').last().rename('price')

    # 2) quarterly eps (from FMP)
    q_eps = get_quarterly_eps_fmp(ticker)
    if q_eps.empty:
        return pd.DataFrame({"price": month_close, "ttm_eps": pd.NA, "pe": pd.NA})

    # 3) compute TTM EPS: rolling sum of last 4 quarters
    q_eps['ttm_eps'] = q_eps['eps'].rolling(4).sum()

    # 4) expand quarterly TTM EPS to monthly index by forward-filling
    monthly_index = month_close.index
    # reindex quarterly ttm_eps to monthly by forward-fill
    ttm_monthly = q_eps['ttm_eps'].reindex(monthly_index, method='ffill').rename('ttm_eps')

    # 5) compute P/E
    pe = month_close / ttm_monthly

    return pd.DataFrame({"price": month_close, "ttm_eps": ttm_monthly, "pe": pe})
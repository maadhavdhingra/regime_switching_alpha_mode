"""
Alpha Vantage Data Fetcher for Regime-Switching Factor Strategy
Handles API rate limits and data storage for stock data.
"""

import pandas as pd
import requests
import time
import logging
from datetime import datetime, timedelta
import os
from typing import Optional, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Alpha Vantage API client with rate limiting and error handling."""
    
    def __init__(self, api_key: str = "PRVU1DG3AU6FP9G6"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.calls_made = 0
        self.last_call_time = None
        self.rate_limit_delay = 12  # seconds between calls (5 per minute)
        
    def _rate_limit(self):
        """Enforce rate limiting: max 5 calls per minute."""
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.calls_made += 1
        
    def fetch_daily_adjusted(self, symbol: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """
        Fetch daily adjusted OHLCV data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        self._rate_limit()
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            logger.info(f"Fetching data for {symbol} (call #{self.calls_made})")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                logger.warning(f"API Note for {symbol}: {data['Note']}")
                return None
                
            if 'Time Series (Daily)' not in data:
                logger.error(f"No time series data found for {symbol}")
                return None
            
            # Parse the time series data
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # Clean column names and convert to float
            df.columns = [
                'Open', 'High', 'Low', 'Close', 'Adjusted_Close',
                'Volume', 'Dividend_Amount', 'Split_Coefficient'
            ]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            return None

def fetch_stock_data(ticker: str, start_date: str, end_date: str, 
                    api_key: str = "PRVU1DG3AU6FP9G6") -> Optional[pd.DataFrame]:
    """
    Fetch stock data for a specific date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_key: Alpha Vantage API key
        
    Returns:
        DataFrame with stock data filtered by date range
    """
    client = AlphaVantageClient(api_key)
    df = client.fetch_daily_adjusted(ticker)
    
    if df is None:
        return None
    
    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    # Add ticker column
    df['Ticker'] = ticker
    
    logger.info(f"Filtered data for {ticker}: {len(df)} records from {start_date} to {end_date}")
    return df

def save_to_csv(df: pd.DataFrame, ticker: str, data_dir: str = "data") -> bool:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        ticker: Stock ticker symbol
        data_dir: Directory to save data files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename with current date
        filename = f"{ticker}_daily_adjusted.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=True)
        logger.info(f"Saved {len(df)} records to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save data for {ticker}: {e}")
        return False

def fetch_multiple_stocks(tickers: List[str], start_date: str, end_date: str,
                         data_dir: str = "data", api_key: str = "PRVU1DG3AU6FP9G6") -> dict:
    """
    Fetch and save data for multiple stocks.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_dir: Directory to save data files
        api_key: Alpha Vantage API key
        
    Returns:
        Dictionary with ticker -> success status
    """
    results = {}
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        
        try:
            df = fetch_stock_data(ticker, start_date, end_date, api_key)
            
            if df is not None and len(df) > 0:
                success = save_to_csv(df, ticker, data_dir)
                results[ticker] = success
            else:
                logger.warning(f"No data retrieved for {ticker}")
                results[ticker] = False
                
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            results[ticker] = False
    
    return results

def load_stock_data(ticker: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
    """
    Load stock data from saved CSV file.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing data files
        
    Returns:
        DataFrame with stock data or None if file not found
    """
    try:
        filename = f"{ticker}_daily_adjusted.csv"
        filepath = os.path.join(data_dir, filename)
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} records for {ticker}")
        return df
        
    except FileNotFoundError:
        logger.error(f"Data file not found for {ticker}: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Failed to load data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    # Fetch data for all tickers
    results = fetch_multiple_stocks(tickers, start_date, end_date)
    
    print("\nData fetch results:")
    for ticker, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{ticker}: {status}")
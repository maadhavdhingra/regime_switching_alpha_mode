"""
Feature Engineering for Regime-Switching Factor Strategy
Computes momentum, value, quality, and volatility factors.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactorEngineer:
    """Factory class for computing various factor scores."""
    
    def __init__(self):
        self.factor_functions = {
            'momentum': self.compute_momentum_factor,
            'volatility': self.compute_volatility_factor,
            'value': self.compute_value_factor,
            'quality': self.compute_quality_factor
        }
    
    def compute_momentum_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute momentum factor: 12-month return minus 1-month return.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with momentum scores
        """
        # Calculate returns
        df = df.copy()
        df['Returns'] = df['Adjusted_Close'].pct_change()
        
        # 12-month (252 trading days) cumulative return
        df['Return_12M'] = df['Adjusted_Close'].pct_change(252)
        
        # 1-month (21 trading days) cumulative return
        df['Return_1M'] = df['Adjusted_Close'].pct_change(21)
        
        # Momentum = 12M return - 1M return (to avoid reversal effects)
        momentum = df['Return_12M'] - df['Return_1M']
        
        return momentum.fillna(0)
    
    def compute_volatility_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volatility factor: negative of 30-day rolling volatility.
        Lower volatility gets higher scores (low-vol anomaly).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with volatility scores (negative volatility)
        """
        # Calculate daily returns
        returns = df['Adjusted_Close'].pct_change()
        
        # 30-day rolling volatility (annualized)
        volatility = returns.rolling(window=30).std() * np.sqrt(252)
        
        # Return negative volatility (lower vol = higher score)
        return -volatility.fillna(0)
    
    def compute_value_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute value factor: mock P/E ratio based on price trends.
        This is a placeholder - replace with real fundamentals when available.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with value scores (inverse P/E proxy)
        """
        # Mock P/E calculation based on price momentum and volatility
        # Lower recent performance + higher volatility = potentially undervalued
        
        # Short-term performance (worse = more attractive for value)
        short_perf = df['Adjusted_Close'].pct_change(63)  # 3-month return
        
        # Price-to-moving-average ratio (lower = cheaper)
        ma_200 = df['Adjusted_Close'].rolling(200).mean()
        price_to_ma = df['Adjusted_Close'] / ma_200
        
        # Mock value score: inverse of price momentum and price-to-MA
        # Higher scores for lower recent returns and lower price-to-MA ratios
        value_score = -(short_perf + price_to_ma - 1)
        
        return value_score.fillna(0)
    
    def compute_quality_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute quality factor: mock ROE based on price stability and trend.
        This is a placeholder - replace with real fundamentals when available.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with quality scores
        """
        # Mock quality based on price trend consistency and profitability proxy
        
        # Price trend strength (consistent upward trend = quality)
        # Use 50-day and 200-day MA relationship
        ma_50 = df['Adjusted_Close'].rolling(50).mean()
        ma_200 = df['Adjusted_Close'].rolling(200).mean()
        trend_strength = (ma_50 / ma_200) - 1
        
        # Price stability (lower volatility = higher quality)
        returns = df['Adjusted_Close'].pct_change()
        stability = -returns.rolling(60).std()  # Negative volatility
        
        # Volume consistency (steady volume = institutional interest)
        volume_consistency = -df['Volume'].rolling(30).std() / df['Volume'].rolling(30).mean()
        
        # Combine factors for quality score
        quality_score = (
            0.4 * trend_strength + 
            0.4 * stability + 
            0.2 * volume_consistency.fillna(0)
        )
        
        return quality_score.fillna(0)

def compute_all_factors(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Compute all factor scores for a single stock.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with all factor scores
    """
    engineer = FactorEngineer()
    
    # Compute each factor
    factors_df = pd.DataFrame(index=df.index)
    factors_df['Ticker'] = ticker
    
    for factor_name, factor_func in engineer.factor_functions.items():
        try:
            factors_df[factor_name] = factor_func(df)
            logger.info(f"Computed {factor_name} factor for {ticker}")
        except Exception as e:
            logger.error(f"Failed to compute {factor_name} for {ticker}: {e}")
            factors_df[factor_name] = 0
    
    return factors_df

def compute_multi_stock_factors(stock_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute factors for multiple stocks and combine into single DataFrame.
    
    Args:
        stock_data_dict: Dictionary with ticker -> DataFrame mapping
        
    Returns:
        Combined DataFrame with all factors for all stocks
    """
    all_factors = []
    
    for ticker, df in stock_data_dict.items():
        if df is None or len(df) == 0:
            logger.warning(f"No data for {ticker}, skipping factor computation")
            continue
            
        factors_df = compute_all_factors(df, ticker)
        all_factors.append(factors_df)
    
    if not all_factors:
        logger.error("No factors computed for any stocks")
        return pd.DataFrame()
    
    # Combine all factors
    combined_factors = pd.concat(all_factors, axis=0)
    combined_factors = combined_factors.sort_index()
    
    logger.info(f"Computed factors for {len(stock_data_dict)} stocks")
    return combined_factors

def normalize_factors(factors_df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """
    Normalize factor scores across stocks for each time period.
    
    Args:
        factors_df: DataFrame with raw factor scores
        method: Normalization method ('zscore', 'rank', 'minmax')
        
    Returns:
        DataFrame with normalized factor scores
    """
    if factors_df.empty:
        return factors_df
    
    factor_columns = ['momentum', 'volatility', 'value', 'quality']
    normalized_df = factors_df.copy()
    
    # Group by date for cross-sectional normalization
    for date, group in factors_df.groupby(factors_df.index):
        if len(group) < 2:  # Need at least 2 stocks for normalization
            continue
            
        for factor in factor_columns:
            if factor not in group.columns:
                continue
                
            factor_values = group[factor]
            
            if method == "zscore":
                # Z-score normalization
                mean_val = factor_values.mean()
                std_val = factor_values.std()
                if std_val > 0:
                    normalized_values = (factor_values - mean_val) / std_val
                else:
                    normalized_values = factor_values
                    
            elif method == "rank":
                # Rank-based normalization (0 to 1)
                normalized_values = factor_values.rank(pct=True)
                
            elif method == "minmax":
                # Min-max normalization
                min_val = factor_values.min()
                max_val = factor_values.max()
                if max_val > min_val:
                    normalized_values = (factor_values - min_val) / (max_val - min_val)
                else:
                    normalized_values = factor_values
            else:
                normalized_values = factor_values
            
            # Update the normalized DataFrame
            mask = normalized_df.index == date
            normalized_df.loc[mask, factor] = normalized_values
    
    logger.info(f"Normalized factors using {method} method")
    return normalized_df

def compute_composite_scores(factors_df: pd.DataFrame, 
                           weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compute composite factor scores using specified weights.
    
    Args:
        factors_df: DataFrame with normalized factor scores
        weights: Dictionary with factor -> weight mapping
        
    Returns:
        DataFrame with composite scores added
    """
    if weights is None:
        weights = {
            'momentum': 0.25,
            'volatility': 0.25,
            'value': 0.25,
            'quality': 0.25
        }
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    if total_weight != 1.0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    result_df = factors_df.copy()
    factor_columns = [col for col in weights.keys() if col in factors_df.columns]
    
    # Compute weighted composite score
    result_df['composite_score'] = 0
    for factor in factor_columns:
        result_df['composite_score'] += weights[factor] * factors_df[factor]
    
    logger.info(f"Computed composite scores with weights: {weights}")
    return result_df

def get_factor_statistics(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all factors.
    
    Args:
        factors_df: DataFrame with factor scores
        
    Returns:
        DataFrame with factor statistics
    """
    factor_columns = ['momentum', 'volatility', 'value', 'quality', 'composite_score']
    available_factors = [col for col in factor_columns if col in factors_df.columns]
    
    if not available_factors:
        return pd.DataFrame()
    
    stats_df = factors_df[available_factors].describe()
    
    # Add additional statistics
    for factor in available_factors:
        stats_df.loc['skewness', factor] = factors_df[factor].skew()
        stats_df.loc['kurtosis', factor] = factors_df[factor].kurtosis()
    
    return stats_df

if __name__ == "__main__":
    # Example usage
    from data_fetch import load_stock_data
    
    # Load sample data
    tickers = ["AAPL", "MSFT", "AMZN"]
    stock_data = {}
    
    for ticker in tickers:
        df = load_stock_data(ticker)
        if df is not None:
            stock_data[ticker] = df
    
    if stock_data:
        # Compute factors
        factors_df = compute_multi_stock_factors(stock_data)
        
        # Normalize factors
        normalized_factors = normalize_factors(factors_df, method="zscore")
        
        # Compute composite scores
        final_scores = compute_composite_scores(normalized_factors)
        
        # Get statistics
        stats = get_factor_statistics(final_scores)
        
        print("Factor Statistics:")
        print(stats)
        
        # Save results
        final_scores.to_csv("data/factor_scores.csv")
        print("Saved factor scores to data/factor_scores.csv")
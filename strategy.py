"""
Regime-Switching Factor Strategy Implementation
Dynamic factor allocation based on detected market regimes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeSwitchingStrategy:
    """
    Factor strategy that adjusts factor weights based on market regimes.
    """
    
    def __init__(self, n_stocks: int = None, rebalance_freq: str = 'M'):
        """
        Initialize regime-switching strategy.
        
        Args:
            n_stocks: Number of stocks to select (None for percentage-based)
            rebalance_freq: Rebalancing frequency ('M', 'Q', 'W')
        """
        self.n_stocks = n_stocks
        self.rebalance_freq = rebalance_freq
        self.factor_weights = self._initialize_regime_weights()
        self.portfolio_history = []
        self.performance_history = []
        
    def _initialize_regime_weights(self) -> Dict[int, Dict[str, float]]:
        """
        Initialize factor weights for different regimes.
        
        Returns:
            Dictionary mapping regime -> factor weights
        """
        # Default regime-specific factor weights
        regime_weights = {
            0: {  # Low volatility regime
                'momentum': 0.35,
                'value': 0.25,
                'quality': 0.30,
                'volatility': 0.10
            },
            1: {  # Medium volatility regime
                'momentum': 0.25,
                'value': 0.30,
                'quality': 0.25,
                'volatility': 0.20
            },
            2: {  # High volatility regime
                'momentum': 0.15,
                'value': 0.35,
                'quality': 0.35,
                'volatility': 0.15
            }
        }
        
        return regime_weights
    
    def set_regime_weights(self, regime_weights: Dict[int, Dict[str, float]]):
        """
        Set custom factor weights for each regime.
        
        Args:
            regime_weights: Dictionary mapping regime -> factor weights
        """
        # Validate weights sum to 1 for each regime
        for regime, weights in regime_weights.items():
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"Regime {regime} weights sum to {total_weight}, normalizing")
                regime_weights[regime] = {k: v/total_weight for k, v in weights.items()}
        
        self.factor_weights = regime_weights
        logger.info(f"Updated regime weights: {regime_weights}")
    
    def compute_composite_scores(self, factors_df: pd.DataFrame, 
                               regimes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite factor scores based on current regime.
        
        Args:
            factors_df: DataFrame with factor scores
            regimes_df: DataFrame with regime labels
            
        Returns:
            DataFrame with composite scores
        """
        if factors_df.empty or regimes_df.empty:
            return pd.DataFrame()
        
        # Align data
        common_dates = factors_df.index.intersection(regimes_df.index)
        factors_aligned = factors_df.loc[common_dates]
        regimes_aligned = regimes_df.loc[common_dates]
        
        result_df = factors_aligned.copy()
        result_df['regime'] = regimes_aligned['regime']
        result_df['composite_score'] = 0.0
        
        factor_columns = ['momentum', 'value', 'quality', 'volatility']
        available_factors = [col for col in factor_columns if col in factors_aligned.columns]
        
        # Group by date and compute regime-specific scores
        for date in common_dates:
            date_mask = result_df.index == date
            date_data = result_df.loc[date_mask]
            
            if len(date_data) == 0:
                continue
            
            # Get regime for this date (assume same regime for all stocks on same date)
            current_regime = date_data['regime'].iloc[0]
            
            # Get weights for current regime
            if current_regime in self.factor_weights:
                weights = self.factor_weights[current_regime]
            else:
                # Default equal weights if regime not found
                weights = {factor: 1.0/len(available_factors) for factor in available_factors}
                logger.warning(f"Regime {current_regime} not found, using equal weights")
            
            # Calculate composite score for this date
            composite_score = 0.0
            for factor in available_factors:
                if factor in weights:
                    composite_score += weights[factor] * date_data[factor]
                    
            result_df.loc[date_mask, 'composite_score'] = composite_score
        
        return result_df
    
    def select_portfolio(self, scores_df: pd.DataFrame, 
                        selection_method: str = 'top_pct',
                        top_pct: float = 0.3) -> pd.DataFrame:
        """
        Select portfolio based on composite scores.
        
        Args:
            scores_df: DataFrame with composite scores
            selection_method: Selection method ('top_n', 'top_pct', 'threshold')
            top_pct: Percentage of top stocks to select (for 'top_pct' method)
            
        Returns:
            DataFrame with selected portfolio
        """
        portfolio_selections = []
        
        # Group by rebalancing periods
        rebalance_dates = self._get_rebalance_dates(scores_df.index)
        
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_rebal_date = rebalance_dates[i + 1]
            
            # Get scores for rebalancing date
            period_mask = (scores_df.index >= rebal_date) & (scores_df.index < next_rebal_date)
            period_data = scores_df.loc[period_mask]
            
            if len(period_data) == 0:
                continue
            
            # Use data from rebalancing date for selection
            rebal_data = period_data.groupby('Ticker').first()  # Get first available score per ticker
            
            if len(rebal_data) < 2:
                continue
            
            # Select stocks based on method
            if selection_method == 'top_pct':
                n_select = max(1, int(len(rebal_data) * top_pct))
                selected = rebal_data.nlargest(n_select, 'composite_score')
            elif selection_method == 'top_n' and self.n_stocks:
                n_select = min(self.n_stocks, len(rebal_data))
                selected = rebal_data.nlargest(n_select, 'composite_score')
            elif selection_method == 'threshold':
                threshold = rebal_data['composite_score'].quantile(0.7)
                selected = rebal_data[rebal_data['composite_score'] >= threshold]
            else:
                # Default to top 30%
                n_select = max(1, int(len(rebal_data) * 0.3))
                selected = rebal_data.nlargest(n_select, 'composite_score')
            
            # Create portfolio entries for the period
            for ticker in selected.index:
                portfolio_entry = {
                    'rebalance_date': rebal_date,
                    'period_start': rebal_date,
                    'period_end': next_rebal_date,
                    'ticker': ticker,
                    'composite_score': selected.loc[ticker, 'composite_score'],
                    'regime': selected.loc[ticker, 'regime'],
                    'weight': 1.0 / len(selected)  # Equal weight
                }
                
                # Add factor scores
                for factor in ['momentum', 'value', 'quality', 'volatility']:
                    if factor in selected.columns:
                        portfolio_entry[factor] = selected.loc[ticker, factor]
                
                portfolio_selections.append(portfolio_entry)
        
        if not portfolio_selections:
            return pd.DataFrame()
        
        portfolio_df = pd.DataFrame(portfolio_selections)
        logger.info(f"Portfolio selection completed: {len(portfolio_df)} selections")
        
        return portfolio_df
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Get rebalancing dates based on frequency.
        
        Args:
            date_index: DatetimeIndex from data
            
        Returns:
            List of rebalancing dates
        """
        if len(date_index) == 0:
            return []
        
        start_date = date_index.min()
        end_date = date_index.max()
        
        if self.rebalance_freq == 'M':
            # Monthly rebalancing
            rebal_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        elif self.rebalance_freq == 'Q':
            # Quarterly rebalancing
            rebal_dates = pd.date_range(start=start_date, end=end_date, freq='QS')
        elif self.rebalance_freq == 'W':
            # Weekly rebalancing
            rebal_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        else:
            # Default monthly
            rebal_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Filter to actual dates in data
        rebal_dates = [date for date in rebal_dates if date <= end_date]
        rebal_dates.append(end_date)  # Add end date
        
        return rebal_dates
    
    def calculate_turnover(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio turnover between rebalancing periods.
        
        Args:
            portfolio_df: DataFrame with portfolio selections
            
        Returns:
            DataFrame with turnover metrics
        """
        if portfolio_df.empty:
            return pd.DataFrame()
        
        turnover_data = []
        rebal_dates = sorted(portfolio_df['rebalance_date'].unique())
        
        for i in range(1, len(rebal_dates)):
            prev_date = rebal_dates[i-1]
            curr_date = rebal_dates[i]
            
            prev_portfolio = set(portfolio_df[portfolio_df['rebalance_date'] == prev_date]['ticker'])
            curr_portfolio = set(portfolio_df[portfolio_df['rebalance_date'] == curr_date]['ticker'])
            
            if len(prev_portfolio) == 0:
                continue
            
            # Calculate turnover metrics
            unchanged = len(prev_portfolio.intersection(curr_portfolio))
            total_positions = len(prev_portfolio)
            turnover_rate = 1 - (unchanged / total_positions)
            
            turnover_data.append({
                'rebalance_date': curr_date,
                'prev_positions': len(prev_portfolio),
                'curr_positions': len(curr_portfolio),
                'unchanged_positions': unchanged,
                'turnover_rate': turnover_rate
            })
        
        return pd.DataFrame(turnover_data)
    
    def get_strategy_summary(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        Generate strategy summary statistics.
        
        Args:
            portfolio_df: DataFrame with portfolio selections
            
        Returns:
            Dictionary with strategy summary
        """
        if portfolio_df.empty:
            return {}
        
        summary = {
            'total_selections': len(portfolio_df),
            'unique_stocks': portfolio_df['ticker'].nunique(),
            'rebalance_periods': portfolio_df['rebalance_date'].nunique(),
            'avg_portfolio_size': portfolio_df.groupby('rebalance_date')['ticker'].count().mean(),
            'rebalance_frequency': self.rebalance_freq
        }
        
        # Regime distribution
        regime_dist = portfolio_df['regime'].value_counts(normalize=True)
        for regime, pct in regime_dist.items():
            summary[f'regime_{regime}_pct'] = pct
        
        # Factor score statistics
        factor_columns = ['momentum', 'value', 'quality', 'volatility', 'composite_score']
        for factor in factor_columns:
            if factor in portfolio_df.columns:
                summary[f'{factor}_mean'] = portfolio_df[factor].mean()
                summary[f'{factor}_std'] = portfolio_df[factor].std()
        
        # Calculate turnover if possible
        turnover_df = self.calculate_turnover(portfolio_df)
        if not turnover_df.empty:
            summary['avg_turnover'] = turnover_df['turnover_rate'].mean()
            summary['max_turnover'] = turnover_df['turnover_rate'].max()
            summary['min_turnover'] = turnover_df['turnover_rate'].min()
        
        return summary

def run_regime_switching_strategy(factors_df: pd.DataFrame,
                                regimes_df: pd.DataFrame,
                                regime_weights: Dict[int, Dict[str, float]] = None,
                                selection_method: str = 'top_pct',
                                top_pct: float = 0.3,
                                rebalance_freq: str = 'M') -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete regime-switching factor strategy.
    
    Args:
        factors_df: DataFrame with factor scores
        regimes_df: DataFrame with regime labels
        regime_weights: Custom regime-specific factor weights
        selection_method: Portfolio selection method
        top_pct: Percentage of stocks to select
        rebalance_freq: Rebalancing frequency
        
    Returns:
        Tuple of (portfolio_df, strategy_summary)
    """
    # Initialize strategy
    strategy = RegimeSwitchingStrategy(rebalance_freq=rebalance_freq)
    
    # Set custom weights if provided
    if regime_weights:
        strategy.set_regime_weights(regime_weights)
    
    # Compute regime-specific composite scores
    scores_df = strategy.compute_composite_scores(factors_df, regimes_df)
    
    # Select portfolio
    portfolio_df = strategy.select_portfolio(
        scores_df, selection_method, top_pct
    )
    
    # Generate summary
    summary = strategy.get_strategy_summary(portfolio_df)
    
    logger.info("Regime-switching strategy execution completed")
    return portfolio_df, summary

if __name__ == "__main__":
    # Example usage
    from data_fetch import load_stock_data
    from feature_engineering import compute_multi_stock_factors, normalize_factors
    from regime_detection import detect_market_regimes
    
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
        normalized_factors = normalize_factors(factors_df)
        
        # Detect regimes (using market index approach)
        from regime_detection import create_market_regime_index
        market_regimes, _ = create_market_regime_index(stock_data)
        
        # Run strategy
        portfolio_df, summary = run_regime_switching_strategy(
            normalized_factors, market_regimes
        )
        
        print("Strategy Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save results
        portfolio_df.to_csv("data/portfolio_selections.csv", index=False)
        print("Saved portfolio selections to data/portfolio_selections.csv")
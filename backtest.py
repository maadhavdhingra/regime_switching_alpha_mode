"""
Backtesting Module for Regime-Switching Factor Strategy

This module provides comprehensive backtesting functionality for evaluating
the performance of regime-switching investment strategies.

Author: Quant Finance Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RegimeBacktester:
    """
    A comprehensive backtesting engine for regime-switching strategies.
    
    Features:
    - Monthly rebalancing simulation
    - Performance metrics calculation (Sharpe, Max DD, etc.)
    - Vectorized operations for efficiency
    - Transaction cost modeling
    - Detailed performance attribution
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001,
                 rebalance_freq: str = 'M'):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting portfolio value
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            transaction_cost: Transaction cost per trade (0.1% default)
            rebalance_freq: Rebalancing frequency ('M' for monthly)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        
        # Results storage
        self.portfolio_returns = None
        self.portfolio_values = None
        self.positions = None
        self.metrics = {}
        
    def run_backtest(self, 
                     price_data: pd.DataFrame,
                     strategy_signals: pd.DataFrame,
                     regime_periods: pd.Series) -> Dict:
        """
        Execute the complete backtesting simulation.
        
        Args:
            price_data: DataFrame with price data (columns: tickers, index: dates)
            strategy_signals: DataFrame with strategy positions/weights
            regime_periods: Series with regime classifications
            
        Returns:
            Dictionary containing all backtest results and metrics
        """
        print("Starting backtest simulation...")
        
        # Validate inputs
        self._validate_inputs(price_data, strategy_signals)
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Align data
        common_dates = returns.index.intersection(strategy_signals.index)
        returns = returns.loc[common_dates]
        strategy_signals = strategy_signals.loc[common_dates]
        
        # Simulate portfolio
        portfolio_results = self._simulate_portfolio(returns, strategy_signals)
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_results, regime_periods)
        
        # Store results
        self.portfolio_returns = portfolio_results['returns']
        self.portfolio_values = portfolio_results['values']
        self.positions = portfolio_results['positions']
        self.metrics = metrics
        
        print("Backtest completed successfully!")
        return {
            'portfolio_returns': self.portfolio_returns,
            'portfolio_values': self.portfolio_values,
            'positions': self.positions,
            'metrics': self.metrics,
            'regime_performance': self._analyze_regime_performance(regime_periods)
        }
    
    def _validate_inputs(self, price_data: pd.DataFrame, strategy_signals: pd.DataFrame):
        """Validate input data consistency."""
        if price_data.empty or strategy_signals.empty:
            raise ValueError("Input data cannot be empty")
        
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise ValueError("Price data must have DatetimeIndex")
        
        if not isinstance(strategy_signals.index, pd.DatetimeIndex):
            raise ValueError("Strategy signals must have DatetimeIndex")
    
    def _simulate_portfolio(self, 
                          returns: pd.DataFrame, 
                          strategy_signals: pd.DataFrame) -> Dict:
        """
        Simulate portfolio performance with monthly rebalancing.
        
        Args:
            returns: Asset returns DataFrame
            strategy_signals: Strategy position signals
            
        Returns:
            Dictionary with portfolio returns, values, and positions
        """
        # Initialize containers
        portfolio_values = [self.initial_capital]
        portfolio_returns = []
        portfolio_positions = []
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(returns.index)
        
        current_weights = pd.Series(0.0, index=returns.columns)
        current_value = self.initial_capital
        
        for i, date in enumerate(returns.index):
            # Check if rebalancing date
            if date in rebalance_dates:
                # Get new target weights
                if date in strategy_signals.index:
                    target_weights = strategy_signals.loc[date]
                    target_weights = target_weights.fillna(0.0)
                    
                    # Normalize weights to sum to 1
                    if target_weights.sum() > 0:
                        target_weights = target_weights / target_weights.sum()
                    
                    # Calculate transaction costs
                    turnover = np.abs(target_weights - current_weights).sum()
                    transaction_costs = turnover * self.transaction_cost * current_value
                    
                    # Update weights after costs
                    current_value -= transaction_costs
                    current_weights = target_weights.copy()
            
            # Calculate daily return
            if i > 0:  # Skip first day
                daily_returns = returns.iloc[i]
                portfolio_return = (current_weights * daily_returns).sum()
                
                # Update portfolio value
                current_value *= (1 + portfolio_return)
                portfolio_returns.append(portfolio_return)
                portfolio_values.append(current_value)
                portfolio_positions.append(current_weights.copy())
            else:
                portfolio_positions.append(current_weights.copy())
        
        # Convert to pandas objects
        portfolio_returns = pd.Series(portfolio_returns, index=returns.index[1:])
        portfolio_values = pd.Series(portfolio_values, index=returns.index)
        
        return {
            'returns': portfolio_returns,
            'values': portfolio_values,
            'positions': pd.DataFrame(portfolio_positions, index=returns.index)
        }
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List:
        """Get rebalancing dates based on frequency."""
        if self.rebalance_freq == 'M':
            # Monthly rebalancing - first trading day of each month
            monthly_dates = []
            current_month = None
            
            for date in date_index:
                if current_month is None or date.month != current_month:
                    monthly_dates.append(date)
                    current_month = date.month
            
            return monthly_dates
        else:
            # Default to monthly if not specified
            return self._get_rebalance_dates(date_index)
    
    def _calculate_metrics(self, 
                          portfolio_results: Dict, 
                          regime_periods: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_results: Portfolio simulation results
            regime_periods: Regime classifications for attribution
            
        Returns:
            Dictionary of performance metrics
        """
        returns = portfolio_results['returns']
        values = portfolio_results['values']
        
        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        
        # Drawdown analysis
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = returns[returns < 0].mean()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std != 0 else np.inf
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Sortino Ratio': f"{sortino_ratio:.3f}",
            'Calmar Ratio': f"{calmar_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Avg Win': f"{avg_win:.4f}" if not np.isnan(avg_win) else "N/A",
            'Avg Loss': f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "N/A",
            'Number of Trades': len(returns),
            'Raw Values': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }
        
        return metrics
    
    def _analyze_regime_performance(self, regime_periods: pd.Series) -> Dict:
        """
        Analyze performance attribution by regime.
        
        Args:
            regime_periods: Series with regime classifications
            
        Returns:
            Dictionary with regime-specific performance metrics
        """
        if self.portfolio_returns is None or regime_periods is None:
            return {}
        
        # Align regime periods with returns
        common_dates = self.portfolio_returns.index.intersection(regime_periods.index)
        returns_aligned = self.portfolio_returns.loc[common_dates]
        regimes_aligned = regime_periods.loc[common_dates]
        
        regime_performance = {}
        
        for regime in regimes_aligned.unique():
            regime_mask = regimes_aligned == regime
            regime_returns = returns_aligned[regime_mask]
            
            if len(regime_returns) > 0:
                regime_performance[f'Regime_{regime}'] = {
                    'Total Return': f"{regime_returns.sum():.2%}",
                    'Avg Daily Return': f"{regime_returns.mean():.4f}",
                    'Volatility': f"{regime_returns.std() * np.sqrt(252):.2%}",
                    'Win Rate': f"{(regime_returns > 0).mean():.2%}",
                    'Number of Days': len(regime_returns)
                }
        
        return regime_performance
    
    def print_summary(self):
        """Print a formatted summary of backtest results."""
        if not self.metrics:
            print("No backtest results available. Run backtest first.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        for key, value in self.metrics.items():
            if key != 'Raw Values':
                print(f"{key:<20}: {value}")
        
        print("="*60)
    
    def get_drawdown_series(self) -> pd.Series:
        """Get the drawdown time series."""
        if self.portfolio_values is None:
            return pd.Series()
        
        rolling_max = self.portfolio_values.expanding().max()
        return (self.portfolio_values - rolling_max) / rolling_max


def run_simple_backtest(price_data: pd.DataFrame, 
                       strategy_signals: pd.DataFrame,
                       regime_periods: pd.Series = None) -> Dict:
    """
    Convenience function to run a simple backtest.
    
    Args:
        price_data: Price data DataFrame
        strategy_signals: Strategy position signals
        regime_periods: Optional regime classifications
        
    Returns:
        Backtest results dictionary
    """
    backtester = RegimeBacktester()
    results = backtester.run_backtest(price_data, strategy_signals, regime_periods)
    backtester.print_summary()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Regime-Switching Strategy Backtester")
    print("Run this module through the main notebook for full functionality.")
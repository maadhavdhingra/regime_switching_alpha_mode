"""
Regime Detection for Market State Classification
Uses Hidden Markov Models and volatility-based methods to identify market regimes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Using alternative regime detection methods.")

class RegimeDetector:
    """Market regime detection using various statistical methods."""
    
    def __init__(self, n_regimes: int = 3, method: str = "hmm"):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            method: Detection method ('hmm', 'volatility', 'gmm')
        """
        self.n_regimes = n_regimes
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "Low Volatility",
            1: "Medium Volatility", 
            2: "High Volatility"
        }
        
    def _prepare_features(self, returns: pd.Series, window: int = 20) -> pd.DataFrame:
        """
        Prepare features for regime detection.
        
        Args:
            returns: Time series of returns
            window: Rolling window for feature calculation
            
        Returns:
            DataFrame with regime detection features
        """
        features = pd.DataFrame(index=returns.index)
        
        # Rolling volatility
        features['volatility'] = returns.rolling(window).std()
        
        # Rolling mean return
        features['mean_return'] = returns.rolling(window).mean()
        
        # Rolling skewness
        features['skewness'] = returns.rolling(window).skew()
        
        # Rolling kurtosis
        features['kurtosis'] = returns.rolling(window).kurtosis()
        
        # VIX-like measure (volatility of volatility)
        features['vol_of_vol'] = features['volatility'].rolling(window).std()
        
        # Momentum (cumulative return over window)
        features['momentum'] = returns.rolling(window).sum()
        
        # Fill missing values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def detect_regimes_hmm(self, features: pd.DataFrame) -> np.array:
        """
        Detect regimes using Hidden Markov Model.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of regime labels
        """
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, falling back to GMM")
            return self.detect_regimes_gmm(features)
        
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes, 
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            # Fit model
            self.model.fit(features_scaled)
            
            # Predict regimes
            regimes = self.model.predict(features_scaled)
            
            logger.info(f"HMM regime detection completed with {self.n_regimes} regimes")
            return regimes
            
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            logger.info("Falling back to GMM method")
            return self.detect_regimes_gmm(features)
    
    def detect_regimes_gmm(self, features: pd.DataFrame) -> np.array:
        """
        Detect regimes using Gaussian Mixture Model.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of regime labels
        """
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit Gaussian Mixture Model
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            
            # Fit and predict
            regimes = self.model.fit_predict(features_scaled)
            
            logger.info(f"GMM regime detection completed with {self.n_regimes} regimes")
            return regimes
            
        except Exception as e:
            logger.error(f"GMM regime detection failed: {e}")
            return self.detect_regimes_volatility(features)
    
    def detect_regimes_volatility(self, features: pd.DataFrame) -> np.array:
        """
        Simple volatility-based regime detection.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of regime labels
        """
        try:
            # Use volatility for simple regime classification
            volatility = features['volatility']
            
            # Define quantile-based thresholds
            if self.n_regimes == 2:
                threshold = volatility.quantile(0.5)
                regimes = (volatility > threshold).astype(int)
            elif self.n_regimes == 3:
                low_threshold = volatility.quantile(0.33)
                high_threshold = volatility.quantile(0.67)
                regimes = np.where(volatility <= low_threshold, 0,
                                 np.where(volatility <= high_threshold, 1, 2))
            else:
                # For more regimes, use equal-width quantiles
                quantiles = np.linspace(0, 1, self.n_regimes + 1)
                thresholds = volatility.quantile(quantiles[1:-1])
                regimes = pd.cut(volatility, 
                               bins=[-np.inf] + thresholds.tolist() + [np.inf],
                               labels=False)
            
            logger.info(f"Volatility-based regime detection completed")
            return regimes
            
        except Exception as e:
            logger.error(f"Volatility regime detection failed: {e}")
            # Return simple alternating pattern as fallback
            return np.tile(np.arange(self.n_regimes), 
                          len(features) // self.n_regimes + 1)[:len(features)]
    
    def fit_predict(self, returns: pd.Series, feature_window: int = 20) -> np.array:
        """
        Fit regime detection model and predict regimes.
        
        Args:
            returns: Time series of returns
            feature_window: Window for feature calculation
            
        Returns:
            Array of regime labels
        """
        # Prepare features
        features = self._prepare_features(returns, feature_window)
        
        # Choose detection method
        if self.method == "hmm":
            regimes = self.detect_regimes_hmm(features)
        elif self.method == "gmm":
            regimes = self.detect_regimes_gmm(features)
        elif self.method == "volatility":
            regimes = self.detect_regimes_volatility(features)
        else:
            logger.warning(f"Unknown method {self.method}, using GMM")
            regimes = self.detect_regimes_gmm(features)
        
        return regimes
    
    def get_regime_statistics(self, returns: pd.Series, regimes: np.array) -> pd.DataFrame:
        """
        Compute statistics for each regime.
        
        Args:
            returns: Time series of returns
            regimes: Array of regime labels
            
        Returns:
            DataFrame with regime statistics
        """
        regime_stats = []
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) == 0:
                continue
            
            stats = {
                'regime': regime,
                'regime_name': self.regime_names.get(regime, f"Regime {regime}"),
                'count': len(regime_returns),
                'frequency': len(regime_returns) / len(returns),
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'min_return': regime_returns.min(),
      """
Regime Detection for Market State Classification
Uses Hidden Markov Models and volatility-based methods to identify market regimes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Using alternative regime detection methods.")

class RegimeDetector:
    """Market regime detection using various statistical methods."""
    
    def __init__(self, n_regimes: int = 3, method: str = "hmm"):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            method: Detection method ('hmm', 'volatility', 'gmm')
        """
        self.n_regimes = n_regimes
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "Low Volatility",
            1: "Medium Volatility", 
            2: "High Volatility"
        }
        
    def _prepare_features(self, returns: pd.Series, window: int = 20) -> pd.DataFrame:
        """
        Prepare features for regime detection.
        
        Args:
            returns: Time series of returns
            window: Rolling window for feature calculation
            
        Returns:
            DataFrame with regime detection features
        """
        features = pd.DataFrame(index=returns.index)
        
        # Rolling volatility
        features['volatility'] = returns.rolling(window).std()
        
        # Rolling mean return
        features['mean_return'] = returns.rolling(window).mean()
        
        # Rolling skewness
        features['skewness'] = returns.rolling(window).skew()
        
        # Rolling kurtosis
        features['kurtosis'] = returns.rolling(window).kurtosis()
        
        # VIX-like measure (volatility of volatility)
        features['vol_of_vol'] = features['volatility'].rolling(window).std()
        
        # Momentum (cumulative return over window)
        features['momentum'] = returns.rolling(window).sum()
        
        # Fill missing values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def detect_regimes_hmm(self, features: pd.DataFrame) -> np.array:
        """
        Detect regimes using Hidden Markov Model.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of regime labels
        """
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, falling back to GMM")
            return self.detect_regimes_gmm(features)
        
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes, 
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            # Fit model
            self.model.fit(features_scaled)
            
            # Predict regimes
            regimes = self.model.predict(features_scaled)
            
            logger.info(f"HMM regime detection completed with {self.n_regimes} regimes")
            return regimes
            
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            logger.info("Falling back to GMM method")
            return self.detect_regimes_gmm(features)
    
    def detect_regimes_gmm(self, features: pd.DataFrame) -> np.array:
        """
        Detect regimes using Gaussian Mixture Model.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of regime labels
        """
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit Gaussian Mixture Model
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            
            # Fit and predict
            regimes = self.model.fit_predict(features_scaled)
            
            logger.info(f"GMM regime detection completed with {self.n_regimes} regimes")
            return regimes
            
        except Exception as e:
            logger.error(f"GMM regime detection failed: {e}")
            return self.detect_regimes_volatility(features)
    
    def detect_regimes_volatility(self, features: pd.DataFrame) -> np.array:
        """
        Simple volatility-based regime detection.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of regime labels
        """
        try:
            # Use volatility for simple regime classification
            volatility = features['volatility']
            
            # Define quantile-based thresholds
            if self.n_regimes == 2:
                threshold = volatility.quantile(0.5)
                regimes = (volatility > threshold).astype(int)
            elif self.n_regimes == 3:
                low_threshold = volatility.quantile(0.33)
                high_threshold = volatility.quantile(0.67)
                regimes = np.where(volatility <= low_threshold, 0,
                                 np.where(volatility <= high_threshold, 1, 2))
            else:
                # For more regimes, use equal-width quantiles
                quantiles = np.linspace(0, 1, self.n_regimes + 1)
                thresholds = volatility.quantile(quantiles[1:-1])
                regimes = pd.cut(volatility, 
                               bins=[-np.inf] + thresholds.tolist() + [np.inf],
                               labels=False)
            
            logger.info(f"Volatility-based regime detection completed")
            return regimes
            
        except Exception as e:
            logger.error(f"Volatility regime detection failed: {e}")
            # Return simple alternating pattern as fallback
            return np.tile(np.arange(self.n_regimes), 
                          len(features) // self.n_regimes + 1)[:len(features)]
    
    def fit_predict(self, returns: pd.Series, feature_window: int = 20) -> np.array:
        """
        Fit regime detection model and predict regimes.
        
        Args:
            returns: Time series of returns
            feature_window: Window for feature calculation
            
        Returns:
            Array of regime labels
        """
        # Prepare features
        features = self._prepare_features(returns, feature_window)
        
        # Choose detection method
        if self.method == "hmm":
            regimes = self.detect_regimes_hmm(features)
        elif self.method == "gmm":
            regimes = self.detect_regimes_gmm(features)
        elif self.method == "volatility":
            regimes = self.detect_regimes_volatility(features)
        else:
            logger.warning(f"Unknown method {self.method}, using GMM")
            regimes = self.detect_regimes_gmm(features)
        
        return regimes
    
    def get_regime_statistics(self, returns: pd.Series, regimes: np.array) -> pd.DataFrame:
        """
        Compute statistics for each regime.
        
        Args:
            returns: Time series of returns
            regimes: Array of regime labels
            
        Returns:
            DataFrame with regime statistics
        """
        regime_stats = []
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) == 0:
                continue
            
            stats = {
                'regime': regime,
                'regime_name': self.regime_names.get(regime, f"Regime {regime}"),
                'count': len(regime_returns),
                'frequency': len(regime_returns) / len(returns),
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'min_return': regime_returns.min(),          
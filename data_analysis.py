import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

def perform_pca(returns, n_components=None, explain_variance=0.95):
    """
    Perform Principal Component Analysis (PCA) on returns data.
    @param returns: DataFrame of returns
    @param n_components: Number of components to retain (optional)
    @param explain_variance: Target explained variance ratio (default: 0.95)
    @return: principal components, explained variance ratios, and factor loadings
    """
    # Standardize the data
    scaler = StandardScaler()
    standardized_returns = scaler.fit_transform(returns)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(standardized_returns)

    # Determine the number of components to explain the target variance
    if n_components is None:
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= explain_variance) + 1
        pca = PCA(n_components=n_components)
        pca.fit(standardized_returns)

    principal_components = pca.transform(standardized_returns)
    explained_variance_ratios = pca.explained_variance_ratio_
    factor_loadings = pca.components_

    return principal_components, explained_variance_ratios, factor_loadings

def detect_market_regimes(returns, n_regimes=2, method='hmm'):
    """
    Detect market regimes using Hidden Markov Models (HMM).
    @param returns: DataFrame of returns
    @param n_regimes: Number of regimes to detect (default: 2)
    @param method: Detection method (default: 'hmm')
    @return: regime classifications and transition probabilities
    """
    if method != 'hmm':
        raise ValueError("Currently, only 'hmm' method is supported.")

    # Standardize the data
    scaler = StandardScaler()
    standardized_returns = scaler.fit_transform(returns)

    # Fit HMM
    hmm = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000)
    hmm.fit(standardized_returns)

    # Predict regimes
    regimes = hmm.predict(standardized_returns)
    transition_probabilities = hmm.transmat_

    return regimes, transition_probabilities

def calculate_risk_metrics(returns, weights, alpha=0.05):
    """
    Calculate portfolio risk metrics.
    @param returns: DataFrame of returns
    @param weights: Portfolio weights
    @param alpha: Confidence level for VaR and CVaR (default: 0.05)
    @return: VaR, CVaR, volatility
    """
    # Portfolio returns
    portfolio_returns = returns.dot(weights)

    # Portfolio volatility
    volatility = np.std(portfolio_returns)

    # Value at Risk (VaR)
    var = np.percentile(portfolio_returns, alpha * 100)

    # Conditional Value at Risk (CVaR)
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    return {
        'VaR': var,
        'CVaR': cvar,
        'Volatility': volatility
    }
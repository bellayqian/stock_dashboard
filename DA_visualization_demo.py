"""
Data Analysis Visualization Demo

This script demonstrates how to use the data analysis visualization functions
to analyze stock market data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_analysis import *
from data_analysis_visualization import *

# Set plot style
plt.style.use('ggplot')

def run_pca_analysis(returns_data, output_dir=None):
    """
    Run PCA analysis and generate visualizations.
    
    Parameters:
    ----------
    returns_data : pandas.DataFrame
        DataFrame of stock returns
    output_dir : str, optional
        Directory to save output figures
    """
    print("=== PCA Analysis ===")
    
    # Get tickers
    tickers = returns_data.columns.tolist()
    
    # Perform PCA on the returns data
    principal_components, explained_variance_ratios, factor_loadings = perform_pca(returns_data)
    
    print(f"Explained Variance Ratios: {explained_variance_ratios}")
    print(f"Principal Components Shape: {principal_components.shape}")
    
    # View the principal components as a DataFrame
    components_df = pd.DataFrame(
        principal_components, 
        index=returns_data.index, 
        columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
    )
    print("\nPrincipal Components (first 5 rows):")
    print(components_df.head())
    
    # View the factor loadings as a DataFrame
    factor_loadings_df = pd.DataFrame(
        factor_loadings, 
        columns=returns_data.columns, 
        index=[f'PC{i+1}' for i in range(factor_loadings.shape[0])]
    )
    print("\nFactor Loadings (first 5 components):")
    print(factor_loadings_df.head())
    
    # Create visualizations
    print("\nGenerating PCA visualizations...")
    
    # 1. PCA biplot (keeping this one)
    fig1 = plot_pca_biplot(principal_components, factor_loadings, tickers)
    if output_dir:
        fig1.savefig(os.path.join(output_dir, 'pca_biplot.png'))
    
    # 2. PCA clustering plot (adding this one)
    fig2 = plot_pca_clustering(principal_components, factor_loadings, tickers)
    if output_dir:
        fig2.savefig(os.path.join(output_dir, 'pca_clustering.png'))

    # 3. Interactive PCA clustering plot (adding this one)
    fig3 = create_pca_clustering_plot(principal_components, factor_loadings, tickers)
    if output_dir:
        fig3.write_html(os.path.join(output_dir, 'interactive_pca_clustering.html'))

    print("PCA analysis complete.")
    return {
        'principal_components': principal_components,
        'explained_variance_ratios': explained_variance_ratios,
        'factor_loadings': factor_loadings
    }

def run_market_regime_analysis(returns_data, n_regimes=3, output_dir=None):
    """
    Run market regime analysis and generate visualizations.
    
    Parameters:
    ----------
    returns_data : pandas.DataFrame
        DataFrame of stock returns
    n_regimes : int, optional
        Number of regimes to detect
    output_dir : str, optional
        Directory to save output figures
    """
    print("\n=== Market Regime Analysis ===")
    
    # Detect market regimes using HMM
    regimes, transition_probabilities = detect_market_regimes(returns_data, n_regimes=n_regimes)
    
    print(f"Regime Counts: {np.bincount(regimes)}")
    print(f"Transition Probabilities:\n{transition_probabilities}")
    
    # Create visualizations
    print("\nGenerating market regime visualizations...")
    
    # 1. Regime characteristics (keeping this one)
    fig1 = plot_regime_characteristics(returns_data, regimes)
    if output_dir:
        fig1.savefig(os.path.join(output_dir, 'regime_characteristics.png'))
    
    # 2. Interactive market regimes timeline (keeping this one)
    fig2 = create_market_regimes_timeline(returns_data, regimes)
    if output_dir:
        fig2.write_html(os.path.join(output_dir, 'interactive_market_regimes.html'))
    
    print("Market regime analysis complete.")
    return {
        'regimes': regimes,
        'transition_probabilities': transition_probabilities
    }

def run_risk_metrics_analysis(returns_data, output_dir=None):
    """
    Run risk metrics analysis and generate visualizations.
    
    Parameters:
    ----------
    returns_data : pandas.DataFrame
        DataFrame of stock returns
    output_dir : str, optional
        Directory to save output figures
    """
    print("\n=== Risk Metrics Analysis ===")
    
    # Get tickers
    tickers = returns_data.columns.tolist()
    
    # Create different portfolio weights
    n_assets = len(tickers)
    
    # 1. Equal weights
    equal_weights = np.ones(n_assets) / n_assets
    
    # 2. Random weights
    np.random.seed(42)  # For reproducibility
    random_weights = np.random.random(n_assets)
    random_weights /= np.sum(random_weights)
    
    # 3. Tech-heavy weights (assuming first 25% of tickers are tech)
    tech_weights = np.zeros(n_assets)
    tech_count = max(1, int(n_assets * 0.25))
    tech_weights[:tech_count] = 0.8 / tech_count
    tech_weights[tech_count:] = 0.2 / (n_assets - tech_count)
    
    # Calculate risk metrics
    weights_list = [equal_weights, random_weights, tech_weights]
    labels = ['Equal Weights', 'Random Weights', 'Tech-Heavy']
    
    print("Example risk metrics for equally weighted portfolio:")
    equal_metrics = calculate_risk_metrics(returns_data, equal_weights, alpha=0.05)
    print(f"Value at Risk (VaR): {equal_metrics['VaR']:.4f}")
    print(f"Conditional Value at Risk (CVaR): {equal_metrics['CVaR']:.4f}")
    print(f"Volatility: {equal_metrics['Volatility']:.4f}")
    
    # Create visualizations
    print("\nGenerating risk metrics visualizations...")
    
    # 1. Interactive risk metrics comparison (keeping this one)
    fig1 = create_risk_metrics_comparison(returns_data, weights_list, labels)
    if output_dir:
        fig1.write_html(os.path.join(output_dir, 'interactive_risk_metrics_comparison.html'))
    
    # 2. Interactive risk metrics heatmap (keeping this one)
    fig2 = create_risk_metrics_heatmap(returns_data, tickers)
    if output_dir:
        fig2.write_html(os.path.join(output_dir, 'interactive_risk_metrics_heatmap.html'))
    
    print("Risk metrics analysis complete.")
    return {
        'equal_metrics': equal_metrics,
        'weights_list': weights_list,
        'labels': labels
    }

def run_network_analysis(returns_data, threshold=0.34, output_dir=None):
    """
    Run network analysis and generate visualizations.
    
    Parameters:
    ----------
    returns_data : pandas.DataFrame
        DataFrame of stock returns
    threshold : float, optional
        Correlation threshold for network
    output_dir : str, optional
        Directory to save output figures
    """
    print("\n=== Network Analysis ===")
    
    # Create interactive network visualization
    fig = create_interactive_network(returns_data, threshold)
    if output_dir:
        fig.write_html(os.path.join(output_dir, 'interactive_network.html'))
    
    print("Network analysis complete.")
    return {}

def main():
    """Main function to run the analysis"""
    print("Starting data analysis visualization demo...")
    
    # Create output directory if it doesn't exist
    output_dir = 'output_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the example data
    data_path = 'example_data/50_daily_returns.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
    
    print(f"Loading returns data from {data_path}...")
    returns_data = pd.read_csv(data_path, index_col=0)
    print(f"Loaded data with shape {returns_data.shape}")
    
    # Run PCA analysis
    pca_results = run_pca_analysis(returns_data, output_dir)
    
    # Run market regime analysis
    regime_results = run_market_regime_analysis(returns_data, n_regimes=3, output_dir=output_dir)
    
    # Run risk metrics analysis
    risk_results = run_risk_metrics_analysis(returns_data, output_dir)
    
    # Run network analysis
    network_results = run_network_analysis(returns_data, threshold=0.34, output_dir=output_dir)
    
    print("\nAll analyses complete. Output figures saved to:", output_dir)
    
    # Display some plots (uncomment to show plots during execution)
    # plt.show()
    
    return {
        'pca': pca_results,
        'regimes': regime_results,
        'risk': risk_results,
        'network': network_results
    }

if __name__ == "__main__":
    results = main()
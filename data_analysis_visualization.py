"""
Data Analysis Visualization Module
=================================

Functions to visualize results from data analysis operations including:
- PCA (Principal Component Analysis)
- Market Regime Detection
- Risk Metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.cluster import hierarchy
from data_analysis import perform_pca, detect_market_regimes, calculate_risk_metrics

def plot_explained_variance(explained_variance_ratios, title=None):
    """
    Create a plot showing explained variance ratios from PCA.
    
    Parameters:
    ----------
    explained_variance_ratios : array-like
        Explained variance ratios from PCA
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the explained variance plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create x-axis labels
    x_labels = [f'PC{i+1}' for i in range(len(explained_variance_ratios))]
    
    # Plot individual and cumulative explained variance
    ax.bar(x_labels, explained_variance_ratios, alpha=0.7, label='Individual')
    ax.plot(x_labels, np.cumsum(explained_variance_ratios), marker='o', 
            color='red', label='Cumulative')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    # Add grid, labels, and title
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Explained Variance by Principal Component')
    
    # Add legend
    ax.legend()
    
    # Add 95% variance threshold line
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Threshold')
    
    # Annotate the number of components needed for 95% variance
    cumulative = np.cumsum(explained_variance_ratios)
    n_components_95 = np.argmax(cumulative >= 0.95) + 1
    ax.annotate(f'{n_components_95} PCs explain\n95% of variance', 
                xy=(n_components_95-1, cumulative[n_components_95-1]),
                xytext=(n_components_95+1, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_factor_loadings_heatmap(factor_loadings, tickers, n_components=None, title=None):
    """
    Create a heatmap of factor loadings from PCA.
    
    Parameters:
    ----------
    factor_loadings : array-like
        Factor loadings from PCA
    tickers : list
        List of ticker symbols
    n_components : int, optional
        Number of components to display (defaults to all)
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the factor loadings heatmap
    """
    # Limit to specified number of components
    if n_components is not None:
        factor_loadings = factor_loadings[:n_components]
    
    # Create a DataFrame for better display
    loadings_df = pd.DataFrame(
        factor_loadings, 
        columns=tickers,
        index=[f'PC{i+1}' for i in range(factor_loadings.shape[0])]
    )
    
    # Determine figure size based on number of tickers
    figsize = (min(20, max(10, len(tickers) / 3)), max(8, n_components * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(loadings_df, cmap='coolwarm', center=0, annot=False, 
                linewidths=0.5, ax=ax)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Factor Loadings Heatmap')
    
    # Format x-axis labels
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    return fig

def plot_pca_clustering(principal_components, factor_loadings, tickers, n_clusters=3, title=None):
    """
    Create a scatter plot of PC1 vs PC2 with clustering.
    
    Parameters:
    ----------
    principal_components : array-like
        Principal components from PCA
    factor_loadings : array-like
        Factor loadings from PCA
    tickers : list
        List of ticker symbols
    n_clusters : int, optional
        Number of clusters to identify
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the PCA clustering plot
    """
    from sklearn.cluster import KMeans
    
    # Use only the first two principal components
    # Create DataFrame for the ticker loadings on PC1 and PC2
    pc_df = pd.DataFrame({
        'PC1': factor_loadings[0, :],  # First PC loadings
        'PC2': factor_loadings[1, :],  # Second PC loadings
        'Ticker': tickers
    })
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pc_df[['PC1', 'PC2']])
    pc_df['Cluster'] = clusters
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot clusters with different colors
    for cluster in range(n_clusters):
        cluster_data = pc_df[pc_df['Cluster'] == cluster]
        ax.scatter(
            cluster_data['PC1'], 
            cluster_data['PC2'],
            alpha=0.7, 
            label=f'Cluster {cluster+1}'
        )
    
    # Add ticker labels
    for i, ticker in enumerate(tickers):
        ax.annotate(
            ticker, 
            (pc_df.iloc[i]['PC1'], pc_df.iloc[i]['PC2']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # Add grid, labels, and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Principal Component 1 Loadings')
    ax.set_ylabel('Principal Component 2 Loadings')
    ax.legend()
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('PCA Clustering of Stocks by Factor Loadings')
    
    plt.tight_layout()
    return fig

def plot_pca_biplot(principal_components, factor_loadings, tickers, title=None):
    """
    Create a biplot showing both the principal components and factor loadings.
    
    Parameters:
    ----------
    principal_components : array-like
        Principal components from PCA
    factor_loadings : array-like
        Factor loadings from PCA
    tickers : list
        List of ticker symbols
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the biplot
    """
    # Use only the first two principal components
    pc1 = principal_components[:, 0]
    pc2 = principal_components[:, 1]
    
    # Scale the factor loadings for visualization
    n = factor_loadings.shape[1]
    scale = np.abs(principal_components).max() / np.abs(factor_loadings).max() * 0.7
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the principal components
    ax.scatter(pc1, pc2, alpha=0.3, color='blue')
    
    # Plot the factor loadings as vectors
    for i, ticker in enumerate(tickers):
        ax.arrow(0, 0, factor_loadings[0, i] * scale, factor_loadings[1, i] * scale,
                 head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(factor_loadings[0, i] * scale * 1.1, factor_loadings[1, i] * scale * 1.1,
                ticker, color='red', ha='center', va='center')
    
    # Add a circle to represent correlations of 1
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='gray', alpha=0.5)
    ax.add_patch(circle)
    
    # Set plot limits
    max_val = np.max(np.abs(pc1)), np.max(np.abs(pc2))
    limit = max(max_val) * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add axis labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('PCA Biplot: Stocks and Principal Components')
    
    plt.tight_layout()
    return fig

def plot_market_regimes_timeline(returns, regimes, title=None):
    """
    Create a timeline plot showing market regimes and returns.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    regimes : array-like
        Regime classifications from detect_market_regimes
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the market regimes timeline
    """
    # Calculate portfolio returns (equally weighted)
    portfolio_returns = returns.mean(axis=1)
    
    # Create a DataFrame with dates, portfolio returns, and regimes
    regimes_df = pd.DataFrame({
        'Date': returns.index,
        'Returns': portfolio_returns.values,
        'Regime': regimes
    })
    
    # Count regimes
    n_regimes = len(np.unique(regimes))
    
    # Create a color map for regimes
    cmap = plt.cm.get_cmap('viridis', n_regimes)
    regime_colors = {i: cmap(i) for i in range(n_regimes)}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot portfolio returns
    ax1.plot(regimes_df['Date'], regimes_df['Returns'], color='black', alpha=0.7)
    
    # Highlight different regimes with background colors
    regime_changes = np.where(np.diff(regimes) != 0)[0]
    regime_dates = [regimes_df['Date'].iloc[0]] + [regimes_df['Date'].iloc[i+1] for i in regime_changes] + [regimes_df['Date'].iloc[-1]]
    regime_values = [regimes[0]] + [regimes[i+1] for i in regime_changes] + [regimes[-1]]
    
    for i in range(len(regime_dates)-1):
        ax1.axvspan(regime_dates[i], regime_dates[i+1], 
                   alpha=0.3, color=regime_colors[regime_values[i]])
    
    # Set labels and title for returns plot
    ax1.set_ylabel('Portfolio Returns')
    ax1.grid(True, linestyle='--', alpha=0.7)
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Market Regimes and Portfolio Returns')
    
    # Plot regimes as a heatmap
    dates = pd.to_datetime(regimes_df['Date'])
    for i in range(len(dates)-1):
        ax2.axvspan(dates[i], dates[i+1], alpha=1.0, color=regime_colors[regimes[i]])
    
    # Set labels for regimes plot
    ax2.set_ylabel('Regime')
    ax2.set_xlabel('Date')
    
    # Create a custom legend for regimes
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=regime_colors[i], 
                                    alpha=0.7, label=f'Regime {i+1}') 
                      for i in range(n_regimes)]
    ax2.legend(handles=legend_elements, loc='lower right', ncol=n_regimes)
    
    plt.tight_layout()
    return fig

def plot_regime_transition_diagram(transition_matrix, title=None):
    """
    Create a diagram showing transitions between market regimes.
    
    Parameters:
    ----------
    transition_matrix : array-like
        Transition probability matrix from detect_market_regimes
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the transition diagram
    """
    n_regimes = transition_matrix.shape[0]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_regimes):
        G.add_node(i, label=f'Regime {i+1}')
    
    # Add edges with weights
    for i in range(n_regimes):
        for j in range(n_regimes):
            if transition_matrix[i, j] > 0:
                G.add_edge(i, j, weight=transition_matrix[i, j])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define node positions (circular layout)
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000, 
                          node_color='lightblue', alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: f'Regime {i+1}' for i in range(n_regimes)},
                           font_size=14, font_weight='bold')
    
    # Draw edges with varying thickness based on transition probability
    for u, v, data in G.edges(data=True):
        width = data['weight'] * 10  # Scale for visibility
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=width,
                              alpha=0.7, edge_color='gray', 
                              connectionstyle='arc3,rad=0.1',
                              arrowsize=20)
    
    # Draw edge labels (transition probabilities)
    edge_labels = {(u, v): f'{data["weight"]:.2f}' for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                font_size=12, label_pos=0.3)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title('Market Regime Transition Diagram', fontsize=16)
    
    # Remove axis
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_regime_characteristics(returns, regimes, title=None):
    """
    Create visualizations showing characteristics of different market regimes.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    regimes : array-like
        Regime classifications from detect_market_regimes
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing regime characteristics
    """
    # Calculate portfolio returns (equally weighted)
    portfolio_returns = returns.mean(axis=1)
    
    # Create a DataFrame with returns and regimes
    data = pd.DataFrame({'Returns': portfolio_returns.values, 'Regime': regimes})
    
    # Count regimes
    n_regimes = len(np.unique(regimes))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Distribution of returns in each regime
    sns.boxplot(x='Regime', y='Returns', data=data, ax=axes[0])
    axes[0].set_title('Distribution of Returns by Regime')
    axes[0].set_xlabel('Regime')
    axes[0].set_ylabel('Returns')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Mean and volatility by regime
    regime_stats = data.groupby('Regime')['Returns'].agg(['mean', 'std']).reset_index()
    
    bar_width = 0.35
    x = np.arange(n_regimes)
    
    axes[1].bar(x - bar_width/2, regime_stats['mean'], bar_width, label='Mean Return', alpha=0.7)
    axes[1].bar(x + bar_width/2, regime_stats['std'], bar_width, label='Volatility', alpha=0.7)
    
    axes[1].set_xlabel('Regime')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Mean Return and Volatility by Regime')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Regime {i+1}' for i in range(n_regimes)])
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Regime duration histogram
    regime_runs = []
    current_regime = regimes[0]
    run_length = 1
    
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            run_length += 1
        else:
            regime_runs.append((current_regime, run_length))
            current_regime = regimes[i]
            run_length = 1
    
    # Add the last run
    regime_runs.append((current_regime, run_length))
    
    # Create a DataFrame of runs
    runs_df = pd.DataFrame(regime_runs, columns=['Regime', 'Duration'])
    
    # Plot histogram for each regime
    for i in range(n_regimes):
        regime_durations = runs_df[runs_df['Regime'] == i]['Duration']
        axes[2].hist(regime_durations, alpha=0.7, label=f'Regime {i+1}', bins=10)
    
    axes[2].set_xlabel('Duration (Days)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Regime Durations')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Market Regime Characteristics', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_risk_metrics_comparison(returns, weights_list, labels, alpha=0.05, title=None):
    """
    Create bar charts comparing risk metrics for different portfolios.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    weights_list : list of arrays
        List of portfolio weights arrays
    labels : list of str
        Labels for each portfolio
    alpha : float, optional
        Confidence level for VaR and CVaR
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing risk metrics comparison
    """
    # Calculate risk metrics for each portfolio
    risk_metrics_list = []
    
    for weights in weights_list:
        metrics = calculate_risk_metrics(returns, weights, alpha)
        risk_metrics_list.append(metrics)
    
    # Prepare data for plotting
    metrics_df = pd.DataFrame(risk_metrics_list, index=labels)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot VaR
    axes[0].bar(metrics_df.index, metrics_df['VaR'].abs(), alpha=0.7)
    axes[0].set_title(f'Value at Risk (VaR) at {alpha*100}% Level')
    axes[0].set_ylabel('Absolute Value')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot CVaR
    axes[1].bar(metrics_df.index, metrics_df['CVaR'].abs(), alpha=0.7)
    axes[1].set_title(f'Conditional VaR (CVaR) at {alpha*100}% Level')
    axes[1].set_ylabel('Absolute Value')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Volatility
    axes[2].bar(metrics_df.index, metrics_df['Volatility'], alpha=0.7)
    axes[2].set_title('Volatility (Standard Deviation)')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Risk Metrics Comparison', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_risk_metrics_heatmap(returns, tickers, alpha=0.05, title=None):
    """
    Create a heatmap of risk metrics for individual stocks.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    tickers : list
        List of ticker symbols
    alpha : float, optional
        Confidence level for VaR and CVaR
    title : str, optional
        Title for the plot
        
    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing risk metrics heatmap
    """
    # Calculate risk metrics for each individual stock
    risk_data = {}
    
    for ticker in tickers:
        # Create a weight vector with 1 for the current ticker and 0 for others
        weights = np.zeros(len(tickers))
        weights[tickers.index(ticker)] = 1
        
        # Calculate risk metrics
        metrics = calculate_risk_metrics(returns, weights, alpha)
        
        # Store the metrics
        risk_data[ticker] = {
            'VaR': metrics['VaR'],
            'CVaR': metrics['CVaR'],
            'Volatility': metrics['Volatility']
        }
    
    # Create DataFrames for each metric
    var_df = pd.DataFrame({ticker: risk_data[ticker]['VaR'] for ticker in tickers}, index=['VaR'])
    cvar_df = pd.DataFrame({ticker: risk_data[ticker]['CVaR'] for ticker in tickers}, index=['CVaR'])
    vol_df = pd.DataFrame({ticker: risk_data[ticker]['Volatility'] for ticker in tickers}, index=['Volatility'])
    
    # Combine into a single DataFrame
    metrics_df = pd.concat([var_df, cvar_df, vol_df])
    
    # Determine figure size based on number of tickers
    figsize = (min(20, max(10, len(tickers) / 3)), 5)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(metrics_df, cmap='YlOrRd', annot=True, fmt='.4f', 
                linewidths=0.5, ax=ax)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Risk Metrics by Stock (alpha={alpha})')
    
    # Format x-axis labels
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    return fig

# Interactive Plotly visualizations for integration with dashboard

def create_pca_clustering_plot(principal_components, factor_loadings, tickers, n_clusters=3):
    """
    Create an interactive scatter plot of PC1 vs PC2 with clustering using Plotly.
    
    Parameters:
    ----------
    principal_components : array-like
        Principal components from PCA
    factor_loadings : array-like
        Factor loadings from PCA
    tickers : list
        List of ticker symbols
    n_clusters : int, optional
        Number of clusters to identify
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive PCA clustering plot
    """
    from sklearn.cluster import KMeans
    
    # Use only the first two principal components loadings
    pc_df = pd.DataFrame({
        'PC1': factor_loadings[0, :],
        'PC2': factor_loadings[1, :],
        'Ticker': tickers
    })
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pc_df[['PC1', 'PC2']])
    pc_df['Cluster'] = clusters
    
    # Create figure
    fig = px.scatter(
        pc_df, x='PC1', y='PC2', 
        color='Cluster', 
        text='Ticker',
        color_discrete_sequence=px.colors.qualitative.G10,
        hover_data={'PC1': ':.3f', 'PC2': ':.3f', 'Ticker': True, 'Cluster': True}
    )
    
    # Update layout
    fig.update_layout(
        title="PCA Clustering of Stocks by Factor Loadings",
        xaxis_title="Principal Component 1 Loadings",
        yaxis_title="Principal Component 2 Loadings",
        legend_title="Cluster",
        height=600,
        template="plotly_white"
    )
    
    # Update marker size and add text
    fig.update_traces(
        marker=dict(size=12),
        textposition='top center'
    )
    
    return fig

def create_pca_explained_variance_plot(explained_variance_ratios):
    """
    Create an interactive plot showing explained variance from PCA using Plotly.
    
    Parameters:
    ----------
    explained_variance_ratios : array-like
        Explained variance ratios from PCA
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive explained variance plot
    """
    # Create x-axis labels
    x_labels = [f'PC{i+1}' for i in range(len(explained_variance_ratios))]
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratios)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add individual explained variance bars
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=explained_variance_ratios,
            name="Individual",
            marker_color='rgb(55, 83, 109)',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Add cumulative variance line
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=cumulative_variance,
            name="Cumulative",
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            line=dict(width=2, color='red')
        ),
        secondary_y=True
    )
    
    # Add 95% threshold line
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=[0.95] * len(x_labels),
            name="95% Threshold",
            mode='lines',
            line=dict(width=2, color='green', dash='dash')
        ),
        secondary_y=True
    )
    
    # Determine the number of components needed for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    # Add annotation
    fig.add_annotation(
        x=x_labels[n_components_95-1],
        y=cumulative_variance[n_components_95-1],
        text=f"{n_components_95} PCs explain<br>95% of variance",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        ax=50,
        ay=-30
    )
    
    # Update layout
    fig.update_layout(
        title="Explained Variance by Principal Component",
        xaxis=dict(title="Principal Components"),
        yaxis=dict(
            title="Individual Explained Variance",
            titlefont=dict(color="rgb(55, 83, 109)"),
            tickfont=dict(color="rgb(55, 83, 109)"),
            tickformat=".1%"
        ),
        yaxis2=dict(
            title="Cumulative Explained Variance",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            tickformat=".1%",
            range=[0, 1.05]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        template="plotly_white"
    )
    
    return fig

def create_pca_factor_loadings_heatmap(factor_loadings, tickers, n_components=5):
    """
    Create an interactive heatmap of PCA factor loadings using Plotly.
    
    Parameters:
    ----------
    factor_loadings : array-like
        Factor loadings from PCA
    tickers : list
        List of ticker symbols
    n_components : int, optional
        Number of components to display
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive factor loadings heatmap
    """
    # Limit to specified number of components
    factor_loadings_display = factor_loadings[:n_components]
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=factor_loadings_display,
        x=tickers,
        y=[f'PC{i+1}' for i in range(n_components)],
        colorscale='RdBu_r',
        zmid=0,
        text=[[f'{val:.3f}' for val in row] for row in factor_loadings_display],
        hovertemplate='PC: %{y}<br>Stock: %{x}<br>Loading: %{text}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="PCA Factor Loadings Heatmap",
        xaxis=dict(title="Stocks", tickangle=-45),
        yaxis=dict(title="Principal Components"),
        height=500,
        width=1000,
        template="plotly_white"
    )
    
    return fig

def create_market_regimes_timeline(returns, regimes):
    """
    Create an interactive timeline plot showing market regimes using Plotly.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    regimes : array-like
        Regime classifications from detect_market_regimes
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive market regimes timeline
    """
    # Calculate portfolio returns (equally weighted)
    portfolio_returns = returns.mean(axis=1)
    
    # Create a DataFrame with dates, portfolio returns, and regimes
    dates = pd.to_datetime(returns.index)
    regimes_df = pd.DataFrame({
        'Date': dates,
        'Returns': portfolio_returns.values,
        'Regime': regimes
    })
    
    # Count regimes
    n_regimes = len(np.unique(regimes))
    
    # Create a color map for regimes
    colors = px.colors.qualitative.D3[:n_regimes]
    
    # Create figure with two subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Add portfolio returns trace
    fig.add_trace(
        go.Scatter(
            x=regimes_df['Date'],
            y=regimes_df['Returns'],
            mode='lines',
            name='Portfolio Returns',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    # Add regime background colors
    for i in range(n_regimes):
        # Filter data for this regime
        regime_data = regimes_df[regimes_df['Regime'] == i]
        
        # Skip if no data for this regime
        if regime_data.empty:
            continue
        
        # Group consecutive dates for this regime
        regime_data.loc[:, 'date_diff'] = (regime_data['Date'].diff() > pd.Timedelta(days=1)).cumsum()
        groups = regime_data.groupby('date_diff')
        
        for _, group in groups:
            if len(group) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[group['Date'].min(), group['Date'].min(), 
                           group['Date'].max(), group['Date'].max()],
                        y=[min(portfolio_returns) * 1.1, max(portfolio_returns) * 1.1, 
                           max(portfolio_returns) * 1.1, min(portfolio_returns) * 1.1],
                        fill="toself",
                        fillcolor=colors[i],
                        line=dict(width=0),
                        showlegend=False,
                        opacity=0.3,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
    
    # Add regime heatmap
    for i in range(n_regimes):
        # Filter data for this regime
        regime_data = regimes_df[regimes_df['Regime'] == i]
        
        # Skip if no data for this regime
        if regime_data.empty:
            continue
        
        # Group consecutive dates
        regime_data.loc[:, 'date_diff'] = (regime_data['Date'].diff() > pd.Timedelta(days=1)).cumsum()
        groups = regime_data.groupby('date_diff')
        
        # Keep track of group indices for regime i
        prev_groups = []
        for group_idx, (group_key, group) in enumerate(groups):
            prev_groups.append((group_key, group))
            
            # Only add the regime to legend once
            add_to_legend = (group_idx == 0)
            
            if len(group) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[group['Date'].min(), group['Date'].max()],
                        y=[i, i],
                        mode='lines',
                        line=dict(color=colors[i], width=20),
                        name=f'Regime {i+1}' if add_to_legend else "",
                        hoverinfo='name',
                        showlegend=add_to_legend
                    ),
                    row=2, col=1
                )
    
    # Update layout
    fig.update_layout(
        title="Market Regimes and Portfolio Returns",
        xaxis_title="Date",
        yaxis_title="Returns",
        yaxis2=dict(
            title="Regime",
            range=[-0.5, n_regimes-0.5],
            tickvals=list(range(n_regimes)),
            ticktext=[f'Regime {i+1}' for i in range(n_regimes)]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        template="plotly_white"
    )
    
    return fig
    
def create_risk_metrics_comparison(returns, weights_list, labels, alpha=0.05):
    """
    Create an interactive bar chart comparing risk metrics using Plotly.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    weights_list : list of arrays
        List of portfolio weights arrays
    labels : list of str
        Labels for each portfolio
    alpha : float, optional
        Confidence level for VaR and CVaR
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive risk metrics comparison
    """
    # Calculate risk metrics for each portfolio
    risk_metrics_list = []
    
    for weights in weights_list:
        metrics = calculate_risk_metrics(returns, weights, alpha)
        risk_metrics_list.append(metrics)
    
    # Create a combined DataFrame
    metrics_df = pd.DataFrame(risk_metrics_list, index=labels)
    
    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=["Value at Risk (VaR)", 
                                      "Conditional VaR (CVaR)",
                                      "Volatility"],
                       shared_yaxes=False)
    
    # Add VaR bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=metrics_df['VaR'].abs(),
            name="VaR",
            marker_color='firebrick',
            opacity=0.7,
            text=metrics_df['VaR'].abs().round(4),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add CVaR bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=metrics_df['CVaR'].abs(),
            name="CVaR",
            marker_color='royalblue',
            opacity=0.7,
            text=metrics_df['CVaR'].abs().round(4),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Add Volatility bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=metrics_df['Volatility'],
            name="Volatility",
            marker_color='darkgreen',
            opacity=0.7,
            text=metrics_df['Volatility'].round(4),
            textposition='auto'
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=f"Risk Metrics Comparison (alpha={alpha*100}%)",
        showlegend=False,
        height=500,
        template="plotly_white"
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Absolute Value", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Value", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=3)
    
    return fig

def create_risk_metrics_heatmap(returns, tickers, alpha=0.05):
    """
    Create an interactive heatmap of risk metrics for individual stocks using Plotly.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    tickers : list
        List of ticker symbols
    alpha : float, optional
        Confidence level for VaR and CVaR
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive risk metrics heatmap
    """
    # Calculate risk metrics for each individual stock
    risk_data = {}
    
    for ticker in tickers:
        # Create a weight vector with 1 for the current ticker and 0 for others
        weights = np.zeros(len(tickers))
        weights[tickers.index(ticker)] = 1
        
        # Calculate risk metrics
        metrics = calculate_risk_metrics(returns, weights, alpha)
        
        # Store the metrics
        risk_data[ticker] = {
            'VaR': metrics['VaR'],
            'CVaR': metrics['CVaR'],
            'Volatility': metrics['Volatility']
        }
    
    # Create DataFrames for each metric
    var_df = pd.DataFrame({ticker: risk_data[ticker]['VaR'] for ticker in tickers}, index=['VaR'])
    cvar_df = pd.DataFrame({ticker: risk_data[ticker]['CVaR'] for ticker in tickers}, index=['CVaR'])
    vol_df = pd.DataFrame({ticker: risk_data[ticker]['Volatility'] for ticker in tickers}, index=['Volatility'])
    
    # Combine into a single DataFrame
    metrics_df = pd.concat([var_df, cvar_df, vol_df])
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=metrics_df.values,
        x=tickers,
        y=metrics_df.index,
        colorscale='YlOrRd',
        text=[[f'{val:.4f}' for val in row] for row in metrics_df.values],
        hovertemplate='Metric: %{y}<br>Stock: %{x}<br>Value: %{text}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Risk Metrics by Stock (alpha={alpha*100}%)",
        xaxis=dict(title="Stocks", tickangle=-45),
        yaxis=dict(title="Risk Metrics"),
        height=500,
        width=1000,
        template="plotly_white"
    )
    
    return fig

def create_interactive_network(returns, threshold=0.34):
    """
    Create an interactive network visualization of stock correlations using Plotly.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame of stock returns
    threshold : float, optional
        Correlation threshold for network edges
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive network visualization
    """
    from data_navigation import build_correlation_network
    
    # Build correlation network
    G = build_correlation_network(returns, threshold)
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, k=0.5)
    
    # Create node trace
    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        marker=dict(
            size=10,
            color='blue',
            line=dict(width=1, color='black')
        ),
        hoverinfo='text'
    )
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G.get_edge_data(edge[0], edge[1]).get('weight', 0)
        edge_text.append(f"{edge[0]} - {edge[1]}: Correlation = {weight:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='text',
        text=edge_text
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title=f'Stock Correlation Network (Corr > {threshold:.2f})',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        width=700,
        template='plotly_white'
    )
    
    return fig
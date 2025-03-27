[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Aem1sI_4)

# Midterm Project: Financial Data Analysis Package

## Overview
This project provides a Python package for analyzing financial data, including tools for:
- Fetching and processing market data.
- Building correlation networks.
- Performing PCA and detecting market regimes.
- Calculating portfolio risk metrics.

Additionally, an interactive web application is available for exploring the data.

1. **Data Scaffolding**:
   - Fetch historical market data for multiple securities using real-time data providers
   - Process and clean financial data for analysis
   - Calculate daily returns and other key metrics
   - Handle missing data and outliers in financial time series

2. **Data Navigation**:
   - Build correlation networks to visualize relationships between securities.
   - Identify connected components and query paths in the network.

3. **Data Analysis**:
   - Perform PCA to extract principal factors driving market movements.
   - Detect market regimes using Hidden Markov Models (HMM).
   - Calculate portfolio risk metrics (VaR, CVaR, volatility).

4. **Visualization**:
   - Visualize correlation networks with interactive stock selection
   - Dynamic dashboard for exploring financial data relationships
   - Interactive PCA component visualization with clustering
   - Market regime timeline visualization with color-coded periods
   - Risk metrics visualization with comparative analysis tools

---

## Installation

### From PyPI
Install the package directly from PyPI:
```bash
pip install financial-analysis-package
```

### From Source
Clone the repository and install the package:
```bash
git clone https://github.com/class-account/midterm-project.git
cd midterm-project
pip install -r requirements.txt
```

---

## Usage

### 1. Data Scaffolding
#### Fetch Market Data
```python
from realtime_market_data import RealTimeMarketData

# Initialize the market data handler
market_data = RealTimeMarketData(source="yahoo")

# Add tickers to track
market_data.add_tickers(['AAPL', 'MSFT', 'GOOGL'])

# Fetch historical data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # 1 year of data

# Get historical price data
price_data = market_data.fetch_market_data(
    start_date=start_date,
    end_date=end_date,
    fields=['Close'],
    frequency='daily'
)

# Calculate returns
daily_returns = market_data.calculate_returns(price_data, method='simple')
```

### 2. Data Navigation
#### Build Correlation Network
```python
from data_navigation import build_correlation_network, connected_component, path_query

# Build a correlation network with threshold 0.5
G = build_correlation_network(daily_returns, threshold=0.5)

# Find connected components
components = connected_component(G)

# Check if there's a path between two stocks
is_connected = path_query(G, 'AAPL', 'MSFT')
```

### 3. Data Analysis
#### Perform PCA
```python
from data_analysis import perform_pca
principal_components, explained_variance_ratios, factor_loadings = perform_pca(daily_returns)
```

#### Detect Market Regimes
```python
from data_analysis import detect_market_regimes
regimes, transition_probabilities = detect_market_regimes(daily_returns, n_regimes=3)
```

#### Calculate Risk Metrics
```python
from data_analysis import calculate_risk_metrics
weights = [0.2, 0.3, 0.5]
risk_metrics = calculate_risk_metrics(daily_returns, weights)
print(risk_metrics)
```

### 4. Interactive Visualization
The interactive dashboard provides a comprehensive visual interface to explore stock market data:

```python
# Run the interactive dashboard
python app.py
```
Dashboard features include:

- Network visualization with adjustable correlation thresholds
- Interactive stock selection and information display
- Real-time data integration and historical charts
- PCA analysis visualization and factor loading exploration
- Market regime detection with color-coded timeline
- Risk metrics comparison with portfolio analysis tools

To generate static visualizations for use with GitHub Pages:
```python
# Generate visualization files
python DA_visualization_demo.py

# Build static site
python build_static.py
```

---

## Interactive Website to Explore Data
Explore the data interactively through our web application. The app allows you to:
   - Visualize stock correlation networks and discover relationships
   - Explore PCA results and factor loadings
   - Detect and analyze market regimes
   - Compare risk metrics across different portfolios
   - Monitor real-time stock data and historical patterns

[Click here to access the web](https://example-app-url.com)

---

## Contributions
Bella Qian: Data Scaffolding + Interactive Visualization

Suhwan Bong: Data Analysis + publishing the package on PyPI

Yibin Xiong: Data Navigation
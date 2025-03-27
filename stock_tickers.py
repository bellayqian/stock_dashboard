# Technology
tech = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "GOOGL", # Alphabet Inc.
    "META",  # Meta Platforms, Inc.
    "NVDA",  # NVIDIA Corporation
    "ADBE",  # Adobe Inc.
    "CSCO",  # Cisco Systems, Inc.
    "INTC",  # Intel Corporation
    "ORCL",  # Oracle Corporation
    "CRM",   # Salesforce, Inc.
    "IBM"    # International Business Machines Corporation
]

# Consumer Discretionary
consumer_discretionary = [
    "AMZN",  # Amazon.com, Inc.
    "TSLA",  # Tesla, Inc.
    "HD",    # The Home Depot, Inc.
    "MCD",   # McDonald’s Corporation
    "NKE",   # NIKE, Inc.
    "SBUX",  # Starbucks Corporation
    "GM"     # General Motors Company
]

# Healthcare
healthcare = [
    "JNJ",   # Johnson & Johnson
    "UNH",   # UnitedHealth Group Incorporated
    "ABT",   # Abbott Laboratories
    "MRK",   # Merck & Co., Inc.
    "AMGN",  # Amgen Inc.
    "PFE",   # Pfizer Inc.
    "CVS"    # CVS Health Corporation
]

# Financials
financials = [
    "BRK-B", # Berkshire Hathaway Inc.
    "JPM",   # JPMorgan Chase & Co.
    "V",     # Visa Inc.
    "MA",    # Mastercard Incorporated
    "PYPL",  # PayPal Holdings, Inc.
    "GS",    # The Goldman Sachs Group, Inc.
    "MS",    # Morgan Stanley
    "WFC"    # Wells Fargo & Company
]

# Consumer Staples
consumer_staples = [
    "PG",    # Procter & Gamble Co.
    "WMT",   # Walmart Inc.
    "KO",    # The Coca-Cola Company
    "PEP"    # PepsiCo, Inc.
]

# Energy
energy = [
    "XOM",   # Exxon Mobil Corporation
    "CVX"    # Chevron Corporation
]

# Industrials
industrials = [
    "BA",    # The Boeing Company
    "MMM",   # 3M Company
    "LMT",   # Lockheed Martin Corporation
    "UNP",   # Union Pacific Corporation
    "CAT",   # Caterpillar Inc.
    "RTX",   # Raytheon Technologies Corporation
    "GE"     # General Electric Company
]

# Communication Services
communication_services = [
    "NFLX",  # Netflix, Inc.
    "T",     # AT&T Inc.
    "VZ"     # Verizon Communications Inc.
]

# Utilities
utilities = [
    "NEE"    # NextEra Energy, Inc.
]

all_tickers = tech + consumer_discretionary + healthcare + \
financials + consumer_staples + energy + industrials + \
communication_services + utilities

if __name__ == '__main__':
    from realtime_market_data import RealTimeMarketData
    import pandas as pd
    from datetime import datetime, timedelta
    # Initialize data handler
    market_data = RealTimeMarketData(source="yahoo")
    market_data.add_tickers(all_tickers)
    # Calculate a reasonable historical period (e.g., 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    # Use the fetch_market_data function (note no tickers parameter)
    price_data = market_data.fetch_market_data(
        start_date=start_date,
        end_date=end_date,
        fields=['Open', 'High', 'Low', 'Close', 'Volume'],
        frequency='daily'
    )
    # Calculate daily returns
    daily_returns = market_data.calculate_returns(price_data, method='simple')
    # Save the returns data to a CSV file for sharing
    daily_returns.to_csv('example_data/50_daily_returns.csv')

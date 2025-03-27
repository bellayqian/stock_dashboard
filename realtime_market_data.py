"""
Real-Time Market Data Module
===========================

Tools for fetching and managing real-time or near real-time stock market data
from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import websocket
import json
import time
import threading
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RealTimeMarketData')

class RealTimeMarketData:
    """
    A class for fetching and managing real-time or near real-time stock market data
    from various sources.
    """
    
    def __init__(self, source="yahoo", api_key=None, update_interval=60):
        """
        Initialize the RealTimeMarketData object.
        
        Parameters:
        ----------
        source : str, optional
            The data source to use ("yahoo", "alphavantage", "iex")
        api_key : str, optional
            API key for premium data sources (required for Alpha Vantage and IEX)
        update_interval : int, optional
            Default interval in seconds for automatic data updates
        """
        self.source = source
        self.api_key = api_key
        self.update_interval = update_interval
        self.data = {}
        self.latest_prices = {}
        self.tickers = []
        self.websocket = None
        self.updating = False
        self.update_thread = None
        
        # Initialize data source clients
        if source == "alphavantage" and api_key:
            self.alpha_client = TimeSeries(key=api_key, output_format='pandas')
        
        logger.info(f"Initialized RealTimeMarketData with source: {source}")
    
    def add_tickers(self, tickers):
        """
        Add tickers to be tracked in real-time.
        
        Parameters:
        ----------
        tickers : list
            List of ticker symbols to track
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        for ticker in tickers:
            if ticker not in self.tickers:
                self.tickers.append(ticker)
                
        logger.info(f"Added tickers: {tickers}")
        logger.info(f"Total tickers being tracked: {len(self.tickers)}")
    
    def _fetch_yahoo_latest(self, tickers):
        """
        Fetch latest data from Yahoo Finance with improved error handling.
        
        Parameters:
        ----------
        tickers : list
            List of ticker symbols
            
        Returns:
        -------
        dict
            Dictionary of latest data for each ticker
        """
        try:
            # Get the current time
            now = datetime.now()
            
            # For intraday data, use a broader time range to ensure we get data
            # Yahoo has a 15-minute delay, so we look back further
            start_time = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            end_time = now.strftime('%Y-%m-%d')
            
            # Fetch data with interval="1d" instead of "1m" for more reliability
            # We'll get today's data only
            daily_data = yf.download(
                tickers=tickers,
                start=start_time, 
                end=end_time,
                interval="1d",  # Changed from "1m" to "1d"
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                progress=False  # Reduce console output
            )
            
            # Process the data
            result = {}
            
            # Handle single ticker case
            if len(tickers) == 1:
                ticker = tickers[0]
                if not daily_data.empty:
                    latest_data = daily_data.iloc[-1]
                    self.latest_prices[ticker] = latest_data['Close']
                    
                    # Calculate change from previous close if available
                    change = 0
                    change_pct = 0
                    if len(daily_data) > 1:
                        prev_close = daily_data.iloc[-2]['Close']
                        change = latest_data['Close'] - prev_close
                        change_pct = (change / prev_close) * 100
                    
                    result[ticker] = {
                        'price': latest_data['Close'],
                        'volume': latest_data['Volume'],
                        'timestamp': daily_data.index[-1],
                        'change': change,
                        'change_pct': change_pct,
                        'history': daily_data
                    }
            else:
                # Handle multiple tickers
                for ticker in tickers:
                    if ticker in daily_data.columns.levels[0]:
                        ticker_data = daily_data[ticker]
                        if not ticker_data.empty:
                            latest_data = ticker_data.iloc[-1]
                            self.latest_prices[ticker] = latest_data['Close']
                            
                            # Calculate change from previous day if available
                            change = 0
                            change_pct = 0
                            if len(ticker_data) > 1:
                                prev_close = ticker_data.iloc[-2]['Close']
                                change = latest_data['Close'] - prev_close
                                change_pct = (change / prev_close) * 100
                            
                            result[ticker] = {
                                'price': latest_data['Close'],
                                'volume': latest_data['Volume'],
                                'timestamp': ticker_data.index[-1],
                                'change': change,
                                'change_pct': change_pct,
                                'history': ticker_data
                            }
            
            # Alternative approach: Get live data for each ticker individually
            if len(result) < len(tickers):
                for ticker in tickers:
                    if ticker not in result:
                        try:
                            # Get real-time quote
                            stock = yf.Ticker(ticker)
                            live_data = stock.history(period="1d")
                            
                            if not live_data.empty:
                                latest_data = live_data.iloc[-1]
                                self.latest_prices[ticker] = latest_data['Close']
                                
                                result[ticker] = {
                                    'price': latest_data['Close'],
                                    'volume': latest_data['Volume'],
                                    'timestamp': live_data.index[-1],
                                    'change': latest_data['Close'] - live_data.iloc[0]['Open'],
                                    'change_pct': (latest_data['Close'] / live_data.iloc[0]['Open'] - 1) * 100,
                                    'history': live_data
                                }
                        except Exception as ticker_e:
                            logger.warning(f"Could not fetch individual data for {ticker}: {str(ticker_e)}")
            
            logger.info(f"Fetched latest data from Yahoo for {len(result)} tickers")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data: {str(e)}")
            # Fallback to individual ticker fetching
            result = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    history = stock.history(period="1d")
                    
                    if not history.empty and 'regularMarketPrice' in info:
                        price = info.get('regularMarketPrice', None)
                        if price:
                            self.latest_prices[ticker] = price
                            
                            # Try to get previous close
                            prev_close = info.get('previousClose', None)
                            change = 0
                            change_pct = 0
                            if prev_close:
                                change = price - prev_close
                                change_pct = (change / prev_close) * 100
                            
                            result[ticker] = {
                                'price': price,
                                'volume': info.get('volume', 0),
                                'timestamp': datetime.now(),
                                'change': change,
                                'change_pct': change_pct,
                                'history': history
                            }
                except Exception as ticker_e:
                    logger.warning(f"Could not fetch data for {ticker}: {str(ticker_e)}")
            
            logger.info(f"Fetched latest data from Yahoo fallback method for {len(result)} tickers")
            return result
    
    def _fetch_alphavantage_latest(self, tickers):
        """
        Fetch latest data from Alpha Vantage.
        
        Parameters:
        ----------
        tickers : list
            List of ticker symbols
            
        Returns:
        -------
        dict
            Dictionary of latest data for each ticker
        """
        if not self.api_key:
            logger.error("API key is required for Alpha Vantage")
            return {}
        
        result = {}
        
        for ticker in tickers:
            try:
                # Alpha Vantage has a rate limit of 5 API calls per minute for free tier
                # So we need to be careful about how many tickers we request
                data, meta_data = self.alpha_client.get_intraday(
                    symbol=ticker,
                    interval='1min',
                    outputsize='compact'
                )
                
                if not data.empty:
                    latest_data = data.iloc[0]  # Alpha Vantage returns newest data first
                    self.latest_prices[ticker] = latest_data['4. close']
                    result[ticker] = {
                        'price': latest_data['4. close'],
                        'volume': latest_data['5. volume'],
                        'timestamp': data.index[0],
                        'change': latest_data['4. close'] - latest_data['1. open'],
                        'change_pct': (latest_data['4. close'] / latest_data['1. open'] - 1) * 100,
                        'history': data
                    }
                
                # Respect Alpha Vantage rate limits
                time.sleep(12)  # Sleep to avoid hitting rate limits
                
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage data for {ticker}: {str(e)}")
        
        logger.info(f"Fetched latest data from Alpha Vantage for {len(result)} tickers")
        return result
    
    def _fetch_iex_latest(self, tickers):
        """
        Fetch latest data from IEX Cloud.
        Note: Implementation placeholder - requires IEX Cloud subscription.
        
        Parameters:
        ----------
        tickers : list
            List of ticker symbols
            
        Returns:
        -------
        dict
            Dictionary of latest data for each ticker
        """
        logger.warning("IEX Cloud implementation requires a subscription")
        # IEX Cloud implementation would go here
        return {}
    
    def fetch_latest_data(self, tickers=None):
        """
        Fetch the latest available data, with fallbacks to multiple sources.
        
        Parameters:
        ----------
        tickers : list, optional
            List of specific tickers to update (defaults to all tracked tickers)
            
        Returns:
        -------
        dict
            Dictionary of latest data for each ticker
        """
        if tickers is None:
            tickers = self.tickers
        elif isinstance(tickers, str):
            tickers = [tickers]
        
        if not tickers:
            logger.warning("No tickers specified for update")
            return {}
        
        # Try primary source
        if self.source == "yahoo":
            result = self._fetch_yahoo_latest(tickers)
            
            # If primary source failed completely, try Alpha Vantage as backup
            if not result and self.api_key:
                logger.info("Yahoo Finance failed, trying Alpha Vantage as backup")
                result = self._fetch_alphavantage_latest(tickers)
        elif self.source == "alphavantage":
            result = self._fetch_alphavantage_latest(tickers)
        elif self.source == "iex":
            result = self._fetch_iex_latest(tickers)
        else:
            logger.error(f"Unknown data source: {self.source}")
            result = {}
            
        return result

    def get_latest_prices_individual(self, tickers=None):
        """
        Get latest prices by querying tickers individually.
        
        Parameters:
        ----------
        tickers : list, optional
            List of tickers to query
            
        Returns:
        -------
        dict
            Dictionary of latest prices and data
        """
        if tickers is None:
            tickers = self.tickers
            
        result = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get basic info
                price = None
                try:
                    # First try to get it from info
                    info = stock.info
                    price = info.get('regularMarketPrice', None)
                    prev_close = info.get('previousClose', None)
                    volume = info.get('volume', 0)
                except:
                    # If that fails, try history
                    history = stock.history(period="1d")
                    if not history.empty:
                        price = history.iloc[-1]['Close']
                        volume = history.iloc[-1]['Volume']
                        prev_close = None
                
                if price:
                    change = 0
                    change_pct = 0
                    if prev_close:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                        
                    self.latest_prices[ticker] = price
                    result[ticker] = {
                        'price': price,
                        'volume': volume,
                        'timestamp': datetime.now(),
                        'change': change,
                        'change_pct': change_pct
                    }
                    
                # Prevent overloading the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error fetching {ticker}: {str(e)}")
                
        return result

    def get_latest_prices(self):
        """
        Get the latest prices for all tracked tickers.
        
        Returns:
        -------
        dict
            Dictionary of latest prices for each ticker
        """
        return self.latest_prices.copy()

    def fetch_market_data(self, tickers=None, start_date=None, end_date=None, fields=None, frequency='daily'):
        """
        Retrieve historical price data for multiple securities
        
        Parameters:
        ----------
        tickers : list or str
            List of ticker symbols or a single ticker as string
        start_date : str or datetime
            Start date for historical data
        end_date : str or datetime
            End date for historical data
        fields : list, optional
            List of data fields to retrieve. Options include:
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
            If None, all fields will be retrieved
        frequency : str, optional
            Data frequency, options: 'daily', 'weekly', 'monthly'
            
        Returns:
        -------
        pandas.DataFrame
            DataFrame with multi-level columns for tickers and fields
        """
        # Default to all fields if none specified
        if fields is None:
            fields = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        if tickers is None:
            tickers = self.tickers

        # Handle single ticker case
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Map frequency to yfinance interval
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        interval = interval_map.get(frequency, '1d')
        
        # Use yfinance to download data with proper grouping
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        
        # Create properly structured multi-level DataFrame
        # Check if data already has multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            # Filter to keep only requested fields if needed
            if fields != ['Open', 'High', 'Low', 'Close', 'Volume']:
                filtered_columns = [(ticker, field) for ticker in tickers for field in fields if (ticker, field) in data.columns]
                data = data.loc[:, filtered_columns]
        else:
            # For single ticker, need to restructure to multi-level
            ticker = tickers[0]
            # Create a new multi-level DataFrame
            multi_data = pd.DataFrame()
            for field in fields:
                if field in data.columns:
                    multi_data[(ticker, field)] = data[field]
            data = multi_data
            # Convert to proper MultiIndex
            data.columns = pd.MultiIndex.from_tuples(data.columns, names=['Ticker', 'Price'])
        
        return data

    def calculate_returns(self, data, method='simple', period=1):
        """
        Calculate returns from price data
        
        Parameters:
        ----------
        data : pandas.DataFrame
            Price data with multi-level columns (ticker, field)
        method : str, optional
            'simple' or 'log' returns
        period : int, optional
            Period for return calculation (e.g., 1 for daily, 5 for weekly)
            
        Returns:
        -------
        pandas.DataFrame
            DataFrame with returns data
        """
        # Extract close prices for all tickers
        close_data = pd.DataFrame()
        
        # If we have multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            tickers = data.columns.get_level_values(0).unique()
            for ticker in tickers:
                # Try to get Close first, since we're using auto_adjust=True
                if ('Close' in data[ticker].columns):
                    close_data[ticker] = data[ticker]['Close']
                # Fallback to Adj Close if available 
                elif ('Adj Close' in data[ticker].columns):
                    close_data[ticker] = data[ticker]['Adj Close']
        else:
            # Try to get Close column
            if 'Close' in data.columns:
                close_data = data['Close']
            # Fallback to Adj Close if available
            elif 'Adj Close' in data.columns:
                close_data = data['Adj Close']
            else:
                close_data = data  # Use whatever is available
        
        # Calculate returns based on method
        if method == 'simple':
            returns = close_data.pct_change(period).dropna()
        elif method == 'log':
            returns = np.log(close_data / close_data.shift(period)).dropna()
        
        return returns


class WebSocketRealTimeData:
    """
    A class for real-time data using WebSocket connections.
    Note: Many real-time WebSocket feeds require paid subscriptions.
    This implementation provides a structure that can be extended for specific providers.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the WebSocketRealTimeData object.
        
        Parameters:
        ----------
        api_key : str, optional
            API key for the WebSocket data provider
        """
        self.api_key = api_key
        self.websocket = None
        self.connected = False
        self.tickers = []
        self.latest_data = {}
        self.callbacks = []
    
    def connect(self, provider="finnhub"):
        """
        Connect to the WebSocket provider.
        
        Parameters:
        ----------
        provider : str, optional
            The WebSocket provider to use
            
        Returns:
        -------
        bool
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.warning("Already connected to WebSocket")
            return False
        
        if provider == "finnhub":
            return self._connect_finnhub()
        else:
            logger.error(f"Unsupported WebSocket provider: {provider}")
            return False
    
    def _connect_finnhub(self):
        """
        Connect to Finnhub WebSocket API.
        
        Returns:
        -------
        bool
            True if connection successful, False otherwise
        """
        if not self.api_key:
            logger.error("API key is required for Finnhub WebSocket")
            return False
        
        try:
            # Define WebSocket callbacks
            def on_message(ws, message):
                data = json.loads(message)
                if 'type' in data and data['type'] == 'trade':
                    for trade in data['data']:
                        symbol = trade['s']
                        price = trade['p']
                        volume = trade['v']
                        timestamp = trade['t']
                        
                        self.latest_data[symbol] = {
                            'price': price,
                            'volume': volume,
                            'timestamp': timestamp
                        }
                        
                        # Call user callbacks
                        for callback in self.callbacks:
                            callback(symbol, price, volume, timestamp)
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {str(error)}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
                self.connected = False
            
            def on_open(ws):
                logger.info("WebSocket connection opened")
                self.connected = True
                
                # Subscribe to tickers
                for ticker in self.tickers:
                    ws.send(json.dumps({'type': 'subscribe', 'symbol': ticker}))
            
            # Create WebSocket connection
            self.websocket = websocket.WebSocketApp(
                f"wss://ws.finnhub.io?token={self.api_key}",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket connection in a separate thread
            websocket_thread = threading.Thread(target=self.websocket.run_forever)
            websocket_thread.daemon = True
            websocket_thread.start()
            
            # Wait for connection to establish
            time.sleep(3)
            
            return self.connected
            
        except Exception as e:
            logger.error(f"Error connecting to Finnhub WebSocket: {str(e)}")
            return False
    
    def subscribe(self, tickers):
        """
        Subscribe to real-time data for specific tickers.
        
        Parameters:
        ----------
        tickers : list
            List of ticker symbols to subscribe to
            
        Returns:
        -------
        bool
            True if subscription successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to WebSocket")
            return False
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        for ticker in tickers:
            if ticker not in self.tickers:
                self.tickers.append(ticker)
                if self.websocket:
                    self.websocket.send(json.dumps({'type': 'subscribe', 'symbol': ticker}))
        
        logger.info(f"Subscribed to tickers: {tickers}")
        return True
    
    def unsubscribe(self, tickers=None):
        """
        Unsubscribe from real-time data for specific tickers or all tickers.
        
        Parameters:
        ----------
        tickers : list, optional
            List of ticker symbols to unsubscribe from (defaults to all tickers)
            
        Returns:
        -------
        bool
            True if unsubscription successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to WebSocket")
            return False
        
        if tickers is None:
            tickers = self.tickers.copy()
        elif isinstance(tickers, str):
            tickers = [tickers]
        
        for ticker in tickers:
            if ticker in self.tickers:
                self.tickers.remove(ticker)
                if self.websocket:
                    self.websocket.send(json.dumps({'type': 'unsubscribe', 'symbol': ticker}))
        
        logger.info(f"Unsubscribed from tickers: {tickers}")
        return True
    
    def add_callback(self, callback):
        """
        Add a callback function to be called on data updates.
        
        Parameters:
        ----------
        callback : function
            Callback function that takes (symbol, price, volume, timestamp) as arguments
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback):
        """
        Remove a callback function.
        
        Parameters:
        ----------
        callback : function
            Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def disconnect(self):
        """
        Disconnect from the WebSocket.
        
        Returns:
        -------
        bool
            True if disconnection successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to WebSocket")
            return False
        
        try:
            if self.websocket:
                self.websocket.close()
            
            self.connected = False
            logger.info("Disconnected from WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from WebSocket: {str(e)}")
            return False
    
    def get_latest_data(self, ticker=None):
        """
        Get the latest data for a specific ticker or all tickers.
        
        Parameters:
        ----------
        ticker : str, optional
            Ticker symbol (defaults to all tickers)
            
        Returns:
        -------
        dict
            Latest data for the ticker(s)
        """
        if ticker:
            return self.latest_data.get(ticker, None)
        else:
            return self.latest_data.copy()


# Example usage for real-time data streaming
if __name__ == "__main__":
    # Example 1: Using RealTimeMarketData with Yahoo Finance
    rt_data = RealTimeMarketData(source="yahoo", update_interval=30)
    rt_data.add_tickers(['AAPL', 'MSFT', 'GOOGL'])
    
    # Define a callback function
    def data_callback(data):
        print(f"Data update at {datetime.now()}")
        for ticker, info in data.items():
            print(f"{ticker}: ${info['price']:.2f} ({info['change_pct']:.2f}%)")
    
    # Start streaming with callback
    rt_data.start_streaming(callback=data_callback)
    
    # Let it run for a while
    try:
        print("Streaming data for 2 minutes. Press Ctrl+C to stop early.")
        time.sleep(120)
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Stop streaming
    rt_data.stop_streaming()
    
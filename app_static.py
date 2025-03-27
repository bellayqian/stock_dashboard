import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
import json
import os
from datetime import datetime, timedelta

# Import your custom modules
from data_navigation import build_correlation_network, connected_component, path_query
from data_analysis import perform_pca, detect_market_regimes, calculate_risk_metrics
from stock_tickers import all_tickers, tech, consumer_discretionary, healthcare, financials, consumer_staples, energy, industrials, communication_services, utilities

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load the daily returns data - using static data file
df_returns = pd.read_csv('example_data/50_daily_returns.csv', index_col=0)
df_returns.index = pd.to_datetime(df_returns.index)  # Ensure dates are parsed correctly

# Create a correlation network
correlation_threshold = 0.34  # Default threshold
G = build_correlation_network(df_returns, correlation_threshold)

# Function to convert networkx graph to plotly format
def network_graph_to_plotly(G, pos=None):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Extract the edge weight (correlation)
        weight = edge[2]['weight'] if 'weight' in edge[2] else 1.0
        edge_weights.append(weight)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    # Sector-to-color mapping
    sector_colors = {
        'tech': '#1f77b4',  # blue
        'consumer_discretionary': '#ff7f0e',  # orange
        'healthcare': '#2ca02c',  # green
        'financials': '#d62728',  # red
        'consumer_staples': '#9467bd',  # purple
        'energy': '#8c564b',  # brown
        'industrials': '#e377c2',  # pink
        'communication_services': '#7f7f7f',  # gray
        'utilities': '#bcbd22'  # olive
    }
    
    # Function to determine sector for a ticker
    def get_ticker_sector(ticker):
        if ticker in tech:
            return 'tech'
        elif ticker in consumer_discretionary:
            return 'consumer_discretionary'
        elif ticker in healthcare:
            return 'healthcare'
        elif ticker in financials:
            return 'financials'
        elif ticker in consumer_staples:
            return 'consumer_staples'
        elif ticker in energy:
            return 'energy'
        elif ticker in industrials:
            return 'industrials'
        elif ticker in communication_services:
            return 'communication_services'
        elif ticker in utilities:
            return 'utilities'
        else:
            return 'other'
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Node size based on degree
        node_size.append(10 + 5 * G.degree(node))
        # Node color based on sector
        sector = get_ticker_sector(node)
        node_color.append(sector_colors.get(sector, '#17becf'))  # default to cyan if not found
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2))
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Stock Correlation Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Add sector legend
    legend_x = 1.05
    legend_y = 1.0
    for sector, color in sector_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            showlegend=True,
            name=sector.replace('_', ' ').title()
        ))
    
    return fig, pos

# Prepare initial network graph
initial_graph, pos = network_graph_to_plotly(G)

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Interactive Stock Market Dashboard", className="text-center my-4"),
            html.P("Explore stock correlations, analyze market regimes, and monitor market data", 
                   className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Network Visualization"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Correlation Threshold:"),
                            dcc.Slider(
                                id='correlation-threshold',
                                min=0.1,
                                max=0.9,
                                step=0.05,
                                value=correlation_threshold,
                                marks={i/10: str(i/10) for i in range(1, 10)},
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Search Stocks:"),
                            dbc.InputGroup([
                                dbc.Input(id="stock-search", placeholder="Enter ticker (e.g., AAPL)"),
                                dbc.Button("Highlight", id="search-button", color="primary"),
                            ]),
                            html.Div(id="search-results", className="mt-2"),
                        ], width=6),
                    ], className="mb-3"),
                    dcc.Graph(id='network-graph', figure=initial_graph),
                    html.Div(id='click-data', className="mt-3"),
                ])
            ], className="mb-4"),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Information"),
                dbc.CardBody([
                    html.Div(id='stock-info-placeholder', children=[
                        html.P("Click on a stock in the network to see detailed information.", className="text-muted")
                    ]),
                    html.Div(id='stock-info', style={'display': 'none'}, children=[
                        dbc.Tabs([
                            dbc.Tab(label="Price Data", children=[
                                dcc.Graph(id='price-chart')
                            ]),
                            dbc.Tab(label="Analytics", children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("PCA Analysis", className="mt-3"),
                                        dcc.Graph(id='pca-chart')
                                    ], width=6),
                                    dbc.Col([
                                        html.H5("Market Regimes", className="mt-3"),
                                        dcc.Graph(id='regime-chart')
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Risk Metrics", className="mt-3"),
                                        html.Div(id='risk-metrics')
                                    ], width=12),
                                ]),
                            ]),
                            dbc.Tab(label="Connected Stocks", children=[
                                html.Div(id='connected-stocks')
                            ]),
                        ]),
                    ]),
                ])
            ]),
        ], width=12),
    ]),
    
    # Store components for sharing data between callbacks
    dcc.Store(id='network-pos-store'),
    dcc.Store(id='selected-stock-store'),
    dcc.Store(id='connected-components-store'),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Stock Market Visualization Dashboard - Created with Dash and Plotly", className="text-center text-muted")
        ])
    ])
], fluid=True)

# Callback to update the network graph based on correlation threshold
@app.callback(
    [Output('network-graph', 'figure'),
     Output('network-pos-store', 'data'),
     Output('connected-components-store', 'data')],
    [Input('correlation-threshold', 'value')],
    prevent_initial_call=False
)
def update_network(threshold):
    # Build new correlation network
    G = build_correlation_network(df_returns, threshold)
    
    # Get connected components
    components = list(connected_component(G))
    
    # Convert to suitable format for storage
    components_json = [list(comp) for comp in components]
    
    # Generate new graph
    fig, pos = network_graph_to_plotly(G)
    
    # Convert positions to serializable format
    pos_json = {node: [float(x), float(y)] for node, (x, y) in pos.items()}
    
    return fig, pos_json, components_json

# Callback to handle stock search and highlighting
@app.callback(
    [Output('network-graph', 'figure', allow_duplicate=True),
     Output('search-results', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('stock-search', 'value'),
     State('network-graph', 'figure'),
     State('network-pos-store', 'data')],
    prevent_initial_call=True
)
def highlight_stock(n_clicks, stock_ticker, current_figure, pos_json):
    if not stock_ticker:
        return current_figure, html.P("Please enter a ticker symbol", className="text-danger")
    
    stock_ticker = stock_ticker.upper()
    
    # Check if the ticker exists in our data
    if stock_ticker not in all_tickers:
        return current_figure, html.P(f"Ticker '{stock_ticker}' not found", className="text-danger")
    
    # Check if the ticker exists in the current network
    if stock_ticker not in pos_json:
        return current_figure, html.P(f"Ticker '{stock_ticker}' is not in the current network", className="text-warning")
    
    # Highlight the selected stock in the network
    new_figure = go.Figure(current_figure)
    
    # Find the node trace
    for trace_idx, trace in enumerate(new_figure.data):
        if trace.mode == 'markers':  # This is the node trace
            # Get the index of the selected stock
            stock_idx = trace.text.index(stock_ticker) if stock_ticker in trace.text else None
            
            if stock_idx is not None:
                # Make a copy of the marker colors
                colors = list(trace.marker.color)
                sizes = list(trace.marker.size)
                
                # Change the color and size of the selected stock
                colors[stock_idx] = 'red'
                sizes[stock_idx] = sizes[stock_idx] * 1.5
                
                # Update the trace
                new_figure.data[trace_idx].marker.color = colors
                new_figure.data[trace_idx].marker.size = sizes
    
    return new_figure, html.P(f"Found and highlighted: {stock_ticker}", className="text-success")

# Callback for handling node clicks
@app.callback(
    [Output('click-data', 'children'),
     Output('stock-info-placeholder', 'style'),
     Output('stock-info', 'style'),
     Output('selected-stock-store', 'data')],
    [Input('network-graph', 'clickData')],
    prevent_initial_call=True
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a node to see details", {'display': 'block'}, {'display': 'none'}, None
    
    point = clickData['points'][0]
    if 'text' in point:
        ticker = point['text']
        return f"Selected: {ticker}", {'display': 'none'}, {'display': 'block'}, ticker
    
    return "Click on a node to see details", {'display': 'block'}, {'display': 'none'}, None

# Callback to update the price chart when a stock is selected
@app.callback(
    Output('price-chart', 'figure'),
    [Input('selected-stock-store', 'data')],
    prevent_initial_call=True
)
def update_price_chart(selected_stock):
    if not selected_stock:
        return go.Figure()
    
    # For the static version, we'll use the returns data to create a dummy price chart
    # Start with a price of 100 and apply the returns
    stock_returns = df_returns[selected_stock]
    
    # Calculate cumulative returns
    cumulative_returns = (1 + stock_returns).cumprod() - 1
    
    # Create a price series starting at 100
    start_price = 100
    price_series = start_price * (1 + cumulative_returns)
    
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name=selected_stock
    ))
    
    fig.update_layout(
        title=f"{selected_stock} Price History (Simulated from Returns)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Callback to update the PCA chart
@app.callback(
    Output('pca-chart', 'figure'),
    [Input('selected-stock-store', 'data'),
     Input('connected-components-store', 'data')],
    prevent_initial_call=True
)
def update_pca_chart(selected_stock, components_json):
    if not selected_stock or not components_json:
        return go.Figure()
    
    # Find which component contains the selected stock
    selected_component = None
    for comp in components_json:
        if selected_stock in comp:
            selected_component = comp
            break
    
    if not selected_component:
        # Create empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Selected stock is not connected to other stocks",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Filter returns to only include stocks in the same component
    component_returns = df_returns[selected_component]
    
    # Perform PCA
    principal_components, explained_variance_ratios, _ = perform_pca(component_returns, n_components=2)
    
    # Create a DataFrame with the first two principal components
    pca_df = pd.DataFrame(
        principal_components, 
        columns=['PC1', 'PC2'],
        index=component_returns.index
    )
    
    # Create the figure
    fig = px.scatter(
        pca_df, x='PC1', y='PC2',
        title=f"PCA Analysis for {selected_stock}'s Component",
        labels={'PC1': f'PC1 ({explained_variance_ratios[0]:.2%})', 
                'PC2': f'PC2 ({explained_variance_ratios[1]:.2%})'},
        height=400
    )
    
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    
    return fig

# Callback to update the regime chart
@app.callback(
    Output('regime-chart', 'figure'),
    [Input('selected-stock-store', 'data'),
     Input('connected-components-store', 'data')],
    prevent_initial_call=True
)
def update_regime_chart(selected_stock, components_json):
    if not selected_stock or not components_json:
        return go.Figure()
    
    # Find which component contains the selected stock
    selected_component = None
    for comp in components_json:
        if selected_stock in comp:
            selected_component = comp
            break
    
    if not selected_component:
        # Create empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Selected stock is not connected to other stocks",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Filter returns to only include stocks in the same component
    component_returns = df_returns[selected_component]
    
    # Calculate regimes
    regimes, _ = detect_market_regimes(component_returns, n_regimes=3)
    
    # Create a DataFrame with the regime data
    regime_df = pd.DataFrame({'Regime': regimes}, index=component_returns.index)
    
    # Calculate the returns of the selected stock
    stock_returns = df_returns[selected_stock]
    
    # Combine the regime and return data
    combined_df = pd.DataFrame({
        'Regime': regimes,
        'Return': stock_returns
    }, index=component_returns.index)
    
    # Create cumulative returns
    combined_df['Cumulative Return'] = (1 + combined_df['Return']).cumprod() - 1
    
    # Create the figure
    fig = go.Figure()
    
    # Add cumulative returns line
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Cumulative Return'],
        mode='lines',
        name=f"{selected_stock} Cumulative Return"
    ))
    
    # Add colored background for different regimes
    for regime in sorted(combined_df['Regime'].unique()):
        regime_periods = []
        in_regime = False
        start_date = None
        
        for date, row in combined_df.iterrows():
            if row['Regime'] == regime and not in_regime:
                start_date = date
                in_regime = True
            elif row['Regime'] != regime and in_regime:
                regime_periods.append((start_date, date))
                in_regime = False
        
        # Add the last period if we're still in the regime
        if in_regime:
            regime_periods.append((start_date, combined_df.index[-1]))
        
        # Add colored background for each period
        colors = ['rgba(255, 200, 200, 0.3)', 'rgba(200, 255, 200, 0.3)', 'rgba(200, 200, 255, 0.3)']
        for start, end in regime_periods:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=colors[regime % len(colors)],
                opacity=0.5,
                layer="below",
                line_width=0,
            )
    
    fig.update_layout(
        title=f"Market Regimes and {selected_stock} Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback to update the risk metrics
@app.callback(
    Output('risk-metrics', 'children'),
    [Input('selected-stock-store', 'data'),
     Input('connected-components-store', 'data')],
    prevent_initial_call=True
)
def update_risk_metrics(selected_stock, components_json):
    if not selected_stock or not components_json:
        return html.P("No stock selected")
    
    # Find which component contains the selected stock
    selected_component = None
    for comp in components_json:
        if selected_stock in comp:
            selected_component = comp
            break
    
    if not selected_component:
        return html.P("Selected stock is not connected to other stocks")
    
    # Filter returns to only include stocks in the same component
    component_returns = df_returns[selected_component]
    
    # Calculate equal weights for all stocks in the component
    weights = [1/len(selected_component)] * len(selected_component)
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(component_returns, weights)
    
    # Calculate mean return and volatility for the selected stock
    stock_return = df_returns[selected_stock].mean() * 252  # Annualize
    stock_volatility = df_returns[selected_stock].std() * np.sqrt(252)  # Annualize
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Portfolio Metrics"),
                dbc.CardBody([
                    html.P(f"Value at Risk (VaR): {risk_metrics['VaR']:.2%}"),
                    html.P(f"Conditional VaR (CVaR): {risk_metrics['CVaR']:.2%}"),
                    html.P(f"Portfolio Volatility: {risk_metrics['Volatility']:.2%}")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(f"{selected_stock} Metrics"),
                dbc.CardBody([
                    html.P(f"Expected Annual Return: {stock_return:.2%}"),
                    html.P(f"Annual Volatility: {stock_volatility:.2%}"),
                    html.P(f"Sharpe Ratio: {stock_return / stock_volatility:.2f}")
                ])
            ])
        ], width=6)
    ])

# Callback to update the connected stocks
@app.callback(
    Output('connected-stocks', 'children'),
    [Input('selected-stock-store', 'data'),
     Input('connected-components-store', 'data')],
    prevent_initial_call=True
)
def update_connected_stocks(selected_stock, components_json):
    if not selected_stock or not components_json:
        return html.P("No stock selected")
    
    # Find which component contains the selected stock
    selected_component = None
    for comp in components_json:
        if selected_stock in comp:
            selected_component = comp
            break
    
    if not selected_component:
        return html.P("Selected stock is not connected to other stocks")
    
    # Create a list of connected stocks
    connected_stocks = [stock for stock in selected_component if stock != selected_stock]
    
    if not connected_stocks:
        return html.P("No connected stocks found")
    
    # Group stocks by sector
    sectors = {}
    for ticker in connected_stocks:
        sector = None
        if ticker in tech:
            sector = "Technology"
        elif ticker in consumer_discretionary:
            sector = "Consumer Discretionary"
        elif ticker in healthcare:
            sector = "Healthcare"
        elif ticker in financials:
            sector = "Financials"
        elif ticker in consumer_staples:
            sector = "Consumer Staples"
        elif ticker in energy:
            sector = "Energy"
        elif ticker in industrials:
            sector = "Industrials"
        elif ticker in communication_services:
            sector = "Communication Services"
        elif ticker in utilities:
            sector = "Utilities"
        else:
            sector = "Other"
        
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(ticker)
    
    # Create a table of connected stocks by sector
    return html.Div([
        html.H5(f"Stocks Connected to {selected_stock}", className="mb-3"),
        html.Div([
            dbc.Card([
                dbc.CardHeader(sector),
                dbc.CardBody([
                    html.Div([
                        dbc.Badge(ticker, color="primary", className="mr-1 mb-1")
                        for ticker in tickers
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '5px'})
                ])
            ], className="mb-3")
            for sector, tickers in sectors.items()
        ])
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
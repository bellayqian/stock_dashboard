import os
import shutil
import dash_renderer
from dash.dash import _js_dist_dependencies
import json
from app_static import app

# Create build directory
if os.path.exists('build'):
    shutil.rmtree('build')
os.makedirs('build')
os.makedirs('build/assets', exist_ok=True)

# Copy the static data file
os.makedirs('build/example_data', exist_ok=True)
shutil.copy('example_data/50_daily_returns.csv', 'build/example_data/50_daily_returns.csv')

# Copy pre-generated visualization files
os.makedirs('build/visualizations', exist_ok=True)

# Copy HTML visualizations
html_files = [
    'interactive_market_regimes.html',
    'interactive_network.html',
    'interactive_pca_clustering.html',
    'interactive_risk_metrics_comparison.html',
    'interactive_risk_metrics_heatmap.html'
]

for html_file in html_files:
    source_path = os.path.join('output_figures', html_file)
    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join('build/visualizations', html_file))
        print(f"Copied {html_file}")
    else:
        print(f"Warning: {source_path} not found")

# Copy PNG visualizations
png_files = [
    'pca_biplot.png',
    'pca_clustering.png',
    'regime_characteristics.png'
]

for png_file in png_files:
    source_path = os.path.join('output_figures', png_file)
    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join('build/visualizations', png_file))
        print(f"Copied {png_file}")
    else:
        print(f"Warning: {source_path} not found")

# Create index.html with app layout
app_div = """
<div id="react-entry-point">
    <div class="_dash-loading">
        Loading...
    </div>
</div>
"""

# Get the scripts and stylesheets the app needs
scripts = []
css = []

# Add dashboard-specific assets
if os.path.exists('assets'):
    for file in os.listdir('assets'):
        if file.endswith('.js'):
            scripts.append(f'assets/{file}')
        elif file.endswith('.css'):
            css.append(f'assets/{file}')

# Create the HTML file
with open('build/index.html', 'w') as f:
    f.write(f"""
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Stock Market Visualization</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <!-- Plotly JS -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <!-- jQuery -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <!-- Dashboard CSS -->
        {''.join([f'<link rel="stylesheet" href="{sheet}">' for sheet in css])}
    </head>
    <body>
        <div class="container-fluid p-4">
            <div class="row justify-content-center">
                <div class="col-12">
                    <h1 class="text-center mb-4">Interactive Stock Market Dashboard</h1>
                    <p class="text-center text-muted mb-5">Explore stock correlations, analyze market regimes, and examine market data</p>
                </div>
            </div>
            
            <div class="row">
                <div class="col-12">
                    <div class="alert alert-info">
                        <p><strong>This is a static version of the dashboard.</strong> Pre-generated visualizations are displayed below.</p>
                        <p>For the full interactive experience, please clone the repository and run the app locally.</p>
                    </div>
                </div>
            </div>
            
            <!-- Network Visualization -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h3>Stock Correlation Network</h3>
                        </div>
                        <div class="card-body">
                            <iframe src="visualizations/interactive_network.html" width="100%" height="600px" frameborder="0"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- PCA Visualizations -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h3>Principal Component Analysis</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h4>PCA Clustering</h4>
                                    <iframe src="visualizations/interactive_pca_clustering.html" width="100%" height="500px" frameborder="0"></iframe>
                                </div>
                                <div class="col-md-6">
                                    <h4>PCA Biplot</h4>
                                    <img src="visualizations/pca_biplot.png" class="img-fluid" alt="PCA Biplot">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Market Regimes Visualization -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h3>Market Regimes Analysis</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-8">
                                    <h4>Market Regimes Timeline</h4>
                                    <iframe src="visualizations/interactive_market_regimes.html" width="100%" height="600px" frameborder="0"></iframe>
                                </div>
                                <div class="col-md-4">
                                    <h4>Regime Characteristics</h4>
                                    <img src="visualizations/regime_characteristics.png" class="img-fluid" alt="Regime Characteristics">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Risk Metrics Visualizations -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h3>Risk Metrics Analysis</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-12 mb-4">
                                    <h4>Risk Metrics Comparison</h4>
                                    <iframe src="visualizations/interactive_risk_metrics_comparison.html" width="100%" height="500px" frameborder="0"></iframe>
                                </div>
                                <div class="col-md-12">
                                    <h4>Risk Metrics Heatmap</h4>
                                    <iframe src="visualizations/interactive_risk_metrics_heatmap.html" width="100%" height="500px" frameborder="0"></iframe>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="row mt-5">
                <div class="col text-center">
                    <hr>
                    <p class="text-muted">Stock Market Visualization Dashboard - Created with Python, Dash, and Plotly</p>
                </div>
            </footer>
        </div>
        
        <!-- Dashboard JS -->
        {''.join([f'<script src="{script}"></script>' for script in scripts])}
    </body>
</html>
    """)

print("Static files built successfully in the 'build' directory.")
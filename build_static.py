import os
import shutil

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
    'interactive_pca_biplot.html',
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

# Copy PNG visualizations (only regime_characteristics.png)
png_files = [
    'regime_characteristics.png'
]

for png_file in png_files:
    source_path = os.path.join('output_figures', png_file)
    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join('build/visualizations', png_file))
        print(f"Copied {png_file}")
    else:
        print(f"Warning: {source_path} not found")

# Create a simple HTML file for our visualization gallery
with open('build/index.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Financial Data Analysis & Visualization</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <!-- Plotly JS -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container-fluid p-4">
            <div class="row justify-content-center">
                <div class="col-12">
                    <h1 class="text-center mb-4">Financial Data Analysis Dashboard</h1>
                    <p class="text-center text-muted mb-5">Explore stock correlations, analyze market regimes, and examine market data</p>
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
                            <iframe src="visualizations/interactive_network.html" width="100%" height="700px" frameborder="0" allowfullscreen="true" sandbox="allow-same-origin allow-scripts"></iframe>
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
                                <div class="col-12 mb-4">
                                    <h4>PCA Biplot</h4>
                                    <iframe src="visualizations/interactive_pca_biplot.html" width="100%" height="700px" frameborder="0" allowfullscreen="true" sandbox="allow-same-origin allow-scripts"></iframe>
                                </div>
                                <div class="col-12">
                                    <h4>PCA Clustering</h4>
                                    <iframe src="visualizations/interactive_pca_clustering.html" width="100%" height="700px" frameborder="0" allowfullscreen="true" sandbox="allow-same-origin allow-scripts"></iframe>
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
                                <div class="col-12 mb-4">
                                    <h4>Market Regimes Timeline</h4>
                                    <iframe src="visualizations/interactive_market_regimes.html" width="100%" height="700px" frameborder="0" allowfullscreen="true" sandbox="allow-same-origin allow-scripts"></iframe>
                                </div>
                                <div class="col-12">
                                    <h4>Regime Characteristics</h4>
                                    <div class="d-flex justify-content-center">
                                        <img src="visualizations/regime_characteristics.png" style="max-width: 100%; height: auto;" alt="Regime Characteristics">
                                    </div>
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
                                <div class="col-12 mb-4">
                                    <h4>Risk Metrics Comparison</h4>
                                    <iframe src="visualizations/interactive_risk_metrics_comparison.html" width="100%" height="600px" frameborder="0" allowfullscreen="true" sandbox="allow-same-origin allow-scripts"></iframe>
                                </div>
                                <div class="col-12">
                                    <h4>Risk Metrics Heatmap</h4>
                                    <iframe src="visualizations/interactive_risk_metrics_heatmap.html" width="100%" height="600px" frameborder="0" allowfullscreen="true" sandbox="allow-same-origin allow-scripts"></iframe>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="row mt-5">
                <div class="col text-center">
                    <hr>
                    <p class="text-muted">Financial Data Analysis Dashboard - Created by Bella Qian, Suhwan Bong, and Yibin Xiong</p>
                </div>
            </footer>
        </div>
    </body>
</html>
    """)

print("Static files built successfully in the 'build' directory.")
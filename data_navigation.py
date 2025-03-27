import numpy as np
import pandas as pd
import networkx as nx
from collections import deque

def build_correlation_network(df_return, threshold=None, method='pearson'):
    """
    Build a correlation network from the daily returns of stocks.
    @param df_return: the dataframe of daily returns whose columns are tickers and rows are dates
    @param threshold: the threshold of correlation above which we will put an edge between the two stocks
    @param method: method to calculate correlation. Default: pearson
    @return G: a networkx graph
    """
    corr = df_return.corr(method=method).to_numpy()
    if threshold is None:
        threshold = np.quantile(corr[np.triu_indices(len(corr),k=1)],0.75) # 75 percentile of the correlations
    A = (np.abs(corr) > threshold).astype(int)
    # Create an undirected weight graph with edge weights being the correlations
    # For correlations smaller than the threshold, we set them to 0
    G = nx.from_numpy_array(A * corr)
    # remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    # relabel the nodes with tickers
    label_mapping = {i: ticker for i, ticker in enumerate(df_return.columns)}
    G = nx.relabel_nodes(G, label_mapping)
    return G

def connected_component(G):
    """
    Find connected components in the correlation network by BFS.
    @param G: a networkx graph for (truncated) correlations
    @return a list of connected components
    """
    results = []
    visited = []
    queue = deque()
    for n in G.nodes:
        if n not in visited:
            queue.append(n)
            visited.append(n)
            component = [n]
            while queue:
                curr_node = queue.popleft()
                # visit its neighbors
                for neigh in G.neighbors(curr_node):
                    if neigh not in visited:
                        component.append(neigh)
                        visited.append(neigh)
                        queue.append(neigh)
            results.append(component)
    return results


def path_query(G, source, target):
    """
    Find whether there is a path between two nodes in the correlation network by BFS.
    @param G: a networkx graph for (truncated) correlations
    @param source: one of the tickers of the stocks of interest
    @param target: the other ticker of the stock of interest
    @return whether there is a path between two nodes
    """
    if source not in G.nodes:
        print(f'Invalid ticker "{source}"')
        return None
    if target not in G.nodes:
        print(f'Invalid ticker "{target}"')
        return None
    result = []
    visited = [source]
    queue = deque([source])
    while queue:
        curr_node = queue.popleft()
        if curr_node == target:
            return True
        # explore neighbors
        for neigh in G.neighbors(curr_node):
            if neigh not in visited:
                queue.append(neigh)
                visited.append(neigh)
    return False


if __name__ == '__main__':
    data_dir = './example_data'
    df_return = pd.read_csv(f'{data_dir}/daily_returns.csv')

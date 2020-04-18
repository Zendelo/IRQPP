import pandas as pd

import networkx as nx

df = pd.DataFrame({'a': [0.1, 0.2, 0.8], 'b': [0.4, 0.3, 0.9], 'c': [0.6, 0.7, 0.5]}, index=['a', 'b', 'c'],
                  columns=['a', 'b', 'c'])
print(df)

g = nx.from_pandas_adjacency(df, nx.DiGraph)
sdf = nx.to_pandas_adjacency(nx.stochastic_graph(g))
print(sdf)
pr = nx.pagerank_numpy(g, alpha=1)
print(f'\na:{pr["a"]:.3f}\nb:{pr["b"]:.3f}\nc:{pr["c"]:.3f}')

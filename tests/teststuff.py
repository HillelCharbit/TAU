import igraph as ig
import numpy as np
import networkx as nx
import sys
import os

# Get the absolute path to the script directory
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'tau_community_detection')
sys.path.append(script_dir)

# Import from the local script directly, not from the package
from script import run_clustering

if __name__ == '__main__':
# Open Usoskingraph.csv as a weighted graph using igraph
    g = ig.Graph.Read_Ncol("Usoskingraph.csv", weights=True, directed=False)
    print(g.summary())
    # Run Leiden community detection on the graph
    partition = g.community_leiden(objective_function='modularity', weights='weight')
    print(f"Number of communities found: {len(partition)}")
    print("Community membership for each node:")
    print(partition.membership)
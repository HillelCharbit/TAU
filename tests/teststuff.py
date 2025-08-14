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
    # Now call with weighted parameter
    tau_comms, tau_mod = run_clustering(
        'Usoskin_graph.csv',
        graph_name="example",
        weighted=True  # This should work now
    )
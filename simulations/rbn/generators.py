import numpy as np
import networkx as nx


class GenFailed(RuntimeError):
    pass


def gen_RBN(nodes, edges, strict=True):
    """
    Generates a randomly connected graph with exactly
    `nodes` number of nodes where each node has
    exactly `edges` number of edges.

    Due to the random nature of the process and for
    some combinations of nodes & edges, sometimes
    the algorithm fails. So specifiying `strict` as True
    runs an internal check and raises `GenFailed` if failed.
    """
    G = nx.Graph()
    G.add_nodes_from(range(nodes))
    
    # store which nodes are left to satisfy
    needs_edges = list(G.nodes)
    count = 0
    
    while len(needs_edges) > 1 and count < nodes**2:
        
        coming_from = np.random.choice(needs_edges)
        
        # don't allow self connections
        other_nodes = needs_edges.copy()
        other_nodes.remove(coming_from)
        going_to = np.random.choice(other_nodes)
        
        # connect them
        G.add_edge(coming_from, going_to)
            
        for x in (coming_from, going_to):
            if len(G[x]) >= edges:
                needs_edges.remove(x)
        
        # include this count as there's a bug in nx
        # where sometimes add_edge does nothing
        # which causes issues if at final 2
        count += 1
    
    # built in check
    if strict:
        for k,v in nx.degree(G):
            if v != edges:
                raise GenFailed(
                    'gen_RBN failed to generate for %s nodes %s edges'
                    % (nodes, edges))
    return G


def gen_RBN_retry(nodes, edges, retries=5):
    """
    Wrapper for gen_RBN to account for the fact with some graphs
    you often won't be able to get X exact numbers of edges on all N nodes
    so rerun if a few times until you do.
    """
    attempts = 0
    while attempts < retries:
        try:
            return gen_RBN(nodes, edges)
        except GenFailed:
            attempts += 1
    raise GenFailed("Failed to generate graph after %s attempts" % attempts)

    
def find_edges_for_RBN_network(nodes, start=2, find_all=False):
    """
    Simple function to find the nearest nodes, edges which works.
    
    Specify the number of `nodes` and the number of edges
    to `start` the search from. Where it returns the first it finds,
    unless you ask it to `find_all`.
    """
    edges = []
    for e in range(start, nodes):
        try:
            gen_RBN_retry(n, e)
            edges.append(e)
        except GenFailed:
            pass
        else:
            if find_all:
                edges.append(e)
            else:
                return e
    return edges
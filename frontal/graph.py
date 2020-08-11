from networkx.algorithms.traversal.edgedfs import edge_dfs
from networkx.algorithms.simple_paths import all_simple_paths
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import dag
from copy import deepcopy


def draw_graph(G: nx.DiGraph,
               path: str) -> None:
    """Generate a picture for a given graph

    Args:
        path (str): path of the picture
        G (nx.DiGraph): networkx graph
    """
    nx.draw(G, with_labels=True)
    plt.savefig(path)
    plt.close()


def make_graph(nodes: List,
               edges: List) -> nx.DiGraph:
    """Generate graph from nodes and edges

    Args:
        nodes (List): list of nodes, each item is name of node
        edges (List): list of edges, each tuple is start and end nodes

    Returns:
        nx.DiGraph: DAG graph that include the given nodes and edges
    """
    G = nx.DiGraph()

    # add nodes
    G.add_nodes_from(nodes)

    # add edges
    [G.add_edge(start, end) for start, end in edges]

    draw_graph(G=G, path="graphs/graph.png")

    return G


def is_ancestor(G: nx.DiGraph,
                anscestor: str,
                descendent: str) -> bool:
    """Check if the anscestor/descendent relationship exist

    Args:
        G (nx.DiGraph): networkx graph
        anscestor (str): anscestor node in question
        descendent (str): descendent node in question

    Returns:
        bool: whether the relationship is correct
    """

    anscestors = dag.ancestors(G, descendent)
    if anscestor in anscestors:
        return True
    else:
        return False


def have_collider(path: nx.DiGraph) -> bool:
    """Determine if a given path have collider in it

    Args:
        path (nx.DiGraph): directed path

    Returns:
        bool: whether the path have a collider
    """
    for node in path.nodes():
        if path.in_degree(node) > 1:
            print(f"Path {path.nodes()} have collider at {node}.")
            return True
    return False


def get_collider(path: nx.DiGraph) -> List:
    """Find and return any collider node in the path (graph)

    Args:
        path (nx.DiGraph): directed path

    Returns:
        List: list fo collider node. if no collider, then it's empty list
    """

    colliders = []
    for node in path.nodes():
        if path.in_degree(node) > 1:
            print(f"Path {path.nodes()} have collider at {node}.")
            colliders.append(node)
    return colliders


def replicate_edges(directed_graph: nx.DiGraph,
                    undirected_graph: List,
                    causal_node: str,
                    outcome_node: str) -> nx.DiGraph:
    """Indicate the direction of in path from causal variable to outcome
    variable by looking at the orignal DiGraph, and copy over the direction

    Args:
        directed_graph (nx.DiGraph): original directed graph
        undirected_graph (List): the simple_path generated path from undirected
        graph. This is just a ordered list, so direction for edges
        causal_node (str): causal variable
        outcome_node (str): outcome variable

    Returns:
        nx.DiGraph: a directed graph with direction of causal relationship
    """
    path = nx.DiGraph()

    # generate all the potential edges in List of Tuple format
    undirected_edges = []
    for index in range(1, len(undirected_graph)):
        undirected_edges.append(
            (undirected_graph[index-1], undirected_graph[index]))
        undirected_edges.append(
            (undirected_graph[index], undirected_graph[index-1]))

    # generate all the directed edges in the original graph in List of Tuple
    # format
    directed_edges = directed_graph.edges()

    # look through the potential directed edges, and if similar one exist in
    # directed graph, then grab the correct direciton and update the DiGraph
    for edge in undirected_edges:
        start, end = edge
        if edge in directed_edges:
            path.add_edge(start, end)

    return path


def get_all_path(G: nx.DiGraph,
                 causal_node: str,
                 outcome_node: str) -> List:
    """Find all the pathways from causal variable to outcome variable.

    Args:
        G (nx.DiGraph): the directed graph that have causal and outcome 
        variable included
        causal_node (str): causal variable name
        outcome_node (str): outcome variable name

    Returns:
        List: List of nx.DiGraph, each represent a direct path from causal
        variable to outcome variable
    """

    # first convert the directed graph to undirected, because the function
    # all_simple_path is a nice to find all the non-branching paths from 
    # causal variable to outcome variable
    un_directed_graph = G.to_undirected()
    un_directed_paths = [x for x in all_simple_paths(G=un_directed_graph,
                                                     source=causal_node,
                                                     target=outcome_node)]
    
    # once all the paths are found, need to add directions on the edges
    # this is done by looking at the original directed graph, and copy the
    # appropriate directions
    paths = []
    for i, path_nodes in enumerate(un_directed_paths):
        path = replicate_edges(directed_graph=G,
                               undirected_graph=path_nodes,
                               causal_node=causal_node,
                               outcome_node=outcome_node)

        draw_graph(G=path, path=f"graphs/{i}.png")
        paths.append(path)

    return paths


def met_backdoor_criterion(G: nx.DiGraph,
                           causal_node: str,
                           outcome_node: str) -> bool:
    # edge cases
    if not dag.is_directed_acyclic_graph(G):
        print("The graph is not a Directed Acyclic Graph!")
        return False

    if not is_ancestor(G=G,
                       anscestor=causal_node,
                       descendent=outcome_node):
        print("Cause node is not the ancestor of outcome_node!")
        return False

    # get all paths
    paths = get_all_path(G=G,
                         causal_node=causal_node,
                         outcome_node=outcome_node)

    # report if path have collider
    colliders = []
    for path in paths:
        colliders.extend(get_collider(path))

    return True


def main():
    # nodes = ['A', 'C', 'D', 'O', 'Y']
    # edges = [('A', 'C'), ('C', 'O'), ('C', 'D'), ('D', 'Y'), ('O', 'Y')]

    nodes = ['C', 'D', 'B', 'Y', 'G', 'F', 'A', 'U', 'V']
    edges = [('C', 'D'), ('U', 'B'), ('U', 'A'), ('B', 'D'), ('D', 'Y'),
             ('G', 'Y'), ('F', 'Y'), ('V', 'F'), ('V', 'A'), ('A', 'D')]

    # networkx use large G as graph, not lower case
    G = make_graph(nodes=nodes,
                   edges=edges)

    print(met_backdoor_criterion(G=G,
                                 causal_node='D',
                                 outcome_node='Y'))


if __name__ == "__main__":
    main()

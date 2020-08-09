from typing import List, Tuple
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph, ancestors
from networkx.algorithms.traversal.edgedfs import edge_dfs

import matplotlib.pyplot as plt


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
    
    anscestors = ancestors(G, descendent)
    if anscestor in anscestors:
        return True
    else:
        return False


def find_all_paths(edges: Tuple,
                   nodes_count: int,
                   cause_node: str,
                   result_node: str) -> List:
    paths = []
    path = nx.DiGraph()

    for index, edge in enumerate(edges):

        start, end = edge

        # add nodes
        if not path.has_node(start):
            path.add_node(start)
        if not path.has_node(end):
            path.add_node(end)

        # add edges
        path.add_edge(start, end)

        # find the nodes with only one nodes connected to it
        path_count = index + 1
        end_nodes = [x for x in path.nodes()
                     if path.out_degree(x) == 0 or path.in_degree(x) == 0]

        # check if all nodes are added
        if (cause_node in end_nodes) and (result_node in end_nodes):
            draw_graph(G=path, path=f"graphs/{len(paths)+1}.png")
            paths.append(path)
            path.clear()

    return paths


def met_backdoor_criteria(G: nx.DiGraph,
                          cause_node: str,
                          result_node: str) -> bool:
    # edge cases
    if not is_directed_acyclic_graph(G):
        print("The graph is not a Directed Acyclic Graph!")
        return False

    if not is_ancestor(G=G,
                       anscestor=cause_node,
                       descendent=result_node):
        print("Cause node is not the ancestor of result_node!")
        return False

    # determine using Pearl backdoor criteria (p109)
    # need to collect all edges, not just A -> B
    edges = [x for x in edge_dfs(G=G)]
    paths = find_all_paths(edges=edges,
                           nodes_count=len(G),
                           cause_node=cause_node,
                           result_node=result_node)

    return True


def main():
    nodes = ['C', 'D', 'O', 'Y']
    edges = [('A', 'C'), ('C', 'O'), ('C', 'D'), ('D', 'Y'), ('O', 'Y')]

    G = make_graph(nodes=nodes,
                   edges=edges)

    print(met_backdoor_criteria(G=G,
                                cause_node='D',
                                result_node='Y'))


if __name__ == "__main__":
    main()

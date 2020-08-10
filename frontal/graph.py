from networkx.algorithms.traversal.edgedfs import edge_dfs
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


def is_end_node(graph: nx.DiGraph,
                node: str) -> bool:
    """Check if a node is an end node. Doesn't matter whether it's in or out

    Args:
        graph (nx.DiGraph): graph of networkx     
        node (str): node of networkx

    Returns:
        bool: whether it's end node
    """

    if graph.out_degree(node) + graph.in_degree(node) == 1:
        return True

    return False


def remove_extra_branches(path: nx.DiGraph,
                          end_nodes: List[str]) -> nx.DiGraph:
    """Remove end node that's not in the direct pathway from start to end

    Args:
        path (nx.DiGraph): path in graph format, potentially have extra branches
        that the end nodes are not start or end
        end_nodes (List[str]): start and end nodes

    Returns:
        nx.DiGraph: cleaned direct graph from start to end
    """

    new_path = deepcopy(path)
    for node in path.nodes():
        if (node not in end_nodes) \
                and (is_end_node(graph=new_path, node=node)):
            new_path.remove_node(node)

    return new_path


def find_all_paths(edges: Tuple,
                   nodes_count: int,
                   start_node: str,
                   end_node: str) -> List:
    """Find all path from start_node to end_node

    Args:
        edges (Tuple): deep first search edges
        nodes_count (int): how many total nodes
        start_node (str): start/parent node that could be the cause
        end_node (str): end/child node that may be the result

    Returns:
        List: a list of all the path from start node to end node
    """
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
                     if is_end_node(graph=path, node=x)]

        # if end nodes is the same, then it's one version of the paths
        if (start_node in end_nodes) and (end_node in end_nodes):
            path = remove_extra_branches(path=path,
                                         end_nodes=[start_node, end_node])
            draw_graph(G=path, path=f"graphs/{len(paths)+1}.png")
            paths.append(path)
            path.clear()

    return paths


def met_backdoor_criterion(G: nx.DiGraph,
                           start_node: str,
                           end_node: str) -> bool:
    # edge cases
    if not dag.is_directed_acyclic_graph(G):
        print("The graph is not a Directed Acyclic Graph!")
        return False

    if not is_ancestor(G=G,
                       anscestor=start_node,
                       descendent=end_node):
        print("Cause node is not the ancestor of end_node!")
        return False

    # determine using Pearl backdoor criteria (p109)
    # need to collect all edges, not just A -> B
    edges = [x for x in edge_dfs(G=G)]
    paths = find_all_paths(edges=edges,
                           nodes_count=len(G),
                           start_node=start_node,
                           end_node=end_node)

    return True


def main():
    nodes = ['A', 'C', 'D', 'O', 'Y']
    edges = [('A', 'C'), ('C', 'O'), ('C', 'D'), ('D', 'Y'), ('O', 'Y')]

    # networkx use large G as graph, not lower case
    G = make_graph(nodes=nodes,
                   edges=edges)

    print(met_backdoor_criterion(G=G,
                                 start_node='D',
                                 end_node='Y'))


if __name__ == "__main__":
    main()

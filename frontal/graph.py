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


def list_to_path(node_list: List) -> nx.DiGraph:
    """Convert a list of nodes to directed graph. Edges will be going from 
    first to second node, and so on.

    Args:
        node_list (List): list of node in strings

    Returns:
        nx.DiGraph: path expressed in DiGraph format. Ignoring the original 
        path direction, but use the newly constructed direction for easier 
        access of ancestor and descendants.
    """
    path = nx.DiGraph()
    path.add_nodes_from(node_list)
    for i, node in enumerate(node_list):
        # skip first node
        if i == 0:
            continue

        # start with second node
        path.add_edge(node_list[i-1], node)
    return path


def is_ancestor(G: nx.DiGraph,
                anscestor: str,
                descendant: str) -> bool:
    """Check if the anscestor/descendant relationship exist

    Args:
        G (nx.DiGraph): networkx graph
        anscestor (str): anscestor node in question
        descendant (str): descendant node in question

    Returns:
        bool: whether the relationship is correct
    """

    anscestors = dag.ancestors(G, descendant)
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
                 outcome_node: str) -> Tuple:
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
    directed_paths = []
    for i, path_nodes in enumerate(un_directed_paths):
        path = replicate_edges(directed_graph=G,
                               undirected_graph=path_nodes,
                               causal_node=causal_node,
                               outcome_node=outcome_node)

        draw_graph(G=path, path=f"graphs/{i}.png")
        directed_paths.append(path)

    # make an acutal un_directed_graph
    un_directed_paths = [list_to_path(x) for x in un_directed_paths]

    return directed_paths, un_directed_paths


def met_backdoor_criterion(G: nx.DiGraph,
                           directed_paths: List[nx.DiGraph],
                           undirected_paths: List[nx.DiGraph],
                           causal_node: str,
                           outcome_node: str,
                           condition_nodes: List,
                           unobserved_nodes: set = set(),
                           ) -> bool:
    """check if the backdoor criterion is met in Morgan & Winship book (p109). 

    Args:
        G (nx.DiGraph): the whole directed graph
        directed_paths (List[nx.DiGraph]): the directed paths are used to check
        for collider variables
        undirected_paths (List[nx.DiGraph]): undirected paths are used to check 
        for anscestors of collider variables, and see if they are blocking 
        collider variable
        causal_node (str): causal variable
        outcome_node (str): outcome variable
        condition_nodes (List): condition variable 
        unobserved_nodes (set, optional): if the graph indicate a set of 
        unobserved variables, then use it to check whether they can be 
        conditioned on. Defaults to set().

    Returns:
        bool: indicate whether the list of variables to be conditioned 
        satisfy the backdoor criterion
    """

    # must be acyclic graph
    if not dag.is_directed_acyclic_graph(G):
        print("The graph is not a Directed Acyclic Graph!")
        return False

    # if causal variable is not a anscestor of outcome variable
    if not is_ancestor(G=G,
                       anscestor=causal_node,
                       descendant=outcome_node):
        print("Cause node is not the ancestor of outcome_node!")
        return False

    # check if any of the condition_nodes is in an unobserved node
    # assuming unobserved nodes cannot be conditioned on
    is_unobserved = any(node in unobserved_nodes for node in condition_nodes)
    if is_unobserved:
        print(f"{condition_nodes} contain unobservable variable.")
        return False

    # if any condition variable is descendant of causal_node
    # condition on a descendant of causal variable would diminish causal effect
    for condition_node in condition_nodes:
        if is_ancestor(G=G,
                       anscestor=causal_node,
                       descendant=condition_node):
            return False

    # report if path have collider
    colliders = []
    for path in directed_paths:
        colliders.extend(get_collider(path))

    # if any of the collider or descendant of collider is conditioned, check if 
    # there is any other variable in the set that's going to block the collider
    # if collider is the only variable in the set, then return false
    for undirected_path in undirected_paths:
        for condition_node in condition_nodes:
            for collider in colliders:
                if collider in undirected_path.nodes():
                    print(
                        f"collider {collider}, path {undirected_path.edges()}")
                    # need to actually check the directed path for collider 
                    # descendants
                    index = undirected_paths.index(undirected_path)
                    directed_path = directed_paths[index]
                    collider_descendants = dag.descendants(
                        directed_path, collider)

                    collider_anscestors = dag.ancestors(
                        undirected_path, collider)

                    blocked_collider = any(
                        x in collider_anscestors for x in condition_nodes)
                    
                    # check both conditioned variable and it's descedants
                    if (condition_node == collider) \
                            or (condition_node in collider_descendants):
                        if len(condition_nodes) == 1:
                            print(f"Conditioning on a collider or descendant "
                                  f"of the collider variable does not block the"
                                  f" path.")
                            return False
                        elif len(condition_nodes) > 1 and (blocked_collider):
                            print(f"collider {collider} was blocked")
                            return True
                        else:
                            print("something wrong with the collider blocking "
                                 f"path")

    return True


def main():
    nodes = ['C', 'D', 'B', 'Y', 'G', 'F', 'A', 'U', 'V']
    edges = [('C', 'D'), ('U', 'B'), ('U', 'A'), ('B', 'D'), ('D', 'Y'),
             ('G', 'Y'), ('F', 'Y'), ('V', 'F'), ('V', 'A'), ('A', 'D')]

    # # info to construct a graph
    # nodes = ['D', 'Y', 'V', 'Y2', 'Y1', 'U']
    # edges = [('D', 'Y'), ('V', 'D'), ('V', 'Y2'), ('Y2', 'Y1'), ('Y1', 'Y'),
    #          ('U', 'Y2'), ('U', 'Y')]
    # needs these to check for backdoor criterion
    causal_node = 'D'
    outcome_node = 'Y'
    condition_nodes = ['A', 'B']
    unobserved_nodes = ['V', 'U']

    # networkx use large G as graph, not lower case. follow convention here
    G = make_graph(nodes=nodes,
                   edges=edges)

    # get all paths
    directed_paths, undirected_paths = get_all_path(G=G,
                                                    causal_node=causal_node,
                                                    outcome_node=outcome_node)

    print(met_backdoor_criterion(G=G,
                                 directed_paths=directed_paths,
                                 undirected_paths=undirected_paths,
                                 causal_node=causal_node,
                                 outcome_node=outcome_node,
                                 condition_nodes=condition_nodes,
                                 unobserved_nodes=unobserved_nodes))


if __name__ == "__main__":
    main()

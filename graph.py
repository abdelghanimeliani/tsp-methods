# implement weighted non oriented graph

import numpy as np
import tsplib95 as tsp
from zeroconf import Any
from typing import Dict, Tuple, List
from itertools import permutations

class Node(str):
    def __init__(self, label: str, upper_bound: float=-float('inf') , lower_bound: float=float('inf')) -> None:
        super().__init__()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound


# Node = str 


        
class Base_Graph(object):
    """
    Undirected graph of Any labeled nodes
    """

    def __init__(self, nodes: List[Node], edges: Dict[Tuple[Node, Node], float]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.adjacency_matrix = np.array([[float('inf')] * len(self.nodes)] * len(self.nodes)) # initialize adjacency matrix
        
        # fill adjacency matrix
        for i in range(len(self.nodes)):
            self.adjacency_matrix[i][i] = 0
        for edge in self.edges.items():
            self.adjacency_matrix[self.nodes.index(edge[0][0]), self.nodes.index(edge[0][1])] = edge[1]
            self.adjacency_matrix[self.nodes.index(edge[0][1]), self.nodes.index(edge[0][0])] = edge[1]

    def get_distance(self, node1: Node, node2: Node) -> float:
        """
        return the distance between node1 and node2 or infinity if no edge
        """
        return self.adjacency_matrix[self.nodes.index(node1)][self.nodes.index(node2)]
    
    def add_node(self, node: Node) -> None:
        """
        Adds node to the graph with no edges
        """
        self.nodes.append(node)
        self.adjacency_matrix = np.append(self.adjacency_matrix, [float('inf')] * len(self.nodes), axis=0)
        self.adjacency_matrix = np.append(self.adjacency_matrix, [float('inf')] * len(self.nodes), axis=1)
        self.adjacency_matrix[-1][-1] = 0

    def add_edge(self, node1: Node, node2: Node, weight: float) -> None:
        """
        Creates an edge between the two EXISTING nodes node1 and node2
        """
        self.edges[(node1, node2)] = weight
        self.adjacency_matrix[self.nodes.index(node1)][self.nodes.index(node2)] = weight
        self.adjacency_matrix[self.nodes.index(node2)][self.nodes.index(node1)] = weight

    def get_path_length(self, path: List[Node]) -> float:
        """
        return the cost of the path (cost function)
        """
        return sum(self.get_distance(path[i], path[i+1]) for i in range(len(path)-1))

    def get_upper_bound(self, node: Node) -> float:
        """
        Compute the upper bound of the node (try to minimize the bound)
        """
        pass

    def get_lower_bound(self, node: Node) -> float:
        """
        Compute the lower bound of the node (try to maximize the bound)
        """
        pass

    def get_successors(self, node: Node) -> List[Node]:
        """
        Retrn the list of successors of the node
        """
        pass
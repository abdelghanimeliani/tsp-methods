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


        
class Graph(object):
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

    def exact_solution(self, origin: Node) -> Tuple[List[Node], float]:
        """
        Return the exact solution of the problem as a path, cost tuple
        """
        path = [origin]
        cost = float('inf')
        nodes = [node for node in self.nodes if node != origin]
        for permutation in permutations(nodes):
            weight = 0
            current = origin
            for node in permutation:
                weight += self.get_distance(current, node)
                current = node
            weight += self.get_distance(current, origin)
            if weight < cost:
                cost = weight
                path += list(permutation) + [origin]
        return path, cost

    def branch_and_bound(self, origin: Node) -> Tuple[List[Node], float]:
        """
        Return the exact solution using branch and bound method
        """
        
        # initialize the upper and lower bounds
        upper_bound = 0
        lower_bound = float('inf')

        # initialize the path
        path = [origin]
        cost = 0

        # get a successor to explore
        successor = self.get_successors(origin)[0]



            


g = Graph(
    nodes = [Node(label) for label in ['A', 'B', 'C', 'D']],
    edges = {
        ('A', 'B'): 1,
        ('A', 'C'): 2,
        ('A', 'D'): 3,
        ('B', 'C'): 4,
        ('B', 'D'): 5,
        ('C', 'D'): 6
    }
)
print(g.exact_solution('A'))
# class Graph(object):
#     def __init__(self, nodes: list, edges: dict) -> None:
#         self.nodes = nodes
#         self.adjascency_matrix = np.array([
#             [edges[(i, j)] for i in nodes] for j  in nodes
#         ])

#     def get_distance(self, node1, node2):
#         return self.adjascency_matrix[node1][node2]

#     def get_neighbors(self, node):
#         # return all nodes that are adjacent to node ordered by distance
#         return sorted(
#             [i for i in range(len(self.nodes)) if self.adjascency_matrix[node][i]],
#             key=lambda i: self.get_distance(node, i)
#         )

#     def get_path_length(self, path):
#         """
#         params
#         path : list(node)
#             list of nodes that constitute path
#         """
#         return np.sum([
#             self.get_distance(i, i+1) for i in range(len(path-1))
#         ])
        
#     def branch_and_bound(self, initial):
#         """
#         params
#         initial : int
#             index of initial node
#         """

#         # initialize
#         unvisited = [node for node in self.nodes if node != initial]
#         visited = [initial]
#         bound = float('inf')
#         evaluation = 0

#         # loop
#         while unvisited:
#             last = visited[-1]
#             neighbors = self.get_neighbors(last)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     break
#             else:
#                 raise ValueError('No unvisited neighbors')
#             visited.append(neighbor)
#             unvisited.remove(neighbor)
#             evaluation += self.get_distance(last, neighbor)
#             if evaluation >= bound:
#                 break
#             #?
#             bound = evaluation + self.get_path_length(visited + [0])
#         return evaluation, visited + [initial]

        


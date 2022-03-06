# implement weighted non oriented graph

import numpy as np

class Graph(object):
    def __init__(self, nodes: list, edges: dict) -> None:
        self.nodes = nodes
        self.adjascency_matrix = np.array([
            [edges[(i, j)] for i in nodes] for j  in nodes
        ])

    def get_distance(self, node1, node2):
        return self.adjascency_matrix[node1][node2]

    def get_neighbors(self, node):
        # return all nodes that are adjacent to node ordered by distance
        return sorted(
            [i for i in range(len(self.nodes)) if self.adjascency_matrix[node][i]],
            key=lambda i: self.get_distance(node, i)
        )

    def get_path_length(self, path):
        """
        params
        path : list(node)
            list of nodes that constitute path
        """
        return np.sum([
            self.get_distance(i, i+1) for i in range(len(path-1))
        ])
        
    def branch_and_bound(self, initial):
        """
        params
        initial : int
            index of initial node
        """

        # initialize
        unvisited = [node for node in self.nodes if node != initial]
        visited = [initial]
        bound = float('inf')
        evaluation = 0

        # loop
        while unvisited:
            last = visited[-1]
            neighbors = self.get_neighbors(last)
            for neighbor in neighbors:
                if neighbor not in visited:
                    break
            else:
                raise ValueError('No unvisited neighbors')
            visited.append(neighbor)
            unvisited.remove(neighbor)
            evaluation += self.get_distance(last, neighbor)
            if evaluation >= bound:
                break
            #?
            bound = evaluation + self.get_path_length(visited + [0])
        return evaluation, visited + [initial]

        


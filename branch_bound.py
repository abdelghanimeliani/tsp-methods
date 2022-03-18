from graph import Base_Graph, Node
from typing import List, Tuple
class Branch_and_Bound_Graph(Base_Graph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        return 0

g = Branch_and_Bound_Graph(
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
print(g.branch_and_bound('A'))
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

        


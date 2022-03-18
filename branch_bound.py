from graph import Base_Graph, Node
from typing import List, Tuple
from queue import LifoQueue as stack

class Branch_and_Bound_Graph(Base_Graph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def branch_and_bound(self, origin: Node) -> Tuple[List[Node], float]:
        """
        Return the exact solution using branch and bound method
        """
        # Initialization
        s = stack()
        s.put([origin])
        best_path = []
        best_cost = float('inf')

        # Branch and bound
        while s.qsize():
            current_path = s.get()
           
            # Cut bad branches
            if self.get_path_length(current_path) + len(self.nodes) - len(current_path) < best_cost: 
           
                # If the current path is a solution
                if len(current_path) == len(self.nodes) + 1:
                    best_path = current_path
                    best_cost = self.get_path_length(current_path)

                # If the current path is not a solution
                elif len(current_path) < len(self.nodes):

                    # For each unvisited node in the graph
                    for node in self.nodes:
                        if node not in current_path:
                            # Create a new path to explore
                            s.put(current_path + [node])
                else:
                    s.put(current_path + [origin])
        return best_path, self.get_path_length(best_path)
            
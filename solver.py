from typing import Callable
from tsplib95.models import StandardProblem
from queue import LifoQueue as stack

class Solver:
    def __init__(self, method: str="branch_and_bound") -> None:
        self.method = method

    def _branch_and_bound_solve(self, instance:StandardProblem, heuristic:Callable) ->StandardProblem:
        """
        Return the exact solution using branch and bound method
        """

        # Initialization
        s = stack().put([0])
        best_path = []
        best_cost = float('inf')

        # Branch and bound
        while s.qsize():
            current_path = s.get()
           
            # Cut bad branches
            if heuristic(current_path) < best_cost: 
           
                # If the current path is a solution
                if len(current_path) == len(self.nodes) + 1:
                    best_path = current_path
                    best_cost = instance.trace_tours(current_path)

                # If the current path is neither a solution nor an almost solution (i.e. one node from a solution)
                elif len(current_path) < len(self.nodes):

                    # For each unvisited node in the graph
                    for node in instance.get_nodes():
                        if node not in current_path:
                            # Create a new path to explore
                            s.put(current_path + [node])
                else:
                    s.put(current_path + [0])
        


from typing import Callable
from tsplib95.models import StandardProblem
from queue import LifoQueue as stack
from numpy.random import random, shuffle, randint
from numpy import exp

class Solver:
    def __init__(self, method: str="branch_and_bound") -> None:
        self.method = method
    @classmethod
    def neighbor_Δ_cost(cls, instance:StandardProblem, path:list, neighbor_path:list, nodes_to_swap:list) -> float:
        return instance.get_weight(neighbor_path[nodes_to_swap[0]], neighbor_path[nodes_to_swap[0]] - 1)\
                + instance.get_weight(neighbor_path[nodes_to_swap[0]], neighbor_path[nodes_to_swap[0]] + 1)\
                + instance.get_weight(neighbor_path[nodes_to_swap[1]], neighbor_path[nodes_to_swap[1]] - 1)\
                + instance.get_weight(neighbor_path[nodes_to_swap[1]], neighbor_path[nodes_to_swap[1]] + 1)\
                - instance.get_weight(path[nodes_to_swap[0]], path[nodes_to_swap[0]] - 1)\
                - instance.get_weight(path[nodes_to_swap[0]], path[nodes_to_swap[0]] + 1)\
                - instance.get_weight(path[nodes_to_swap[1]], path[nodes_to_swap[1]] - 1)\
                - instance.get_weight(path[nodes_to_swap[1]], path[nodes_to_swap[1]] + 1)
        

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
        
    def _simulated_annealing_solve(
        self, 
        instance:StandardProblem, 
        initial_temperature:float=100, 
        final_temperature:float=1,
        cooling_rate:float=.5
    ) ->StandardProblem:
        """
        Returns an approximate solution using simulated annealing method
        :param tsplib95.models.StandardProblem instance: tsp instance to solve
        :param float initial_temperature: initial temperature
        :param float final_temperature: final temperature
        :param float cooling_rate: cooling rate
        :return: an approximate solution
        :rtype: tsplib95.models.StandardProblem
        """

        # initialization
        temperature = initial_temperature
        path = shuffle(list(instance.get_nodes()))
        cost = instance.trace_tours(path)

        # Simulated annealing
        while temperature > final_temperature:
            # compute neighbor path (swap two nodes at random)
            nodes_to_swap = randint(0, len(path), 2)
            neighbor_path = path[:]
            neighbor_path[nodes_to_swap[0]], neighbor_path[nodes_to_swap[1]] = neighbor_path[nodes_to_swap[1]], neighbor_path[nodes_to_swap[0]]

            # compute cost difference
            Δ_cost = Solver.neighbor_Δ_cost(instance, path, neighbor_path, nodes_to_swap)

            # Accept or reject neighbor path
            if Δ_cost < 0 or exp(-Δ_cost / temperature) > random():
                path = neighbor_path
                cost = instance.trace_tours(path)

            # Cooling
            temperature *= cooling_rate

        # Return the best path
        return StandardProblem(
            name=f"{instance.name} Simulated Annealing Solution",
            comment=f"Simulated Annealing Solution to {instance.name}",
            dimension=instance.dimension,
            type="TOUR",
            dimension=instance.dimension,
            tour_section=path,
        ) 



    def solve(self, instance:StandardProblem, method:str="simulated_annealing", *args, **kwargs) ->StandardProblem:
        
        if method == "branch_and_bound":
            return self._branch_and_bound_solve(instance, lambda path: instance.trace_tours(path), *args, **kwargs)
        elif method == "simulated_annealing":
            return self._simulated_annealing_solve(instance, *args, **kwargs)
        else:
            raise ValueError("Method not implemented")

    def __call__(self, *args, **kwds) -> StandardProblem:
        return self.solve(*args, **kwds)
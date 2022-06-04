
from ast import While
from typing import Callable
from tsplib95.models import StandardProblem
from queue import LifoQueue as stack
from numpy.random import random, shuffle, randint, sample
from numpy import exp
import logging
from tsplib95 import load
logging.basicConfig(level=logging.DEBUG)
from numpy import array
from time import perf_counter



class Solver:
    def __init__(self, method: str="branch_and_bound") -> None:
        self.method = method
    @classmethod
    def neighbor_Δ_cost(cls, instance:StandardProblem, path:list, neighbor_path:list, nodes_to_swap:list) -> float:
        mod = len(path)
        return instance.get_weight(neighbor_path[nodes_to_swap[0]], neighbor_path[(nodes_to_swap[0] - 1) % mod])\
                + instance.get_weight(neighbor_path[nodes_to_swap[0]], neighbor_path[(nodes_to_swap[0] + 1) % mod])\
                + instance.get_weight(neighbor_path[nodes_to_swap[1]], neighbor_path[(nodes_to_swap[1] - 1) % mod])\
                + instance.get_weight(neighbor_path[nodes_to_swap[1]], neighbor_path[(nodes_to_swap[1] + 1) % mod])\
                - instance.get_weight(path[nodes_to_swap[0]], path[(nodes_to_swap[0] - 1) % mod])\
                - instance.get_weight(path[nodes_to_swap[0]], path[(nodes_to_swap[0] + 1) % mod])\
                - instance.get_weight(path[nodes_to_swap[1]], path[(nodes_to_swap[1] - 1) % mod])\
                - instance.get_weight(path[nodes_to_swap[1]], path[(nodes_to_swap[1] + 1) % mod])        


    def _get_adjacency_matrix(self, instance:StandardProblem) -> array:
        nodes = list(instance.get_nodes())
        return array(
            [
                [instance.get_weight(i, j) for j in instance.get_nodes()] for i in instance.get_nodes()
            ]
        )

    def _branch_and_bound_solve(self, instance:StandardProblem):
        """
        Return the exact solution using branch and bound method
        """
        # Initialization
        s = stack()
        s.put([1])
        best_path = []
        best_cost = float('inf')

        # Branch and bound
        while s.qsize():
            current_path = s.get()
           
            # Cut bad branches
            if instance.trace_tours([current_path]) + instance.dimension - len(current_path) < best_cost: 
           
                # If the current path is a solution
                if len(current_path) == instance.dimension:
                    best_path = current_path
                    best_cost = instance.trace_tours([current_path])

                # If the current path is not a solution
                elif len(current_path) < instance.dimension - 1:

                    # For each unvisited node in the graph
                    for node in instance.get_nodes():
                        if node not in current_path:
                            # Create a new path to explore
                            s.put(current_path + [node])
                else:
                    s.put(current_path)
        return best_path, self.trace_tours([best_path])

    def _nearest_neighbor_solve(self, instance:StandardProblem):
        unvisited = [node for node in instance.get_nodes() if node != 1]
        print("unvisited")
        visited = [1]
        while unvisited:
            last = visited[-1]
            neighbors = [instance.get_nodes()].sort(key=lambda x: instance.get_weight(last, x), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    break
            else:
                raise ValueError('No unvisited neighbors')
            visited.append(neighbor)
            unvisited.remove(neighbor)
        return visited, instance.trace_tours([visited])

    def _simulated_annealing_solve(
        self, 
        instance:StandardProblem, 
        initial_temperature:float=100, 
        final_temperature:float=1,
        cooling_rate:float=.5
    ):
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
        path = list(instance.get_nodes())
        shuffle(path)
        cost = instance.trace_tours([path])

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
                cost = instance.trace_tours([path])
                # logging.debug(f"New cost: {cost}")

            # Cooling
            temperature *= cooling_rate

        # Return the best path
        return path, instance.trace_tours([path])[0]
        #! Important: Not yet implemented
        # TODO: Implement GA
    def _genetic_algorithm_solve(
        self, 
        instance:StandardProblem, 
        population_size:int=100, 
        mutation_rate:float=.1, 
        max_iterations:int=100
    ):
        
        # initialize
        population = [sample(list(instance.get_nodes()), len(instance.get_nodes())) for _ in population_size]

    def _2_opt_solve(self, instance: StandardProblem, initial_tour:list):

        # define local utils
        def swap(tour, first_index, other_index):
            tour[first_index:other_index] = tour[other_index - 1: first_index - 1: -1]
            return tour
        best = initial_tour
        should_exit = False
        while not should_exit:
            should_exit = True
            # chose two edges to swap
            for first_index in range(1, instance.dimension - 2):
                for other_index in range(first_index + 1, instance.dimension):
                    if other_index - first_index == 1: continue # if nodes are adjacent do nothing
                    if instance.get_weight(first_index, first_index + 1) + instance.get_weight(other_index, other_index + 1) >\
                       instance.get_weight(first_index, other_index) + instance.get_weight(first_index + 1, other_index + 1):
                       swapped = swap(best, first_index, other_index)
                       should_exit = False
            best = swapped
        return best, instance.trace_tours([best])[0]



    def solve(self, instance:StandardProblem, method:str="simulated_annealing", benchmark:bool=False, *args, **kwargs):
        
        if method == "branch_and_bound":
            path, cost = self._branch_and_bound_solve(instance, *args, **kwargs)
        elif method == "nearest_neighbor":
            path, cost = self._nearest_neighbor_solve(instance)
        elif method == "2-opt":
            path, cost = self._2_opt_solve(instance, self._nearest_neighbor_solve(instance)[0])
        elif method == "simulated_annealing":
            path, cost = self._simulated_annealing_solve(instance, *args, **kwargs)
        elif method == "genetic_algorithm":
            path, cost = self._genetic_algorithm_solve(instance, *args, **kwargs)
        else:
            raise ValueError(f"Method {method} is not implemented")

        return path, cost
       

    def __call__(self, *args, **kwds) -> StandardProblem:
        return self.solve(*args, **kwds)

    
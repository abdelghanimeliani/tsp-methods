from crypt import methods
import logging
from tsplib95 import load
from tsplib95.distances import euclidean, manhattan, maximum
from solver import Solver

def main():
    solver = Solver()
    instance = load('benchmarks/ulysses22.tsp')
    solution = load('benchmarks/ulysses22.opt.tour')
    path, cost = solver(instance, method='PFA', n_initial=1000, max_iter=10)
    opt_path = solution.as_keyword_dict()["TOUR_SECTION"][0]
    opt_cost = instance.trace_tours([opt_path])[0]
    print(f"{path=}\n{cost=}")
    print(f"{opt_path=}\n{opt_cost=}")

if __name__ == '__main__':
    main()
from crypt import methods
import logging
from tsplib95 import load
from tsplib95.distances import euclidean, manhattan, maximum
from solver import Solver

def main():
    solver = Solver()
    instance = load('benchmarks/att48.tsp')
    path, cost = solver(instance, method='PFA', n_initial=1000, max_iter=1000)
    print(f"{path=}\n{cost=}")

if __name__ == '__main__':
    main()
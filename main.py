import logging
from tsplib95 import load
from tsplib95.distances import euclidean, manhattan, maximum
from solver import Solver

def main():
    solver = Solver(method='2-opt')
    instance = load('benchmarks/att48.tsp')
    path, cost = solver(instance)
    print(f"{path=}\n{cost=}")

if __name__ == '__main__':
    main()
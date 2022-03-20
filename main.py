from graph import Node
from branch_bound import Branch_and_Bound_Graph
from exact import Exact_TSP_Graph
import tsplib95 as tsp
from numpy import sqrt
from time import time
class Graph(Branch_and_Bound_Graph, Exact_TSP_Graph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

def main():  
    with open('test.tsp') as f:
        lines = f.readlines()[8:]
        node_coords = []
        for line in lines:
            _, x, y = [float(s) for s in line.split(' ')]
            node_coords.append((x, y))
        g = Graph(
            nodes=[Node(chr(ord('A') + i)) for i in range(len(node_coords))],
            edges={
                (chr(ord('A') + i), chr(ord('A') + j)) : sqrt(
                    (node_coords[i][0] - node_coords[j][0]) ** 2 + (node_coords[i][1] - node_coords[j][1]) ** 2
                )
                for i in range(len(node_coords)) for j in range(len(node_coords))
            }
        ) 
    tic = time()
    print(f"B&B solution: {g.branch_and_bound('A')}")
    toc = time()
    print(f"temps: {toc - tic}")

    tic = time()
    print(f"Exact solution: {g.exact_solution('A')}")
    toc = time()
    print(f"temps: {toc - tic}")


if __name__ == '__main__':
    main()
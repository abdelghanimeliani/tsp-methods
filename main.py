from graph import Node
from branch_bound import Branch_and_Bound_Graph
from exact import Exact_TSP_Graph

class Graph(Branch_and_Bound_Graph, Exact_TSP_Graph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

def main():  
    g = Graph(
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
    print(f"Exact solution: {g.exact_solution('A')}")
    print(f"B&B solution: {g.branch_and_bound('A')}")

if __name__ == '__main__':
    main()
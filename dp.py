from graph import Base_Graph, Node
from typing import List, Tuple
from numpy import min

class Dynamic_Programming_Graph(Base_Graph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def dynamic_progrmming(self, origin: Node, subset: list) -> Tuple[List[Node], float]:
        cost = 0.0
        visited = [origin]
        if len(subset) == 2:
            cost = self.get_distance(visited[0], visited[1])
            return visited, cost

        for j in subset:
            cost = min(
                [
                    self.dynamic_progrmming(j, subset.copy().remove(i)) + self.get_distance(i, j)
                    for i in subset
                    if i not in [j, origin]
                ]
            )

            visited.append(j)

        return visited, cost
from graph import Base_Graph, Node
from itertools import product
from queue import PriorityQueue


class Opt2Graph(Base_Graph):
    def __init__(self, nodes, edges) -> None:
        super().__init__(nodes, edges)
        self.que = PriorityQueue()
    def arcExist(self,i,j):
        if self.graphe[i][j]!=None:
            return True
        return False
    def longueur(self,seq):
        return len(seq)
    def distance(self,seq):
        cout=0
        if self.longueur(seq)>1:
            for i in range(self.longueur(seq)-1):
                if self.arcExist(seq[i],seq[i+1]):
                    cout+=self.graphe[seq[i]][seq[i+1]]
        return cout
    
    def two_opt(self, points):
        for i in range(len(points) - 1):
            for j in range(i + 2, len(points) - 1):
                if self.dist(points[i], points[i+1]) + self.dist(points[j], points[j+1]) > self.dist(points[i], points[j]) + self.dist(points[i+1], points[j+1]):
                    points[i+1:j+1] = reversed(points[i+1:j+1])
    def dist(self, a, b):
        if self.arcExist(a,b):
            return self.graphe[a][b]
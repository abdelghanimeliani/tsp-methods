from functools import reduce
import permutation as perm
import sympy.combinatorics as sym
from random import shuffle


def distance(sigma1: list, sigma2: list) -> int:
    """
    returns the Cayley distance between two permutations
    """
    n = len(sigma1)
    if n != len(sigma2):
        raise IndexError("Permutations of different sizes")
    s1 = perm.Permutation(*sigma1)
    s2 = perm.Permutation(*sigma2)

    return n - len((s1 * s2.inverse()).to_cycles())

def step(sigma1:list, sigma2:list, n_steps: int) -> list:
    s1 = perm.Permutation(*sigma1)
    s2 = perm.Permutation(*sigma2)
    comp = s2 * s1.inverse() if n_steps < 0 else s1 * s2.inverse()
    comp = comp.to_cycles()
    trans = reduce(
        lambda x, y: x * y, 
        [sym.Permutation(*cycle) for cycle in comp], 
        sym.Permutation(list(range(len(sigma1))))
    ).transpositions()
    disp = sym.Permutation(list(range(len(sigma1))))
    for i in range(min(n_steps, len(sigma1))):
        disp *= sym.Permutation(*trans[i])
    
    return disp(sigma1)
    

def get_initial_solutions(n_solutions, size):
    x = [list(range(1, size+1)) for _ in range(n_solutions)]
    for s in x:
        shuffle(s)
    return x

def update(*args):
    pass

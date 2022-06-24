from functools import reduce
import permutation as perm
import sympy.combinatorics as sym
from random import choice, shuffle


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
    disp = sigma1
    for i in range(min(n_steps, len(trans))):
        print(f"{disp=}")
        disp = sym.Permutation(*trans[i])(disp)
    
    return disp
    

def get_initial_solutions(n_solutions, size):
    x = [list(range(1, size+1)) for _ in range(n_solutions)]
    for s in x:
        shuffle(s)
    # print(x)
    return x

def update(solution, population, path_finder, alpha, beta, epsilon, r):
    updater = choice(population)
    res = list(range(1, len(population)+1))
    res = step(res, updater, alpha*r[1])
    res = step(res, path_finder, beta*r[2])
    return res

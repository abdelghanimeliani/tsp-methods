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
    return []

def get_initial_solutions():
    return []

def update(*args):
    pass

permu = sym.Permutation(3, 10)
print(permu(list(range(11))))
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps
import dimod


# EXERCISE 1
def h_ising(J, h, sigma):
    h = -np.dot(h, sigma)
    h -= np.einsum("i,i,i", sigma[:-1], J, sigma[1:])
    return h


# EXERCISE 2
def random_antiferromagnetic_ising(n):
    J = -np.random.rand(n-1)
    h = np.zeros(n)
    return J, h


def ising_solution(J, h):
    solution = ""
    n = len(h)
    e_min = 10 ^ 12  # some large number
    for i in range(2**n):
        sigma = [2*int(x)-1 for x in bin(i)[2:].zfill(n)]
        e = h_ising(J, h, sigma)
        if e < e_min:
            e_min = e
            solution = sigma
    return solution, e_min

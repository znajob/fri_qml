from pennylane import numpy as np
from friqml.utils import eps, sz, sx
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
    solution = None
    n = len(h)
    e_min = np.inf  # some large number
    es = []
    for i in range(2**n):
        sigma = [2*int(x)-1 for x in bin(i)[2:].zfill(n)]
        e = h_ising(J, h, sigma)
        es.append(e)
        if e < e_min:
            e_min = e
            solution = sigma
    es = sorted(es)
    return solution, es


def dimod_qubo_ising(J, h):
    a = -h
    b = np.diag(-J, 1)
    qbm = dimod.BinaryQuadraticModel(a, b, "SPIN")
    return qbm

# The exercise is solved as follows
# n=3
# J,h = random_antiferromagnetic_ising(n)
# sigma,es=ising_solution(J,h)
# model = dimod_qubo_ising(J,h)
# sampleset = dimod.SimulatedAnnealingSampler().sample(model,num_reads=100)


# EXERCISE 3
def classical_ising(J, h):
    H = 0
    n = len(h)
    for i in range(n):
        H -= h[i]*sz(i, n)

    for i in range(n-1):
        H -= J[i]*sz(i, n)@sz(i+1, n)
    return H


# EXERCISE 4
def transverse_ising(J, hz, hx):
    n = len(hz)
    H = classical_ising(J, hz)
    for i in range(n):
        H -= hx[i]*sx(i, n)
    return H

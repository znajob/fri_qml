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
def sz(i, n):
    assert i < n, "i has to be smaller than n"
    return np.kron(np.kron(np.eye(2**(i)), np.array([[1, 0], [0, -1]])), np.eye(2**(n-i-1)))


def classical_ising(J, h):
    H = 0
    n = len(h)
    for i in range(n):
        H -= h[i]*sz(i, n)

    for i in range(n-1):
        H -= J[i]*sz(i, n)@sz(i+1, n)
    return H


# EXERCISE 4
def sx(i, n):
    assert i < n, "i has to be smaller than n"
    return np.kron(np.kron(np.eye(2**(i)), np.array([[0, 1], [1, 0]])), np.eye(2**(n-i-1)))


def transverse_ising(J, hz, hx):
    n = len(hz)
    H = classical_ising(J, hz)
    for i in range(n):
        H -= hx[i]*sx(i, n)
    return H

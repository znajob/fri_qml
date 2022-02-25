import pennylane as qml
from pennylane import numpy as np

eps = 1e-13


def random_state_unnormalized(n=4):
    x = np.random.rand(n)+1j*np.random.rand(n)
    return x


def random_state_normalized(n=4):
    x = random_state_unnormalized(n)
    x /= np.linalg.norm(x)
    return x

import pennylane as qml
from pennylane import numpy as np


def random_state_unnormalized(n=4):
    x = np.random.rand()+1j*np.random.rand()
    return x


def random_state_normalized(n=4):
    x = random_state_unnormalized(n)
    x /= np.linalg.norm(x)
    return x


def single_qubit_device():
    return qml.device('default.qubit', wires=1, shots=None)


THETA = np.pi/4
PHI = np.pi/3


def hidden_preparation():
    qml.RX(THETA, wires=0)
    qml.RZ(PHI, wires=0)


def check_hidden_state(psi):
    pass

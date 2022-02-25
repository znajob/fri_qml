import pennylane as qml
from pennylane import numpy as np


# EXERCISE 5
e5_THETA = np.pi/4
e5_PHI = np.pi/3
def e5_hidden_preparation(wires=0):
    qml.RY(e5_THETA, wires=0)
    qml.RZ(e5_PHI, wires=0)


def e5_check_hidden_state(psi):
    psi0 = [np.cos(e5_THETA/2), np.exp(-1j*e5_PHI)*np.sin(e5_THETA/2)]
    return eps > np.abs(1-np.abs(np.dot(np.conj(psi0), psi)))

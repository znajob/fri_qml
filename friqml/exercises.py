import pennylane as qml
from pennylane import numpy as np


# EXERCISE 5
def e5_hidden_preparation(wires=0):
    THETA = np.pi/4
    PHI = np.pi/3
    qml.RY(THETA, wires=0)
    qml.RZ(PHI, wires=0)


def e5_check_hidden_state(psi):
    psi0 = [np.cos(THETA/2), np.exp(-1j*PHI)*np.sin(THETA/2)]
    return eps > np.abs(1-np.abs(np.dot(np.conj(psi0), psi)))

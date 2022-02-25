import pennylane as qml
from pennylane import numpy as np


def single_qubit_device():
    return qml.device('default.qubit', wires=1, shots=None)


THETA = np.pi/4
PHI = np.pi/3


def hidden_preparation():
    qml.RX(THETA, wires=0)
    qml.RZ(PHI, wires=0)


def check_hidden_state(psi):
    pass

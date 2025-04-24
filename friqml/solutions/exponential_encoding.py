from itertools import product
import pennylane as qml
from pennylane import numpy as np


# EXERCISE 1
def variational_layer(weights, wires):
    """
    Apply a simple variational circuit with RX,RY,RZ rotations and CNOTs.

    Args:
        weights (array): shape (n_qubits,)
        wires (list[int]): list of qubit indices
    """
    n_qubits = len(weights)
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=wires[i])
        qml.RY(weights[i, 1], wires=wires[i])
        qml.RZ(weights[i, 2], wires=wires[i])
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])


# EXERCISE 2
def exponential_encoding(x, wires):
    """
    Encode input x using exponentially scaled RX rotations.

    Args:
        x (float): input value
        wires (list[int]): list of qubit indices
    """
    for i, w in enumerate(wires):
        qml.RX(3**i * x, wires=w)


def parallel_pauli_encoding(x, wires):
    """
    Encode input x using exponentially scaled RX rotations.

    Args:
        x (float): input value
        wires (list[int]): list of qubit indices
    """
    for i, w in enumerate(wires):
        qml.RX(2*np.pi * x, wires=w)


# EXERCISE 3
def pqc_exponential(x, weights):
    n_qubits = len(weights[0])
    variational_layer(weights[0], range(n_qubits))
    for wl in weights[1:]:
        exponential_encoding(x, range(n_qubits))
        variational_layer(wl, range(n_qubits))
    return qml.expval(qml.PauliZ(0))


def pqc_parallel_pauli(x, weights):
    n_qubits = len(weights[0])
    variational_layer(weights[0], range(n_qubits))
    for wl in weights[1:]:
        parallel_pauli_encoding(x, range(n_qubits))
        variational_layer(wl, range(n_qubits))
    return qml.expval(qml.PauliZ(0))

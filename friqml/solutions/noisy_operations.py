
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps


# EXERCISE 1
def e1_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.BitFlip(p, wires=0)
    qml.BitFlip(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


# EXERCISE 2
def e2_circuit(p):
    qml.Hadamard(wires=0)
    qml.DepolarizingChannel(p, wires=0)
    qml.Hadamard(wires=0)
    return qml.probs(wires=[0])


def e2_circuit_bf(p):
    qml.Hadamard(wires=0)
    qml.BitFlip(p, wires=0)
    qml.Hadamard(wires=0)
    return qml.probs(wires=[0])

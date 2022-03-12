
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps

# EXERCISE 1


def cU(alpha, beta, gamma, delta, wires=[0, 1]):
    w0 = wires[0]
    w1 = wires[1]
    # A
    qml.RZ(beta, wires=w1)
    qml.RY(gamma/2., wires=w1)

    # CNOT
    qml.CNOT(wires=[w0, w1])

    # B
    qml.RY(-gamma/2, wires=w1)
    qml.RZ(-(delta+beta)/2, wires=w1)

    # CNOT
    qml.CNOT(wires=[w0, w1])

    # C
    qml.RZ((delta-beta)/2, wires=w1)


# EXERCISE 2
def teleportation(wires=[0, 1, 2]):
    w0 = wires[0]
    w1 = wires[1]
    w2 = wires[2]

    # Entengling auxiliary qubits
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[w1, w2])

    # Change of basis
    qml.CNOT(wires=[w0, w1])
    qml.Hadamard(wires=0)

    # Conditional rotations
    qml.CNOT(wires=[w1, w2])
    qml.CZ(wires=[w0, w2])

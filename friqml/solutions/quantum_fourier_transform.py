
import pennylane as qml
from pennylane import numpy as np


# EXERCISE 1
def amplitude_encoding(x, wires=[0, 1]):
    # Normalize the vector to represent a quantum state
    y = x/np.linalg.norm(x)
    x0 = y[0]
    x1 = y[1]
    x2 = y[2]
    x3 = y[3]
    fi0 = np.arccos(np.sqrt(x0**2+x1**2))
    fi1 = np.arctan2(x1, x0)
    fi2 = np.arctan2(x3, x2)
    fi3 = (fi2-fi1)
    qml.RY(2*fi0, wires=wires[0])
    qml.RY(2*fi1, wires=wires[1])
    U = np.array([[np.cos(fi3), -np.sin(fi3)], [np.sin(fi3), np.cos(fi3)]])
    qml.ControlledQubitUnitary(U, control_wires=[wires[0]], wires=wires[1])


# EXERCISE 2
def qft_rot(k):
    return 2*np.pi/2**k


def qft(wires=[0, 1]):
    qml.Hadamard(wires=wires[0])
    qml.ControlledPhaseShift(qft_rot(2), wires=[wires[1], wires[0]])
    qml.Hadamard(wires=wires[1])
    qml.SWAP(wires=[wires[0], wires[1]])


# EXERCISE 3
def iqft(wires=[0, 1]):
    qml.SWAP(wires=[wires[0], wires[1]])
    qml.Hadamard(wires=wires[1])
    qml.ControlledPhaseShift(-qft_rot(2), wires=[wires[1], wires[0]])
    qml.Hadamard(wires=wires[0])

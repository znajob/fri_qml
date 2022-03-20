import pennylane as qml
from pennylane import numpy as np
from friqml.solutions.quantum_fourier_transform import iqft, qft


# EXERCISE 1
def powers_of_unitary(U, wires=[0, 1, 2]):
    w0, w1, w2 = wires
    qml.Hadamard(wires=w0)
    qml.Hadamard(wires=w1)
    U0 = U
    qml.ControlledQubitUnitary(U0, control_wires=w1, wires=w2)
    U1 = np.linalg.matrix_power(U, 2**1)
    qml.ControlledQubitUnitary(U1, control_wires=w0, wires=w2)


# EXERCISE 2
def qpe(U, wires=[0, 1, 2]):
    powers_of_unitary(U, wires=wires)
    iqft(wires=[0, 1])


# EXERCISE 3
def ipowers_of_unitary(U, wires=[0, 1, 2]):
    Ud = np.conjugate(np.transpose(U))
    w0, w1, w2 = wires
    U1 = np.linalg.matrix_power(Ud, 2**1)
    qml.ControlledQubitUnitary(U1, control_wires=w0, wires=w2)

    U0 = Ud
    qml.ControlledQubitUnitary(U0, control_wires=w1, wires=w2)

    qml.Hadamard(wires=w0)
    qml.Hadamard(wires=w1)


def iqpe(U, wires=[0, 1, 2]):
    qft(wires=wires[:2])
    ipowers_of_unitary(U, wires=wires)

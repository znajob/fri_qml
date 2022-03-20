import pennylane as qml
from pennylane import numpy as np


# EXERCISE 1
def rotation(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def mm(A, wires=[0, 1, 2, 3]):
    # qubit 3 holds the states b and Ab
    # qubits 1,2 are used for QPE
    # qubit 0 is used for postselection
    w0, w1, w2, w3 = wires
    U = expm(2*np.pi*1j*A)
    qpe(U, wires=[w1, w2, w3])
    qml.ControlledQubitUnitary(rotation(
        np.arcsin(0/4.)), control_wires=[w1, w2], wires=w0, control_values='00')
    qml.ControlledQubitUnitary(rotation(
        np.arcsin(1/4.)), control_wires=[w1, w2], wires=w0, control_values='01')
    qml.ControlledQubitUnitary(rotation(
        np.arcsin(2/4.)), control_wires=[w1, w2], wires=w0, control_values='10')
    qml.ControlledQubitUnitary(rotation(
        np.arcsin(3/4.)), control_wires=[w1, w2], wires=w0, control_values='11')
    iqpe(U, wires=[w1, w2, w3])


# EXERCISE 2
def hhl(A, wires=[0, 1, 2, 3]):
    # qubit 3 holds the states b and Ab
    # qubits 1,2 are used for QPE
    # qubit 0 is used for postselection
    w0, w1, w2, w3 = wires
    U = expm(2*np.pi*1j*A)
    qpe(U, wires=[w1, w2, w3])
    # qml.ControlledQubitUnitary(rotation(np.arcsin(0/4.)),control_wires=[w1,w2],wires=w0, control_values='00') We can avoid this since the matrix a should be invertable
    qml.ControlledQubitUnitary(rotation(np.arcsin(1.)), control_wires=[
                               w1, w2], wires=w0, control_values='01')
    qml.ControlledQubitUnitary(rotation(
        np.arcsin(1/2.)), control_wires=[w1, w2], wires=w0, control_values='10')
    qml.ControlledQubitUnitary(rotation(
        np.arcsin(1/3.)), control_wires=[w1, w2], wires=w0, control_values='11')
    iqpe(U, wires=[w1, w2, w3])

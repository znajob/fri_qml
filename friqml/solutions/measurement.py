
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps
from friqml.exercises.measurement import hidden_preparation


# EXERCISE 1
def e1_circuit1():
    hidden_preparation(wires=0)
    return qml.expval(qml.PauliX(0))


def e1_circuit2():
    hidden_preparation(wires=0)
    return qml.expval(qml.PauliY(0))


def e1_circuit3():
    hidden_preparation(wires=0)
    return qml.expval(qml.PauliZ(0))


def e1_reconstruct_hidden_state_dev(dev):
    circuit1 = qml.qnode(dev)(e1_circuit1)
    circuit2 = qml.qnode(dev)(e1_circuit2)
    circuit3 = qml.qnode(dev)(e1_circuit3)
    mx = circuit1()
    my = circuit2()
    mz = circuit3()

    phi = np.arctan2(my, mx)
    theta = np.arctan2(np.sqrt(mx**2+my**2), mz)
    psi = np.array([np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)])
    return psi

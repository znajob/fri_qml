
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps
from friqml.exercises.measurement import hidden_preparation


# EXERCISE 1
def e1_circuit_x():
    qml.RY(np.pi/2, wires=0)
    return qml.sample(qml.PauliX(wires=0))


def e1_circuit_z():
    qml.RY(np.pi/2, wires=0)
    return qml.sample(qml.PauliZ(wires=0))


# EXERCISE 2
def e2_circuit_zz():
    qml.RY(np.pi/2, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.sample(qml.PauliZ(wires=0)), qml.sample(qml.PauliZ(wires=1))


def e2_circuit_zx():
    qml.RY(np.pi/2, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.sample(qml.PauliZ(wires=0)), qml.sample(qml.PauliX(wires=1))


# EXERCISE 3
def e3_circuit1():
    hidden_preparation(wires=0)
    return qml.expval(qml.PauliX(0))


def e3_circuit2():
    hidden_preparation(wires=0)
    return qml.expval(qml.PauliY(0))


def e3_circuit3():
    hidden_preparation(wires=0)
    return qml.expval(qml.PauliZ(0))


def e3_reconstruct_hidden_state(dev):
    circuit1 = qml.qnode(dev)(e3_circuit1)
    circuit2 = qml.qnode(dev)(e3_circuit2)
    circuit3 = qml.qnode(dev)(e3_circuit3)
    mx = circuit1()
    my = circuit2()
    mz = circuit3()

    phi = np.arctan2(my, mx)
    theta = np.arctan2(np.sqrt(mx**2+my**2), mz)
    psi = np.array([np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)])
    return psi

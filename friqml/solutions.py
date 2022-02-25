
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps
from friqml.exercises import e5_hidden_preparation


# EXERCISE 1
def is_quantum_state(psi):
    return eps > np.abs(np.linalg.norm(psi)-1)


# EXERCISE 2
def e2_circuit():
    return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))


# EXERCISE 3
def e3_circuit(phi, theta):
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=0)
    return qml.expval(qml.PauliX(0))


# EXERCISE 4
def compare_states(psi1, psi2):
    if not is_quantum_state(psi1) or not is_quantum_state(psi2):
        return False
    if len(psi1) != len(psi2):
        return False
    return eps > np.abs(np.abs(np.dot(np.conj(psi1), psi2))-1)


# EXERCISE 5
def e5_circuit1():
    e5_hidden_preparation(wires=0)
    return qml.expval(qml.PauliX(0))


def e5_circuit2():
    e5_hidden_preparation(wires=0)
    return qml.expval(qml.PauliY(0))


def e5_circuit3():
    e5_hidden_preparation(wires=0)
    return qml.expval(qml.PauliZ(0))


def e5_reconstruct_hidden_state_dev(dev):
    circuit1 = qml.qnode(dev)(e5_circuit1)
    circuit2 = qml.qnode(dev)(e5_circuit2)
    circuit3 = qml.qnode(dev)(e5_circuit3)
    mx = circuit1()
    my = circuit2()
    mz = circuit3()

    phi = np.arctan2(my, mx)
    theta = np.arctan2(np.sqrt(mx**2+my**2), mz)
    psi = np.array([np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)])
    return psi


# EXERCISE 6
def is_entangled(psi):
    rho = np.outer(np.conj(psi), psi)
    rho1 = rho[:2, :2]+rho[2:, 2:]
    return np.abs(np.trace(np.linalg.matrix_power(rho1, 2))-1) > eps

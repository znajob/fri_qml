
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps


# EXERCISE 1
def is_entangled(psi):
    rho = np.outer(np.conj(psi), psi)
    rho1 = rho[:2, :2]+rho[2:, 2:]
    return np.abs(np.trace(np.linalg.matrix_power(rho1, 2))-1) > eps


# EXERCISE 2
def e2_circuit():
    qml.RY(-np.pi/2, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


# EXERCISE 3: Entanglement game
def proj1(phi):
    psi = np.array([np.cos(phi), np.sin(phi)])
    return psi


def proj2(phi):
    psi = np.array([-np.sin(phi), np.cos(phi)])
    return psi


def observable_a(a):
    if a == 0:
        Pi0 = proj1(np.pi/4)
        Pi1 = proj1(-np.pi/4)
    else:
        Pi0 = proj1(np.pi/8)
        Pi1 = proj2(np.pi/8)
    H = Pi0-Pi1
    return qml.Hamiltonian(H)


def observable_b(b):
    if b == 0:
        Pi0 = proj1(np.pi/4)
        Pi1 = proj1(-np.pi/4)
    else:
        Pi0 = proj1(np.pi/8)
        Pi1 = proj2(np.pi/8)
    H = Pi0-Pi1
    return qml.Hamiltonian(H)


def entangled_phi():
    qml.RY(np.pi/2, wires=0)
    qml.CNOT(wires=[0, 1])


def eg_circuit(a, b):
    entangled_phi()
    return qml.sample(observable_a(a), wires=[0]), qml.sample(observable_b(b), wires=[1])

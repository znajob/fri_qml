
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps


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


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

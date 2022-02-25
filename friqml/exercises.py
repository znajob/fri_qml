import pennylane as qml
from pennylane import numpy as np

THETA = np.pi/4
PHI = np.pi/3
def hidden_preparation(wires=0):
    qml.RY(THETA, wires=0)
    qml.RZ(PHI, wires=0)

def check_hidden_state(psi):
    print(PHI/np.pi,THETA/np.pi)
    psi0=[np.cos(THETA/2),np.exp(-1j*PHI)*np.sin(THETA/2)]
    print(psi0)
    print(psi)
    return eps>np.abs(1-np.abs(np.dot(np.conj(psi0),psi)))

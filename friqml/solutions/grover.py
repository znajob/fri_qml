from pennylane import numpy as np
import pennylane as qml


# EXERCISE 1
def equal_superposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


# EXERCISE 2
def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires)


# EXERCISE 3
def diffusion_operator_two_qubits(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)
        qml.PauliZ(wires=wire)
    qml.ctrl(qml.PauliZ, 0)(wires=1)
    for wire in wires:
        qml.Hadamard(wires=wire)


# EXERCISE 4
def diffusion_operator(wires):
    ctrl_values = [0] * (len(wires) - 1)

    for wire in wires[:-1]:
        qml.Hadamard(wire)

    qml.PauliZ(wires[-1])
    qml.MultiControlledX(
        control_values=ctrl_values,
        wires=wires)

    qml.PauliZ(wires[-1])

    for wire in wires[:-1]:
        qml.Hadamard(wire)

    qml.GlobalPhase(np.pi, wires)

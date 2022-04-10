import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps, sz, sx


# EXERCISE 1
def prepare_ancilla():
    qml.Hadamard(0)
    qml.Hadamard(1)


def prepare_example_1():
    qml.ControlledQubitUnitary([[0, 1], [1, 0]], control_wires=[
                               0, 1], wires=2, control_values="10")


def prepare_example_2():
    fi = np.pi/4
    qml.ControlledQubitUnitary([[np.cos(fi), -np.sin(fi)], [np.sin(fi), np.cos(fi)]],
                               control_wires=[0, 1], wires=2, control_values="11")


def prepare_class():
    qml.ControlledQubitUnitary([[0, 1], [1, 0]], control_wires=[
                               0, 1], wires=3, control_values="11")
    qml.ControlledQubitUnitary([[0, 1], [1, 0]], control_wires=[
                               0, 1], wires=3, control_values="01")


def state_preparation():
    prepare_ancilla()
    prepare_example_1()
    prepare_example_2()
    prepare_class()


# EXERCISE 2
def e2_kernel_circuit():
    state_preparation()
    qml.Hadamard(wires=0)
    return qml.sample(wires=[0, 3])


def postselect(samples):
    pacc = np.mean(samples[:, 0])
    ps = np.mean(samples[samples[:, 0] == 0][:, 1])
    return pacc, ps

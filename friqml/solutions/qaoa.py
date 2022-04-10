import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps, sz, sx


# EXERCISE 1 (Solution taken from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html)
# unitary operator U_B with parameters beta and n
def U_B(beta, n):
    for wire in range(n):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameters gamma and graph
def U_C(gamma, graph):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


# EXERCISE 2
pauli_z_2 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def max_cut(betas, gammas, n, graph, edge=None):
    assert len(gammas) == len(
        betas), 'Betas and gammas should have equal length.'
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    n_layers = len(gammas)
    for i in range(n_layers):
        U_C(gammas[i], graph)
        U_B(betas[i], n)
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))


# EXERCISE 3
# def qaoa_maxcut(n_layers, graph, n):
#     print("\np={:d}".format(n_layers))

#     # initialize the parameters near zero
#     init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

#     # minimize the negative of the objective function
#     def objective(params):
#         betas = params[0]
#         gammas = params[1]
#         neg_obj = 0
#         for edge in graph:
#             # objective for the MaxCut problem
#             neg_obj -= 0.5 * \
#                 (1 - circuit_sample(betas, gammas, n, graph, edge=edge))
#         return neg_obj

#     # initialize optimizer: Adagrad works well empirically
#     opt = qml.AdagradOptimizer(stepsize=0.5)

#     # optimize parameters in objective
#     params = init_params
#     steps = 30
#     for i in range(steps):
#         params = opt.step(objective, params)
#         if (i + 1) % 5 == 0:
#             print("Objective after step {:5d}: {: .7f}".format(
#                 i + 1, -objective(params)))

#     # sample measured bitstrings 100 times
#     bit_strings = []
#     n_samples = 100
#     for i in range(0, n_samples):
#         bit_strings.append(bitstring_to_int(circuit_sample(
#             params[0], params[1], n, graph, edge=None)))

#     return -objective(params), bit_strings

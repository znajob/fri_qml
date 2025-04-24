from itertools import product
import pennylane as qml
from pennylane import numpy as np


# EXERCISE 1
paulis_1q = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex)
}


def get_pauli_strings(n):
    """Generate all n-qubit Pauli strings."""
    labels = list(paulis_1q.keys())
    for term in product(labels, repeat=n):
        yield ''.join(term)


def pauli_matrix_from_string(s):
    """Generate full matrix from Pauli string (e.g. 'IXZ')."""
    mat = paulis_1q[s[0]]
    pw = {}
    if s[0] != "I":
        pw[0] = s[0]
    for i, c in enumerate(s[1:]):
        mat = np.kron(mat, paulis_1q[c])
        if c != "I":
            pw[i+1] = c
    return mat


def pauli_word_to_operator(pauli_word, wires):
    assert len(pauli_word) == len(
        wires), "The number of wires sould be the same as the length of tha Pauli string"
    ops = []
    # terms = pauli_word.strip().split()
    op = qml.Identity(wires[0])
    for pauli, wire in zip(pauli_word, wires):
        if pauli == 'I':
            op = op @ qml.Identity(wire)
        elif pauli == 'X':
            op = op @ qml.PauliX(wire)
        elif pauli == 'Y':
            op = op @ qml.PauliY(wire)
        elif pauli == 'Z':
            op = op @ qml.PauliZ(wire)
    return op


def pauli_decompose(matrix):
    """Decomposes a Hermitian matrix into Pauli strings."""
    n = int(np.log2(matrix.shape[0]))
    if matrix.shape != (2**n, 2**n):
        raise ValueError("Matrix must be 2^n x 2^n in size.")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("Matrix must be Hermitian.")

    coeffs = []
    pstrings = []
    for p_str in get_pauli_strings(n):
        P = pauli_matrix_from_string(p_str)
        # real part only; Hermitian ensures imaginary ~0
        alpha = np.trace(P @ matrix).real / (2**n)
        if not np.isclose(alpha, 0):
            coeffs.append(alpha)
            pstrings.append(p_str)
    return coeffs, pstrings


# EXERCISE 2
def prep(alphas, wires):
    qml.StatePrep(alphas, wires=wires, pad_with=0)


def select(pauli_strings, wires, wires_aux):
    a = len(wires_aux)
    for i, pw in enumerate(pauli_strings):
        controls = [int(s) for s in bin(i)[2:].zfill(a)]
        U = pauli_word_to_operator(pw, wires=wires)
        qml.ctrl(U, control=wires_aux, control_values=controls)


# EXERCISE 3
def block_encoding(A, wires=None, wires_aux=None):
    coeffs, pauli_strings = pauli_decompose(A)
    coeffs = np.array(coeffs, dtype=complex)
    coeffs_normalized = np.sqrt(coeffs/np.sum(np.abs(coeffs)))
    n = int(np.log2(len(A)))
    a = int(np.ceil(np.log2(len(coeffs))))

    assert len(wires) == n, f"The number of wires should be {n}."
    assert len(wires_aux) == a, f"The number of auxiliary wires should be {a}."
    assert len(set(wires) & set(
        wires_aux)) == 0, "The wires and wires_aux should not contain similar elements."

    # PREP
    qml.StatePrep(coeffs_normalized, wires_aux, pad_with=0)
    # SELECT
    select(pauli_strings, wires=wires, wires_aux=wires_aux)
    # PREP^\dag
    qml.adjoint(qml.StatePrep(coeffs_normalized, wires_aux, pad_with=0))
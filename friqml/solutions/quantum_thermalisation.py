import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps, sz, sx
import dimod


# EXERCISE 1
def random_ising(n=5):
    a = 2*np.random.rand(n)-1
    b = 2*np.random.rand(n, n)-1
    qbm = dimod.BinaryQuadraticModel(a, b, "SPIN")
    return qbm


def sample_model(model):
    n_samples = 100
    sampler = dimod.SimulatedAnnealingSampler()
    sample_low = sampler.sample(model, num_reads=n_samples, beta_range=None)
    sample_high = sampler.sample(
        model, num_reads=n_samples, beta_range=(0.05, 0.01))
    energies_low = [s[1] for s in sample_low.record]
    energies_high = [s[1] for s in sample_high.record]
    return energies_low, energies_high, sample_low


# EXERCISE 2
def state_preparation(T=0.1, wires=[0, 1]):
    w0, w1 = wires
    theta = np.arctan2(np.exp(0.5/T), np.exp(-0.5/T))
    qml.RY(2*theta, wires=w0)
    qml.CNOT(wires=wires)
    qml.Hadamard(wires=w0)
    qml.Hadamard(wires=w1)


# EXERCISE 3
def reduced_density(rho):
    # rho = np.einsum("i,j->i,j",state,np.conj(state))
    rho1 = np.array([[rho[0, 0]+rho[1, 1], rho[0, 2]+rho[1, 3]],
                    [rho[2, 0]+rho[3, 1], rho[2, 2]+rho[3, 3]]])
    rho2 = rho[:2, :2]+rho[2:, 2:]
    return rho1, rho2

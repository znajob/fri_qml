
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps
from scipy.stats import rv_continuous


# EXERCISE 1
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


# Samples of theta should be drawn from between 0 and pi
sin_sampler = sin_prob_dist(a=0, b=np.pi)


def haar_random_unitary(wires):
    # Sample phi and omega as normal
    phi, omega = 2 * np.pi * np.random.uniform(size=2)
    theta = sin_sampler.rvs(size=1)  # Sample theta from our new distribution
    qml.Rot(phi, theta, omega, wires=wires)


# EXERCISE 2
def e2_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    return qml.sample()

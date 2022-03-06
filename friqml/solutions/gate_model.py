
import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps
from scipy.stats import rv_continuous


# EXERCISE 1
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


def haar_random_unitary():
    # Sample phi and omega as normal
    phi, omega = 2 * np.pi * np.random.uniform(size=2)
    theta = sin_sampler.rvs(size=1)  # Sample theta from our new distribution
    qml.Rot(phi, theta, omega, wires=0)
    return qml.state()

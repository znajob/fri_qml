import cmath
from qutip import Bloch
import matplotlib.pyplot as plt
from pennylane import numpy as np
from friqml.utils import get_vector


def plot_quantum_state(amplitudes):
    """
    Thin function to abstract the plotting on the Bloch sphere.
    """
    bloch_sphere = Bloch()
    vec, _ = get_vector(amplitudes[0], amplitudes[1])
    bloch_sphere.add_vectors(vec)
    bloch_sphere.show()
    # bloch_sphere.clear()


def plot_histogram(counts):
    x = np.arange(len(counts))
    plt.bar(x, counts)
    plt.xticks(x)
    plt.show()


import pennylane as qml
from pennylane import numpy as np
from friqml.utils import eps, sz, sx


# EXERCISE 1
def small_gap_hamiltonian():
  H = -1000*sz(0,3)*sz(1,3) - 0.01*sz(1,3)*sz(2,3) - 0.5*sz(0,3)
  return H

def big_gap_hamiltonian():
  H = -sx(0,3)-sx(1,3)-sx(2,3)
  return H

  
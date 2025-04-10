from scipy.integrate import solve_ivp
from pennylane import numpy as np

# Exercise 1
def transition_probability(T,v,Delta):
  # Time-dependent Hamiltonian
  def H(t):
      eps = v * t
      return 0.5 * np.array([[eps, Delta], [Delta, -eps]])

  # Schrödinger equation: dψ/dt = -i H ψ
  def schrodinger(t, psi):
      return -1j * H(t).dot(psi)

  # Initial state: |0⟩ = [1, 0]
  psi0 = np.array([1.0, 0.0], dtype=complex)

  # Solve the time evolution
  t_span = (-T, T)
  t_eval = np.linspace(*t_span, 500)
  sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval, rtol=1e-8, atol=1e-8)

  # Probabilities
  P0 = np.abs(sol.y[0])**2  # |⟨0|ψ(t)⟩|²
  P1 = np.abs(sol.y[1])**2  # |⟨1|ψ(t)⟩|²
  return P0,P1
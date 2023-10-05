# DIRECTORIES
import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy.polynomial.hermite as Herm
import math

# CALCULATING THE WAVE FUNCTIONS
def hamiltonian(N, dx, V=None):
    # Returns Hamiltonian as a sparse matrix using finite differences.
    L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))
    H = - (hbar**2) * L / (2 * m * dx**2)
    if V is not None:
        H += scipy.sparse.spdiags(V, 0, N, N)
    return H.tocsc()

def time_evolution_operator(H, dt):
    # Time evolution operator given a Hamiltonian and time step.
    U = scipy.linalg.expm(-1j * H * dt / hbar).toarray()
    U[(U.real**2 + U.imag**2) < 1E-10] = 0
    return scipy.sparse.csc_matrix(U)

def simulate(psi, H, dt):
    # Generates wavefunction and time at the next time step.
    U = time_evolution_operator(H, dt)
    t = 0
    while t < max_time:
        probability_densities.append(psi.real**2 + psi.imag**2)
        times.append(t)
        psi = U @ psi
        t += 1

def gaussian_wavepacket(x, x0, sigma0, p0):
    # Gaussian wavepacket at x0 +/- width sigma0, with energy E.
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0*x - ((x - x0)/(2 * sigma0))**2)

# Animation setup
def animation():
    fig, ax = plt.subplots()
    probability_line, = ax.plot(x, probability_densities[0], label="Probability density")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.min(probability_densities), np.max(probability_densities))

    def update(frame):
        probability_line.set_ydata(probability_densities[frame])
        ax.set_title(f"Time: {times[frame]}")
        return probability_line,

    ani = FuncAnimation(fig, update, frames=len(probability_densities), interval=100)
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('Wave function')
    plt.show()

# HARMONIC POTENTIAL
def harmonic_potential(w, x):
    return (w * x)**2 /2

# INITIAL WAVEFUNCTION
def eigenstate_qho(w, n, x):
    def hermite(x, n):
        xi = np.sqrt(m * w / hbar) * x
        herm_coeffs = np.zeros(n + 1)
        herm_coeffs[n] = 1
        return Herm.hermval(xi, herm_coeffs)

    xi = np.sqrt(m * w / hbar) * x
    prefactor = 1. / math.sqrt(2. ** n * math.factorial(n)) * (m * w / (np.pi * hbar)) ** (0.25)
    psi = prefactor * np.exp(- xi ** 2 / 2) * hermite(x, n)
    return psi

# SETTINGS
grid = 500 # number of points
probability_densities = []  #  l of probability densities
times = []  # l of zeiten
max_time = 50
n = 1   # energy level
m = 1   # m
hbar = 1
a = 128 # x position length
x, dx = np.linspace(-a/2, a/2, grid, endpoint=False, retstep=True)
psi0 = gaussian_wavepacket(x, x0=-30, sigma0=3.0, p0=0.0)
V = harmonic_potential(w=1/50, x=x)

# MAIN
H = hamiltonian(grid, dx, V=V)
simulate(psi0, H, dt=5.0)
animation()

print(sum(probability_densities[0])/(grid/a))


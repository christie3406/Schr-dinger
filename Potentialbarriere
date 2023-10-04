import numpy as np
import scipy.sparse
import scipy.linalg
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import root_scalar

# CALCULATING THE WAVE FUNCTIONS 
def hamiltonian(N, dx, V=None):
    # Returns Hamiltonian as a sparse matrix using finite differences.
    L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))    # Laplace operator
    H = - (hbar**2) * L / (2 * m * dx**2)   # Kinetic energy operator
    if V is not None:   # Potential energy by default 0
        H += scipy.sparse.spdiags(V, 0, N, N)   # Potential energy in shape of N,N
    return H.tocsc()

def time_evolution_operator(H, dt):
    # Time evolution operator given a Hamiltonian and time step.
    U = scipy.linalg.expm(-1j * H * dt / hbar).toarray()
    U[(U.real**2 + U.imag**2) < 1E-10] = 0
    return scipy.sparse.csc_matrix(U)

def simulate(psi, H, dt):
    # Generates wave function and time at the next time step.
    U = time_evolution_operator(H, dt)
    t = 0
    while t < max_time:
        probability_densities.append(psi.real**2 + psi.imag**2)
        times.append(t)
        psi = U @ psi
        t += 1

# ANIMATION
def animation():
    fig, ax = plt.subplots()
    probability_line, = ax.plot(x, probability_densities[0], label="Probability density")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.min(probability_densities), np.max(probability_densities))

    # transparent area representing potential barrier
    ax.fill_between([0, 1], 0, np.max(probability_densities), color='gray', alpha=0.5)

    def update(frame):
        probability_line.set_ydata(probability_densities[frame])
        ax.set_title(f"Time: {times[frame]}")
        return probability_line,

    ani = FuncAnimation(fig, update, frames=len(probability_densities), interval=100)
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('Wave function')
    plt.show()

# POTENTIAL BARRIER
def rectangular_potential_barrier(x, V0, w):
    # Rectangular potential barrier of height V0 and width w.
    return np.where((0 <= x) & (x < w), V0, 0.0)

# INITIAL WAVEFUNCTION
def gaussian_wavepacket(x, x0, sigma0, e):
    # Gaussian wavepacket at x0 +/- width sigma0, with energy E.
    p0 = np.sqrt(2 * e)
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0*x - ((x - x0)/(2 * sigma0))**2)

# SETTINGS
grid = 1024 # number of points
probability_densities = []  # l of arrays
times = []  # l of zeiten
real= []
imaginary=[]
max_time = 20
m = 1   # m
hbar = 1
a = 128 # x position length

x, dx = np.linspace(-a/2, a/2, grid, endpoint=False, retstep=True)
e = 10   # Energie des Wellenpakets (1 bis 25)
V0 = 5  # Potentialenergie Barriere (konstant = 5)
psi0 = gaussian_wavepacket(x, x0=-30.0, sigma0=3.0, e=e)
V = rectangular_potential_barrier(x=x, V0=V0, w=1)  # w = Breite der Barriere

# MAIN
H = hamiltonian(grid, dx, V=V)
simulate(psi0, H, dt=1)
animation()


print(sum(probability_densities[0])/(grid/a))

transmission = sum(probability_densities[15][520:-1])/(grid/a)
print(round(transmission, 1))


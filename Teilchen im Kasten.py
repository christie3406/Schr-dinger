import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        real.append(np.real(psi))
        imaginary.append(np.imag(psi))
        times.append(t)
        psi = U @ psi
        t += 1

# ANIMATION
def animation():
    fig, ax = plt.subplots()
    probability_line, = ax.plot(x, probability_densities[0], label="Psi^2")
    #real_line, = ax.plot(x, real[0], label='Re(Psi)')
    #imaginary_line, = ax.plot(x, imaginary[0], label='Im(Psi)')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0,np.max(probability_densities))
    #ax.set_ylim(np.min(imaginary), np.max(real))

    def update(frame):
        probability_line.set_ydata(probability_densities[frame])
        #real_line.set_ydata(real[frame])
        #imaginary_line.set_ydata(imaginary[frame])
        ax.set_title(f"Time: {times[frame]}")

    ani = FuncAnimation(fig, update, frames=len(probability_densities), interval=100)
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('Wave function')
    plt.show()

# INFINITE POTENTIAL
def infinite_potential():
    V = np.zeros(grid)
    V[0] = np.inf
    V[-1] = np.inf
    return V


# INITIAL WAVEFUNCTION
def eigenstate_isw(n, a, x):
    return np.sqrt(2/a)*np.sin(n*np.pi*x/a)

# INITIAL WAVEFUNCTION
def gaussian_wavepacket(x, x0, sigma0, p0):
    # Gaussian wavepacket at x0 +/- width sigma0, with energy E.
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0*x - ((x - x0)/(2 * sigma0))**2)


# SETTINGS
grid = 500 # number of points
probability_densities = []  # l of probability densities
real = []
imaginary = []
times = []  # l of zeiten
max_time = 50
m = 1   # m
hbar = 1
a = 200 # x position length
x, dx = np.linspace(0, a, grid, endpoint=False, retstep=True)

psi0 = gaussian_wavepacket(x, x0=a/2, sigma0=10.0, p0=4)
#eigenstate_isw(n=1, a=a, x=x)
V = infinite_potential()

# MAIN
H = hamiltonian(grid, dx)
simulate(psi0, H, 10.0)
animation()

# Check for normalisation
print(sum(probability_densities[0])/(grid/a))

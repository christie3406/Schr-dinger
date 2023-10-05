import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy.polynomial.hermite as Herm
import math


# BERECHNUNG DER WELLENFUNKTIONEN
def hamiltonian(N, dx, V=None):
    # Gibt den Hamiltonian als dünnbesetzte Matrix unter Verwendung endlicher Differenzen zurück.
    L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))
    H = - (hbar ** 2) * L / (2 * m * dx ** 2)
    if V is not None:
        H += scipy.sparse.spdiags(V, 0, N, N)
    return H.tocsc()


def zeitentwicklungsoperator(H, dt):
    # zeitentwicklungsoperator mit einem Hamiltonian und einem Zeitschritt.
    U = scipy.linalg.expm(-1j * H * dt / hbar).toarray()
    U[(U.real ** 2 + U.imag ** 2) < 1E-10] = 0
    return scipy.sparse.csc_matrix(U)


def simulieren(psi, H, dt):
    # Berechnet Wellenfunktion und Zeit beim nächsten Zeitschritt.
    U = zeitentwicklungsoperator(H, dt)
    t = 0
    while t < max_zeit:
        aufenthaltswahrscheinlichkeit.append(psi.real ** 2 + psi.imag ** 2)
        zeiten.append(t)
        psi = U @ psi
        t += 1


# ANIMATION
def animation():
    fig, ax = plt.subplots()
    aufenthaltswahrscheinlichkeit_linie, = ax.plot(x, aufenthaltswahrscheinlichkeit[0])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1)

    def update(frame):
        aufenthaltswahrscheinlichkeit_linie.set_ydata(aufenthaltswahrscheinlichkeit[frame])
        ax.set_title(f"Time: {zeiten[frame]}")
        return aufenthaltswahrscheinlichkeit_linie,

    FuncAnimation(fig, update, frames=len(aufenthaltswahrscheinlichkeit), interval=100)
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('Aufenthaltswahrscheinlichkeit')
    plt.show()


# ENERGIEPOTENTIALE
def kastenpotential():
    # Unendlich hohe potentielle Energie an den Rändern- für das Teilchen im Kasten
    V = np.zeros(grid)
    V[0] = np.inf
    V[-1] = np.inf
    return V


def harmonisches_potential(w, x):
    # Harmonisches Potenzial mit der Kreisfrequenz w- für den harmonischen Oszillator
    return (w * x) ** 2 / 2




# AUSGANGSWELLENFUNKTIONEN


def eigenzustand_kasten(n, a, x):
    # Eigenzustand des Teilchens im Kasten mit Energie n und Breite a.
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)




def eigenzustand_oszillator(w, n, x):
    # Eigenzustand des quantenharmonischen Oszillators mit der Kreisfrequenz w und der Energie n.
    def hermite(x, n):
        x1 = np.sqrt(m * w / hbar) * x
        herm_coeffs = np.zeros(n + 1)
        herm_coeffs[n] = 1
        return Herm.hermval(x1, herm_coeffs)

    xi = np.sqrt(m * w / hbar) * x
    prefactor = 1. / math.sqrt(2. ** n * math.factorial(n)) * (m * w / (np.pi * hbar)) ** 0.25
    psi = prefactor * np.exp(- xi ** 2 / 2) * hermite(x, n)
    return psi





# EINSTELLUNGEN
grid = 1024  # Anzahl der Punkte von der  Matrix
aufenthaltswahrscheinlichkeit = []  # Liste der Aufenthaltswahrscheinlichkeiten
zeiten = []  # Liste der Zeiten
max_zeit = 10
m = 1  # Masse
hbar = 1  # reduzierte Plank'sche Konstante
a = 128  # Länge der X-Achse



x, dx = np.linspace(0, a, grid, endpoint=False, retstep=True)

def potentialbarriere(x, V0, w):
    # Potentialbarriere mit der Höhe V0 und der Breite w- für den Tunneleffekt
    return np.where((0 <= x) & (x < w), V0, 0.0)

def gausssche_wellenpaket(x, x0, sigma0, e):
    # Gaußsches Wellenpaket bei x0 +/- Breite sigma0 mit Energie E.
    p0 = np.sqrt(2 * e)
    A = (2 * np.pi * sigma0 ** 2) ** (-0.25)
    return A * np.exp(1j * p0 * x - ((x - x0) / (2 * sigma0)) ** 2)

e = 1   # Energie des Wellenpakets (1 bis 25)
V0 = 5  # Potentialenergie Barriere (konstant = 5)
psi0 = eigenzustand_oszillator(w=1, n=1, x=x)  # Ausgangswellenfunktion
V = harmonisches_potential(x=x)  # Potential

# Misst wie viel Prozent der ursprünglichen Welle durch die Barriere gegangen ist
transmission = sum(aufenthaltswahrscheinlichkeit[20][520:-1])/(grid/a)
print(round(transmission, 2))




psi0 = eigenzustand_oszillator(w=1, n=1, x=x)  # Ausgangswellenfunktion
V = harmonisches_potential(x=x)  # Potential

# HAUPTTEIL
H = hamiltonian(grid, dx)
simulieren(psi0, H, 1.0)
animation()

# Normalisierung prüfen- Summe der Aufenthaltswahrscheinlichkeiten sollte 1 geben
print(sum(aufenthaltswahrscheinlichkeit[0]) / (grid / a))


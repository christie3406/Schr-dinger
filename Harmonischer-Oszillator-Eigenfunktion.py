import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy.polynomial.hermite as Herm
import math

# Choose simple units
m = 1.
w = 1.
hbar = 1.
# Discretized space
grid = 1080
x, dx = numpy.linspace(-12, 12, grid, endpoint=False, retstep=True)


def hermite(x, n):
    xi = numpy.sqrt(m * w / hbar) * x
    herm_coeffs = numpy.zeros(n + 1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)


def stationary_state(x, n):
    xi = numpy.sqrt(m * w / hbar) * x
    prefactor = 1. / math.sqrt(2. ** n * math.factorial(n)) * (m * w / (numpy.pi * hbar)) ** (0.25)
    psi = prefactor * numpy.exp(- xi ** 2 / 2) * hermite(x, n)
    return psi


def klassisch(x, n):
    E = hbar * w * (n + 0.5)
    x_max = numpy.sqrt(2 * E / (m * w ** 2))
    P_klassisch = numpy.zeros(x.shape[0])
    x_inside = abs(x) < (x_max - 0.025)
    P_klassisch[x_inside] = 1. / numpy.pi / numpy.sqrt(x_max ** 2 - x[x_inside] * x[x_inside])
    return P_klassisch


fig, ax = plt.subplots()
n = 70
plt.plot(x, (numpy.conjugate(stationary_state(x, n)) * stationary_state(x, n))/45, label="P_psi(x), n=" + str(n))
plt.plot(x, klassisch(x, n) / 45, label="P_kl(x)")
plt.ylabel('Aufenthaltswahrscheinlichkeit')
plt.xlabel('Position x')
plt.legend()
plt.show()

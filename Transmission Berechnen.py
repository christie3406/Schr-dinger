import numpy as np
import matplotlib.pyplot as plt

def transmission_berechnen(E, V0, a):
    if E < V0:
        k1 = np.sqrt((2 * V0 - 2 * E) / hbar **2)
        u = (V0 * np.sinh(k1 * a)) ** 2
        d = 4 * E * (V0 - E)
    elif E > V0:
        k1 = np.sqrt((2 * E - 2 * V0)/hbar **2)
        u = (V0 * np.sinh(k1 * a)) ** 2
        d = 4 * E * (E - V0)
    else:
        return 1/(1 + (a**2 * V0 / 2 * hbar**2))

    T = 1 / (1 + (u / d))
    return T

# Parameter
hbar = 1    # reduzierte Plank'sche Konstante
V0 = 5  #   Potentialenergie Barriere (konstant = 5)
a = 1   # Breite der Barriere (konstant = 1)
m = 1.0  # Masse des Teilchens (konstant = 1)
energie = np.arange(0, 5.5, 0.5)  # Energie des Teilchens (0 bis 5)

# Berechnungen
ratios = energie / V0   # Berechnet Verhätlnis E/V0
transmission_coeffs = [transmission_berechnen(E, V0, a) for E in energie]   # Berechnet Transmissionsfaktor


# Plot the results
plt.plot(ratios, transmission_coeffs, marker='o')
plt.xlabel('E/V0')
plt.ylabel('Transmission T')
plt.title('Transmission des Wellenpakets mit E=<V (Berechnet)')
plt.grid(True)
plt.show()

import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fractions import Fraction

class HomePage:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titel = tk.Label(self.frame, text="Simulation quantenmechanischer Modelle",font=("Arial", 20),)
        self.titel.pack(pady=30)
        self.instruction = tk.Label(self.frame, text="Wählen Sie eines der folgenden Modelle:", font=("Arial", 14))
        self.instruction.pack(pady=30)
        self.button1 = tk.Button(self.frame, text="Teilchen im Kasten", font=("Arial", 18), command=self.open_page1)
        self.button1.pack(pady=30)
        self.button2 = tk.Button(self.frame, text="Quantenharmonischer Oszillator", font=("Arial", 18), command=self.open_page2)
        self.button2.pack(pady=30)
        self.button3 = tk.Button(self.frame, text="Potentialbarriere",font=("Arial", 18), command=self.open_page3)
        self.button3.pack(pady=30)
        self.frame.pack()

    def open_page1(self):
        self.frame.destroy()
        Page1(self.master)

    def open_page2(self):
        self.frame.destroy()
        Page2(self.master)

    def open_page3(self):
        self.frame.destroy()
        Page3(self.master)

class Page1:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)

        self.label = tk.Label(self.frame, text="Teilchen im Kasten", font=("Arial, 18"))
        self.label.pack(padx=10, pady=10)

        self.back_button = tk.Button(self.frame, text="Zurück", font=("Arial, 14"), command=self.back_to_home)
        self.back_button.pack()

        self.real_state = tk.IntVar()
        self.real = tk.Checkbutton(self.frame, text="Realteil von Psi zeigen", font=("Arial", 16),
                                   variable=self.real_state)
        self.real.pack(padx=10, pady=10)

        self.ima_state = tk.IntVar()
        self.ima = tk.Checkbutton(self.frame, text="Imaginärteil von Psi zeigen", font=("Arial", 16),
                                  variable=self.ima_state)
        self.ima.pack(padx=10, pady=10)

        self.a_label = tk.Label(self.frame, text="Breite des Kastens:", font=("Arial", 16))
        self.a_label.pack(padx=10, pady=(20, 0))
        self.a_state = tk.IntVar()
        self.a = tk.Scale(self.frame, variable=self.a_state, from_=50, to=150, orient="horizontal")
        self.a.pack(padx=10, pady=(0, 20))
        self.a.set(100)

        self.s0_label = tk.Label(self.frame, text="Anfangsbreite des Wellenpakets:", font=("Arial", 16))
        self.s0_label.pack(padx=10, pady=(20, 0))
        self.s0_state = tk.DoubleVar()
        self.s0 = tk.Scale(self.frame, variable=self.s0_state, from_=1, to=10, orient="horizontal")
        self.s0.pack(padx=10, pady=(0, 20))
        self.s0.set(3)

        self.p0_label = tk.Label(self.frame, text="Anfangsimpuls des Wellenpakets:", font=("Arial", 16))
        self.p0_label.pack(padx=10, pady=(20, 0))
        self.p0_state = tk.DoubleVar()
        self.p0 = tk.Scale(self.frame, variable=self.p0_state, from_=0, to=5, orient="horizontal")
        self.p0.pack(padx=10, pady=(0, 20))
        self.p0.set(0)

        self.dt_label = tk.Label(self.frame, text="Geschwindigkeit:", font=("Arial", 16))
        self.dt_label.pack(padx=10, pady=(20, 0))
        self.dt_state = tk.IntVar()
        self.dt = tk.Scale(self.frame, variable=self.dt_state, from_=1, to=20, orient="horizontal")
        self.dt.pack(padx=10, pady=(0, 20))
        self.dt.set(10)

        self.start_button = tk.Button(self.frame, text="Starten", font=("Arial", 18), command=self.infinite_square_well)
        self.start_button.pack(padx=10, pady=20)

        self.slow = tk.Label(self.frame, text="Es dauert bis zu 20 Sekunden, bis die Simulation startet.",
                             font=("Arial", 12))
        self.slow.pack(pady=10)

        self.frame.pack()

    def back_to_home(self):
        self.frame.destroy()
        HomePage(self.master)

    def infinite_square_well(self):
        # CALCULATING THE WAVE FUNCTIONS
        def hamiltonian(N, dx, V=None):
            # Returns Hamiltonian as a sparse matrix using finite differences.
            L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))  # Laplace operator
            H = - (hbar ** 2) * L / (2 * m * dx ** 2)  # Kinetic energy operator
            if V is not None:  # Potential energy by default 0
                H += scipy.sparse.spdiags(V, 0, N, N)  # Potential energy in shape of N,N
            return H.tocsc()

        def time_evolution_operator(H, dt):
            # Time evolution operator given a Hamiltonian and time step.
            U = scipy.linalg.expm(-1j * H * dt / hbar).toarray()
            U[(U.real ** 2 + U.imag ** 2) < 1E-10] = 0
            return scipy.sparse.csc_matrix(U)

        def simulate(psi, H, dt):
            # Generates wave function and time at the next time step.
            U = time_evolution_operator(H, dt)
            t = 0
            while t < max_time:
                probability_densities.append(psi.real ** 2 + psi.imag ** 2)
                real.append(np.real(psi))
                imaginary.append(np.imag(psi))
                times.append(t)
                psi = U @ psi
                t += 1

        # ANIMATION


        def animation():
            fig, ax = plt.subplots()
            probability_line, = ax.plot(x, probability_densities[0], label="Psi^2")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, np.max(probability_densities))
            if self.real_state.get() == 1:
                real_line, = ax.plot(x, real[0], label='Re(Psi)')
                ax.set_ylim(np.min(real), np.max(real))
            if self.ima_state.get() == 1:
                imaginary_line, = ax.plot(x, imaginary[0], label='Im(Psi)')
                ax.set_ylim(np.min(imaginary), np.max(imaginary))

            def update(frame):
                probability_line.set_ydata(probability_densities[frame])
                if self.real_state.get() == 1:
                    real_line.set_ydata(real[frame])
                if self.ima_state.get() == 1:
                    imaginary_line.set_ydata(imaginary[frame])
                ax.set_title(f"Time: {times[frame]}")

            ani = FuncAnimation(fig, update, frames=len(probability_densities), interval=100)
            ax.legend()
            plt.xlabel('Position x')
            plt.ylabel('Wellenfunktion')
            plt.show()

        # INFINITE POTENTIAL
        def infinite_potential():
            V = np.zeros(grid)
            V[0] = np.inf
            V[-1] = np.inf
            return V

        # INITIAL WAVEFUNCTION
        def eigenstate_isw(n, a, x):
            return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)

        # INITIAL WAVEFUNCTION
        def gaussian_wavepacket(x, x0, sigma0, p0):
            # Gaussian wavepacket at x0 +/- width sigma0, with energy E.
            A = (2 * np.pi * sigma0 ** 2) ** (-0.25)
            return A * np.exp(1j * p0 * x - ((x - x0) / (2 * sigma0)) ** 2)

        # SETTINGS
        grid = 500  # number of points
        probability_densities = []  # l of probability densities
        real = []
        imaginary = []
        times = []  # l of zeiten
        max_time = 50
        m = 1  # m
        hbar = 1
        a = self.a_state.get()  # x position length
        x, dx = np.linspace(0, a, grid, endpoint=False, retstep=True)

        psi0 = gaussian_wavepacket(x, x0=a / 2, sigma0 = self.s0_state.get(), p0=self.p0_state.get())
        V = infinite_potential()

        # MAIN
        H = hamiltonian(grid, dx)
        simulate(psi0, H, self.dt_state.get())
        animation()

class Page2:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.label = tk.Label(self.frame, text="Harmonischer Oszillator", font=("Arial, 18"))
        self.label.pack(padx=10, pady=10)

        self.back_button = tk.Button(self.frame, text="Zurück", font=("Arial, 14"), command=self.back_to_home)
        self.back_button.pack()

        self.real_state = tk.IntVar()
        self.real = tk.Checkbutton(self.frame, text="Realteil von Psi zeigen", font=("Arial", 16),
                                   variable=self.real_state)
        self.real.pack(padx=10, pady=10)

        self.ima_state = tk.IntVar()
        self.ima = tk.Checkbutton(self.frame, text="Imaginärteil von Psi zeigen", font=("Arial", 16),
                                  variable=self.ima_state)
        self.ima.pack(padx=10, pady=10)

        self.w_label = tk.Label(self.frame, text="Kreisfrequenz w:", font=("Arial", 16))
        self.w_label.pack(padx=10, pady=(20, 0))
        self.w_state = tk.DoubleVar()
        self.w = ttk.Combobox(self.frame, textvariable=self.w_state, width = 30)
        self.w["values"] = (0.1, 0.05, 0.033, 0.025, 0.02, 0.016)
        self.w.pack(padx=10, pady=(0, 20))
        self.w.current(2)

        self.s0_label = tk.Label(self.frame, text="Anfangsbreite des Wellenpakets:", font=("Arial", 16))
        self.s0_label.pack(padx=10, pady=(20, 0))
        self.s0_state = tk.DoubleVar()
        self.s0 = tk.Scale(self.frame, variable=self.s0_state, from_=1, to=10, orient="horizontal")
        self.s0.pack(padx=10, pady=(0, 20))
        self.s0.set(3)

        self.dt_label = tk.Label(self.frame, text="Geschwindigkeit:", font=("Arial", 16))
        self.dt_label.pack(padx=10, pady=(20, 0))
        self.dt_state = tk.IntVar()
        self.dt = tk.Scale(self.frame, variable=self.dt_state, from_=1, to=20, orient="horizontal")
        self.dt.pack(padx=10, pady=(0, 20))
        self.dt.set(5)

        self.start_button = tk.Button(self.frame, text="Starten", font=("Arial", 18), command=self.harmonic_oscillator)
        self.start_button.pack(padx=10, pady=20)

        self.slow = tk.Label(self.frame, text="Es dauert bis zu 20 Sekunden, bis die Simulation startet.",
                             font=("Arial", 12))
        self.slow.pack(pady=10)
        self.frame.pack()

    def back_to_home(self):
        self.frame.destroy()
        HomePage(self.master)

    def harmonic_oscillator(self):
        def hamiltonian(N, dx, V=None):
            # Returns Hamiltonian as a sparse matrix using finite differences.
            L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))
            H = - (hbar ** 2) * L / (2 * m * dx ** 2)
            if V is not None:
                H += scipy.sparse.spdiags(V, 0, N, N)
            return H.tocsc()

        def time_evolution_operator(H, dt):
            # Time evolution operator given a Hamiltonian and time step.
            U = scipy.linalg.expm(-1j * H * dt / hbar).toarray()
            U[(U.real ** 2 + U.imag ** 2) < 1E-10] = 0
            return scipy.sparse.csc_matrix(U)

        def simulate(psi, H, dt):
            # Generates wave function and time at the next time step.
            U = time_evolution_operator(H, dt)
            t = 0
            while t < max_time:
                probability_densities.append(psi.real ** 2 + psi.imag ** 2)
                real.append(np.real(psi))
                imaginary.append(np.imag(psi))
                times.append(t)
                psi = U @ psi
                t += 1

        def gaussian_wavepacket(x, x0, sigma0, p0):
            # Gaussian wavepacket at x0 +/- width sigma0, with energy E.
            A = (2 * np.pi * sigma0 ** 2) ** (-0.25)
            return A * np.exp(1j * p0 * x - ((x - x0) / (2 * sigma0)) ** 2)

        # Animation setup
        def animation():
            fig, ax = plt.subplots()
            probability_line, = ax.plot(x, probability_densities[0], label="Psi^2")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, np.max(probability_densities))
            if self.real_state.get() == 1:
                real_line, = ax.plot(x, real[0], label='Re(Psi)')
                ax.set_ylim(np.min(real), np.max(real))
            if self.ima_state.get() == 1:
                imaginary_line, = ax.plot(x, imaginary[0], label='Im(Psi)')
                ax.set_ylim(np.min(imaginary), np.max(imaginary))

            def update(frame):
                probability_line.set_ydata(probability_densities[frame])
                if self.real_state.get() == 1:
                    real_line.set_ydata(real[frame])
                if self.ima_state.get() == 1:
                    imaginary_line.set_ydata(imaginary[frame])
                ax.set_title(f"Time: {times[frame]}")

            ani = FuncAnimation(fig, update, frames=len(probability_densities), interval=100)
            ax.legend()
            plt.xlabel('Position x')
            plt.ylabel('Wellenfunktion')
            plt.show()

        # HARMONIC POTENTIAL
        def harmonic_potential(w, x):
            return (w * x) ** 2 / 2

        # SETTINGS
        grid = 500  # number of points
        real = []
        imaginary = []
        probability_densities = []  # l of probability densities
        times = []  # l of zeiten
        max_time = 50
        m = 1  # m
        hbar = 1
        a = 128  # x position length
        x, dx = np.linspace(-a / 2, a / 2, grid, endpoint=False, retstep=True)
        psi0 = gaussian_wavepacket(x, x0=-30, sigma0=self.s0_state.get(), p0=0.0)
        V = harmonic_potential(w=self.w_state.get(), x=x)

        # MAIN
        H = hamiltonian(grid, dx, V=V)
        simulate(psi0, H, dt=self.dt_state.get())
        animation()



class Page3:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)

        self.label = tk.Label(self.frame, text="Potentialbarriere", font=("Arial, 18"))
        self.label.pack(padx=10, pady=10)

        self.back_button = tk.Button(self.frame, text="Zurück", font=("Arial, 14"), command=self.back_to_home)
        self.back_button.pack()

        self.real_state = tk.IntVar()
        self.real = tk.Checkbutton(self.frame, text="Realteil von Psi zeigen", font=("Arial", 16),
                                   variable=self.real_state)
        self.real.pack(padx=10, pady=10)

        self.ima_state = tk.IntVar()
        self.ima = tk.Checkbutton(self.frame, text="Imaginärteil von Psi zeigen", font=("Arial", 16),
                                  variable=self.ima_state)
        self.ima.pack(padx=10, pady=10)

        self.v_label = tk.Label(self.frame, text="Energie der Barriere:", font=("Arial", 16))
        self.v_label.pack(padx=10, pady=(20, 0))
        self.v_state = tk.IntVar()
        self.v = tk.Scale(self.frame, variable=self.v_state, from_=1, to=20, orient="horizontal")
        self.v.pack(padx=10, pady=(0, 20))
        self.v.set(7)

        self.e_label = tk.Label(self.frame, text="Energie des Wellenpakets:", font=("Arial", 16))
        self.e_label.pack(padx=10, pady=(20, 0))
        self.e_state = tk.DoubleVar()
        self.e = tk.Scale(self.frame, variable=self.e_state, from_=1, to=20, orient="horizontal")
        self.e.pack(padx=10, pady=(0, 20))
        self.e.set(5)

        self.w_label = tk.Label(self.frame, text="Breite der Barriere:", font=("Arial", 16))
        self.w_label.pack(padx=10, pady=(20, 0))
        self.w_state = tk.DoubleVar()
        self.w = ttk.Combobox(self.frame, textvariable=self.w_state, width=30)
        self.w["values"] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        self.w.pack(padx=10, pady=(0, 20))
        self.w.current(1)

        self.dt_label = tk.Label(self.frame, text="Geschwindigkeit:", font=("Arial", 16))
        self.dt_label.pack(padx=10, pady=(20, 0))
        self.dt_state = tk.IntVar()
        self.dt = tk.Scale(self.frame, variable=self.dt_state, from_=1, to=20, orient="horizontal")
        self.dt.pack(padx=10, pady=(0, 5))
        self.dt.set(1)

        self.start_button = tk.Button(self.frame, text="Starten", font=("Arial", 18), command=self.barriere)
        self.start_button.pack(padx=10, pady=20)

        self.slow = tk.Label(self.frame, text="Es dauert bis zu 20 Sekunden, bis die Simulation startet.",
                             font=("Arial", 12))
        self.slow.pack(pady=10)
        self.frame.pack()

    def back_to_home(self):
        self.frame.destroy()
        HomePage(self.master)

    def barriere(self):
        def hamiltonian(N, dx, V=None):
            # Returns Hamiltonian as a sparse matrix using finite differences.
            L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))  # Laplace operator
            H = - (hbar ** 2) * L / (2 * m * dx ** 2)  # Kinetic energy operator
            if V is not None:  # Potential energy by default 0
                H += scipy.sparse.spdiags(V, 0, N, N)  # Potential energy in shape of N,N
            return H.tocsc()

        def time_evolution_operator(H, dt):
            # Time evolution operator given a Hamiltonian and time step.
            U = scipy.linalg.expm(-1j * H * dt / hbar).toarray()
            U[(U.real ** 2 + U.imag ** 2) < 1E-10] = 0
            return scipy.sparse.csc_matrix(U)

        def simulate(psi, H, dt):
            # Generates wave function and time at the next time step.
            U = time_evolution_operator(H, dt)
            t = 0
            while t < max_time:
                probability_densities.append(psi.real ** 2 + psi.imag ** 2)
                real.append(np.real(psi))
                imaginary.append(np.imag(psi))
                times.append(t)
                psi = U @ psi
                t += 1

        # ANIMATION


        def animation():
            fig, ax = plt.subplots()
            probability_line, = ax.plot(x, probability_densities[0], label="Psi^2")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, np.max(probability_densities))
            if self.real_state.get() == 1:
                real_line, = ax.plot(x, real[0], label='Re(Psi)')
                ax.set_ylim(np.min(real), np.max(real))
            if self.ima_state.get() == 1:
                imaginary_line, = ax.plot(x, imaginary[0], label='Im(Psi)')
                ax.set_ylim(np.min(imaginary), np.max(imaginary))

            ax.fill_between([0, self.w_state.get()], 0, np.max(probability_densities), color='gray', alpha=0.5)

            def update(frame):
                probability_line.set_ydata(probability_densities[frame])
                if self.real_state.get() == 1:
                    real_line.set_ydata(real[frame])
                if self.ima_state.get() == 1:
                    imaginary_line.set_ydata(imaginary[frame])
                ax.set_title(f"Time: {times[frame]}")

            ani = FuncAnimation(fig, update, frames=len(probability_densities), interval=100)
            ax.legend()
            plt.xlabel('Position x')
            plt.ylabel('Wellenfunktion')
            plt.show()

        # POTENTIAL BARRIER
        def rectangular_potential_barrier(x, V0, w):
            # Rectangular potential barrier of height V0 and width w.
            return np.where((0 <= x) & (x < w), V0, 0.0)

        # INITIAL WAVEFUNCTION
        def gaussian_wavepacket(x, x0, sigma0, e):
            # Gaussian wavepacket at x0 +/- width sigma0, with energy E.
            p0 = np.sqrt(2 * e)
            A = (2 * np.pi * sigma0 ** 2) ** (-0.25)
            return A * np.exp(1j * p0 * x - ((x - x0) / (2 * sigma0)) ** 2)

        # SETTINGS
        grid = 500  # number of points
        probability_densities = []  # l of arrays
        times = []  # l of zeiten
        real = []
        imaginary = []
        max_time = 20
        m = 1  # m
        hbar = 1
        a = 128  # x position length

        x, dx = np.linspace(-a / 2, a / 2, grid, endpoint=False, retstep=True)
        e = self.e_state.get()  # Energie des Wellenpakets (1 bis 25)
        V0 = self.v_state.get()  # Potentialenergie Barriere (konstant = 5)
        psi0 = gaussian_wavepacket(x, x0=-30.0, sigma0=3.0, e=e)
        V = rectangular_potential_barrier(x=x, V0=V0, w=self.w_state.get())  # w = Breite der Barriere

        # MAIN
        H = hamiltonian(grid, dx, V=V)
        simulate(psi0, H, dt=self.dt_state.get())
        animation()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("700x800")
    app = HomePage(root)
    root.mainloop()

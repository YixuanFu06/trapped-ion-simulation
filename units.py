'''
Re-parametrization of the system
'''

import numpy as np
import scipy.constants as const

MASS = 171 * const.atomic_mass  # mass of 171Yb+ in kg, about 2.84e-25 kg
CHARGE = const.e # elementary charge in C, about 1.602e-19 C

FREQ_ARC = 2 * np.pi * 1e6  # typical angular frequency of the trap
TIME = 1.0 / FREQ_ARC  # time unit

# length unit satisfying $m\omega_0^2 l_0=\frac{e^2}{4\pi\epsilon_0 l_0^2}$
LENGTH = (CHARGE**2 / (4 * np.pi * const.epsilon_0 * MASS * FREQ_ARC**2))**(1/3) # length unit, about 2.74 microns

TEMPERATURE = MASS * FREQ_ARC**2 * LENGTH**2 / const.k  # temperature unit, about 6.10K

# express \hbar in our units
HBAR = const.hbar / (MASS * LENGTH**2 / TIME)  # about 7.872e-6 in our units

# \Gamma is the decay rate of the excited state of 171Yb+, about 2\pi*19.6 MHz in SI units
GAMMA = 2 * np.pi * 19.6e6 * TIME  # about 19.6 in our units

S = 1.0  # saturation parameter for Doppler cooling, s=2\Omega^2/\Gamma^2

# WAVENUMBER is the wave number of the cooling laser, K=2\pi/\lambda (369)
WAVENUMBER = 2 * np.pi / (369.52e-9 / LENGTH)  # about 46.6 in our units

# \gamma is the damping rate of the laser cooling satisfying \gamma = \frac{s}{(1+s)^{3/2}}\hbar k^2
GAMMA_LASER = S / (1+S)**(3/2) * HBAR * WAVENUMBER**2  # about 6.044e-3 in our units

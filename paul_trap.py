'''
PaulTrap Physics Core
'''
import units
import numpy as np

class PaulTrap:
    def __init__(self, num_ions: int, frequencies: tuple, gamma_laser: float, gamma_thermal: float, temperature: float):
        """
        Initialize the Paul Trap simulation.

        Args:
            num_ions (int): Number of ions.
            frequencies (tuple/list): Trap frequencies (fx, fy, fz).
            gamma_laser (float): Laser cooling damping rate.
            gamma_thermal (float): Thermal bath coupling rate.
            temperature (float): Target temperature of the thermal bath.
        """
        self.num_ions = num_ions
        # Store initial params for reset
        self.current_time = 0.0
        self.frequencies = np.array(frequencies, dtype=float)

        # Physics Parameters
        self.init_gamma_laser = gamma_laser
        self.init_gamma_thermal = gamma_thermal
        self.gamma_laser = gamma_laser
        self.gamma_thermal = gamma_thermal
        self.temperature = temperature
        self.real_temperature = temperature

        # Switches
        self.is_laser_on = True
        self.is_thermal_on = True
        self.is_stochastic_on = True

        # Precompute effective gamma and noise
        self.gamma_eff = 0.0
        self.recompute_gamma_eff()
        self.recompute_noise_std()

        self.positions = np.zeros((self.num_ions, 3))
        self.velocities = np.zeros((self.num_ions, 3))

        # Cache for deterministic forces
        self.stored_forces = None

        self.reset()

    def reset(self):
        # Reset current time
        self.current_time = 0.0
        self.stored_forces = None

        if self.num_ions == 0:
            self.positions = np.empty((0, 3))
            self.velocities = np.empty((0, 3))
            return

        # Initialize positions based on Boltzmann distribution in harmonic trap
        # P(x) ~ exp(-omega^2 * x^2 / (2T)) => sigma = sqrt(T) / omega
        # Avoid division by zero if frequencies form singularities, though unlikely here
        if self.temperature > 0:
            sigma_pos = np.sqrt(self.temperature) / self.frequencies
            # Use broadcasting: (N, 3) * (3,)
            self.positions = np.random.randn(self.num_ions, 3) * sigma_pos
        else:
            # Fallback to small random displacements if T=0
            self.positions = 2.0 * np.random.rand(self.num_ions, 3) - 1.0

        # Initialize velocities with Maxwell-Boltzmann distribution (Gaussian)
        # P(v) ~ exp(-v^2 / (2T))  => sigma = sqrt(T)
        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            self.velocities = np.random.normal(loc=0.0, scale=sigma, size=(self.num_ions, 3))
        else:
            self.velocities = np.zeros((self.num_ions, 3))

    def add_ion(self, n=1):
        self.num_ions += n

        # Invalidate cache
        self.stored_forces = None

        # Initialize new position similar to reset but for one particle
        if self.temperature > 0:
            sigma_pos = np.sqrt(self.temperature) / self.frequencies
            new_pos = np.random.randn(n, 3) * sigma_pos
        else:
            new_pos = 2.0 * np.random.rand(n, 3) - 1.0

        self.positions = np.vstack((self.positions, new_pos))

        # Initialize new velocity
        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            new_vel = np.random.normal(loc=0.0, scale=sigma, size=(n, 3))
        else:
            new_vel = np.zeros((n, 3))

        self.velocities = np.vstack((self.velocities, new_vel))

    def remove_all_ions(self):
        self.num_ions = 0
        self.positions = np.empty((0, 3))
        self.velocities = np.empty((0, 3))
        self.current_time = 0.0
        self.real_temperature = 0.0
        self.velocities = np.empty((0, 3))
        self.stored_forces = None

    def catch_ions(self, n):
        self.num_ions = n
        self.reset()

    def recompute_gamma_eff(self):
        self.gamma_eff = 0.0
        if self.is_laser_on: self.gamma_eff += self.gamma_laser
        if self.is_thermal_on: self.gamma_eff += self.gamma_thermal

    def recompute_noise_std(self):
        if not self.is_stochastic_on:
            self.random_force_std = 0.0
            return

        # Calculate random force standard deviation
        # Thermal part = 2 * gamma_thermal * T
        term_thermal = 0.0
        if self.is_thermal_on:
            # This is squared!
            term_thermal = 2 * self.gamma_thermal * self.temperature

        # Recoil part = sqrt(1+S) * gamma_laser * hbar * Gamma
        term_recoil = 0.0
        if self.is_laser_on:
            # This is squared!
            term_recoil = units.S / (1+units.S) * units.GAMMA * (units.HBAR * units.WAVENUMBER)**2

        # force = \sqrt{term_thermal + term_recoil}
        self.random_force_std = np.sqrt(term_thermal + term_recoil)

    def set_switches(self, laser=None, thermal=None, stochastic=None):
        if laser is not None: self.is_laser_on = laser
        if thermal is not None: self.is_thermal_on = thermal
        if stochastic is not None: self.is_stochastic_on = stochastic
        self.recompute_gamma_eff()
        self.recompute_noise_std()

        # Invalidate cache because noise params changed (though strictly forces don't dep on noise, good practice)
        self.stored_forces = None

    def set_params(self, freq_x=None, freq_y=None, freq_z=None, gamma_laser=None, gamma_thermal=None, temperature=None):
        if freq_x is not None: self.frequencies[0] = freq_x
        if freq_y is not None: self.frequencies[1] = freq_y
        if freq_z is not None: self.frequencies[2] = freq_z
        if gamma_laser is not None: self.gamma_laser = gamma_laser
        if gamma_thermal is not None: self.gamma_thermal = gamma_thermal
        if temperature is not None: self.temperature = temperature

        self.recompute_gamma_eff()
        self.recompute_noise_std()

        # Invalidate cached forces because frequencies (and thus trap forces) may have changed
        self.stored_forces = None

    def get_total_gamma(self):
        g = 0.0
        if self.is_laser_on: g += self.gamma_laser
        if self.is_thermal_on: g += self.gamma_thermal
        return g

    def compute_forces(self):
        if self.num_ions == 0:
            forces = np.zeros((0, 3))
            return forces

        forces = np.zeros_like(self.positions)

        # Vectorized trap forces
        forces -= self.frequencies**2 * self.positions

        # Add Coulomb forces between ions (Vectorized)
        delta_r = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        distances = np.linalg.norm(delta_r, axis=-1)

        with np.errstate(divide='ignore'):
            inv_dist3 = 1.0 / (distances**3)
        np.fill_diagonal(inv_dist3, 0.0)

        coulomb_forces = np.sum(delta_r * inv_dist3[:, :, np.newaxis], axis=1)
        forces += coulomb_forces
        return forces

    def update(self, dt):
        if self.num_ions == 0:
            return self.positions

        # Calculate effective gamma for damping
        # gamma_eff = 0.0
        # if self.is_laser_on: gamma_eff += self.gamma_laser
        # if self.is_thermal_on: gamma_eff += self.gamma_thermal
        # Use precomputed value
        gamma_eff = self.gamma_eff

        # BBK algorithm
        # Calculate random forces once per step
        random_forces = self.random_force_std * np.random.randn(*self.positions.shape) / np.sqrt(dt)

        # Optimize: reuse cached forces if available
        # Need to recompute if stored_forces is None or if shape mismatch (though reset handles shape)
        if self.stored_forces is None or self.stored_forces.shape != self.positions.shape:
            self.stored_forces = self.compute_forces()

        forces_t = self.stored_forces + random_forces
        v_half = self.velocities * (1 - 0.5 * gamma_eff * dt) + 0.5 * forces_t * dt
        self.positions += v_half * dt

        # Compute new deterministic forces at new position and cache for next step
        # This reduces compute_forces calls from 2 to 1 per step
        new_deterministic_forces = self.compute_forces()
        forces_new = new_deterministic_forces + random_forces
        self.stored_forces = new_deterministic_forces

        self.velocities = (v_half + 0.5 * forces_new * dt) / (1 + 0.5 * gamma_eff * dt)

        # Calculate real temperature from kinetic energy: T = 2/3 * <E_kin>
        # T = (2 * Total KE) / (3 * N) = sum(|v_i|^2) / (3 * N)
        self.real_temperature = np.sum(self.velocities**2) / (3 * self.num_ions)

        self.current_time += dt

        return self.positions

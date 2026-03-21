'''
PaulTrap Physics Core (PyTorch-Accelerated)
'''
import units
import numpy as np
import torch

class PaulTrap:
    def __init__(self, num_ions, frequencies, gamma_laser, gamma_thermal, temperature):
        self.num_ions = num_ions
        # Store initial params for reset
        self.current_time = 0.0

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Physics Parameters
        self.init_gamma_laser = gamma_laser
        self.init_gamma_thermal = gamma_thermal
        self.gamma_laser = gamma_laser
        self.gamma_thermal = gamma_thermal
        self.temperature = temperature

        # Switches
        self.is_laser_on = True
        self.is_thermal_on = True
        self.is_stochastic_on = True

        # Internal State Tensors (prefixed with _)
        # Initialize with zeros, will be set in reset()
        self._positions = torch.zeros((self.num_ions, 3), dtype=torch.float64, device=self.device)
        self._velocities = torch.zeros((self.num_ions, 3), dtype=torch.float64, device=self.device)
        self._frequencies = torch.tensor(frequencies, dtype=torch.float64, device=self.device)

        # Real temperature tracking (scalar tensor to avoid sync in loop)
        self._real_temperature = torch.tensor(temperature, dtype=torch.float64, device=self.device)

        # Precompute effective gamma and noise
        self.gamma_eff = 0.0
        self.recompute_gamma_eff()
        self.recompute_noise_std() # This updates self.random_force_std (float)

        # Cache for deterministic forces
        self.stored_forces = None

        self.reset()

    # --- Properties to maintain Interface Consistency ---

    @property
    def positions(self):
        """Return positions as discrete numpy array (CPU). Copy."""
        return self._positions.cpu().numpy()

    @positions.setter
    def positions(self, value):
        """Set positions from array-like."""
        self._positions = torch.tensor(value, dtype=torch.float64, device=self.device)

    @property
    def velocities(self):
        """Return velocities as discrete numpy array (CPU). Copy."""
        return self._velocities.cpu().numpy()

    @velocities.setter
    def velocities(self, value):
        """Set velocities from array-like."""
        self._velocities = torch.tensor(value, dtype=torch.float64, device=self.device)

    @property
    def frequencies(self):
        """Return frequencies as discrete numpy array (CPU). Copy."""
        return self._frequencies.cpu().numpy()

    @property
    def real_temperature(self):
        """Return scalar real temperature (CPU). Syncs with GPU."""
        return self._real_temperature.item()

    @real_temperature.setter
    def real_temperature(self, value):
        """Set real temperature manually."""
        self._real_temperature = torch.tensor(value, dtype=torch.float64, device=self.device)

    # --- Methods ---

    def reset(self):
        # Reset current time
        self.current_time = 0.0
        self.stored_forces = None

        if self.num_ions == 0:
            self._positions = torch.empty((0, 3), device=self.device)
            self._velocities = torch.empty((0, 3), device=self.device)
            return

        # Initialize positions based on Boltzmann distribution in harmonic trap
        # P(x) ~ exp(-omega^2 * x^2 / (2T)) => sigma = sqrt(T) / omega
        if self.temperature > 0:
            # Avoid sync: keep freq on device
            sigma_pos = np.sqrt(self.temperature) / self._frequencies.cpu().numpy()
            sigma_tensor = torch.tensor(sigma_pos, device=self.device, dtype=torch.float64)
            # Use broadcasting: (N, 3) * (3,)
            self._positions = torch.randn((self.num_ions, 3), device=self.device, dtype=torch.float64) * sigma_tensor
        else:
            # Fallback to small random displacements if T=0
            self._positions = 2.0 * torch.rand((self.num_ions, 3), device=self.device, dtype=torch.float64) - 1.0

        # Initialize velocities with Maxwell-Boltzmann distribution (Gaussian)
        # P(v) ~ exp(-v^2 / (2T))  => sigma = sqrt(T)
        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            self._velocities = torch.normal(mean=0.0, std=float(sigma), size=(self.num_ions, 3), device=self.device, dtype=torch.float64)
        else:
            self._velocities = torch.zeros((self.num_ions, 3), device=self.device, dtype=torch.float64)

        # Reset real temperature tracking (scalar tensor)
        self._real_temperature = torch.tensor(self.temperature, dtype=torch.float64, device=self.device)

    def add_ion(self, n=1):
        self.num_ions += n

        # Invalidate cache
        self.stored_forces = None

        # Initialize new position similar to reset but for one particle
        if self.temperature > 0:
            sigma_pos = np.sqrt(self.temperature) / self._frequencies.cpu().numpy()
            sigma_tensor = torch.tensor(sigma_pos, device=self.device, dtype=torch.float64)
            new_pos = torch.randn((n, 3), device=self.device, dtype=torch.float64) * sigma_tensor
        else:
            new_pos = 2.0 * torch.rand((n, 3), device=self.device, dtype=torch.float64) - 1.0

        self._positions = torch.vstack((self._positions, new_pos))

        # Initialize new velocity
        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            new_vel = torch.normal(mean=0.0, std=float(sigma), size=(n, 3), device=self.device, dtype=torch.float64)
        else:
            new_vel = torch.zeros((n, 3), device=self.device, dtype=torch.float64)

        self._velocities = torch.vstack((self._velocities, new_vel))

    def remove_all_ions(self):
        self.num_ions = 0
        self._positions = torch.empty((0, 3), device=self.device, dtype=torch.float64)
        self._velocities = torch.empty((0, 3), device=self.device, dtype=torch.float64)
        self.current_time = 0.0
        self._real_temperature = torch.tensor(0.0, device=self.device, dtype=torch.float64)
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
            term_thermal = 2 * self.gamma_thermal * self.temperature

        # Recoil part = S / (1+S) * gamma * (hbar * k)^2
        term_recoil = 0.0
        if self.is_laser_on:
            term_recoil = units.S / (1+units.S) * units.GAMMA * (units.HBAR * units.WAVENUMBER)**2

        self.random_force_std = np.sqrt(term_thermal + term_recoil)

    def set_switches(self, laser=None, thermal=None, stochastic=None):
        if laser is not None: self.is_laser_on = laser
        if thermal is not None: self.is_thermal_on = thermal
        if stochastic is not None: self.is_stochastic_on = stochastic
        self.recompute_gamma_eff()
        self.recompute_noise_std()

        # Invalidate cache
        self.stored_forces = None

    def set_params(self, freq_x=None, freq_y=None, freq_z=None, gamma_laser=None, gamma_thermal=None, temperature=None):
        if freq_x is not None: self._frequencies[0] = float(freq_x)
        if freq_y is not None: self._frequencies[1] = float(freq_y)
        if freq_z is not None: self._frequencies[2] = float(freq_z)
        
        if gamma_laser is not None: self.gamma_laser = float(gamma_laser)
        if gamma_thermal is not None: self.gamma_thermal = float(gamma_thermal)
        if temperature is not None: self.temperature = float(temperature)

        self.recompute_gamma_eff()
        self.recompute_noise_std()

        self.stored_forces = None

    def get_total_gamma(self):
        g = 0.0
        if self.is_laser_on: g += self.gamma_laser
        if self.is_thermal_on: g += self.gamma_thermal
        return g

    def compute_forces(self):
        if self.num_ions == 0:
            forces = torch.zeros((0, 3), device=self.device, dtype=torch.float64)
            return forces

        forces = torch.zeros_like(self._positions, dtype=torch.float64)

        # Vectorized trap forces
        # frequencies is (3,), positions is (N, 3)
        forces -= self._frequencies**2 * self._positions

        # Add Coulomb forces between ions (Vectorized)
        # delta_r: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        delta_r = self._positions.unsqueeze(1) - self._positions.unsqueeze(0)

        # distances: (N, N)
        distances = torch.norm(delta_r, dim=-1)

        # Handle division by zero (diagonal)
        dist_pow3 = distances.pow(3)
        # Avoid zero division by setting diagonal to INF or 1.0 (masked later)
        # We fill diagonal of dist_pow3 with 1.0 safely
        dist_pow3.fill_diagonal_(1.0)

        inv_dist3 = 1.0 / dist_pow3
        inv_dist3.fill_diagonal_(0.0) # Explicitly zero out self-interaction

        # coulomb_forces: sum over j (axis 1) of delta_r_ij * inv_dist3_ij
        # delta_r: (N, N, 3)
        # inv_dist3: (N, N) -> (N, N, 1)
        coulomb_forces = torch.sum(delta_r * inv_dist3.unsqueeze(-1), dim=1)
        forces += coulomb_forces
        return forces

    def update(self, dt):
        """
        Updates the simulation.
        Returns the internal positions tensor to avoid CPU sync in tight loops.
        Note: The return value is technically a Tensor, not a Numpy array, 
        but known consumers (gui.py, tui.py) ignore the return value anyway.
        Access .positions property to get a Numpy array.
        """
        if self.num_ions == 0:
            return self._positions

        gamma_eff = self.gamma_eff

        # BBK algorithm
        # Calculate random forces once per step
        random_forces = self.random_force_std * torch.randn(self._positions.shape, device=self.device, dtype=torch.float64) / np.sqrt(dt)

        # Optimize: reuse cached forces if available
        if self.stored_forces is None or self.stored_forces.shape != self._positions.shape:
            self.stored_forces = self.compute_forces()

        forces_t = self.stored_forces + random_forces
        v_half = self._velocities * (1 - 0.5 * gamma_eff * dt) + 0.5 * forces_t * dt
        self._positions += v_half * dt

        # Compute new deterministic forces at new position and cache
        new_deterministic_forces = self.compute_forces()
        forces_new = new_deterministic_forces + random_forces
        self.stored_forces = new_deterministic_forces

        self._velocities = (v_half + 0.5 * forces_new * dt) / (1 + 0.5 * gamma_eff * dt)

        # Update Real Temperature (on device)
        # T = sum(|v_i|^2) / (3 * N)
        self._real_temperature = torch.sum(self._velocities**2) / (3 * self.num_ions)

        self.current_time += dt

        return self._positions

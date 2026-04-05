'''
PaulTrap Physics Core (PyTorch-Accelerated, torch.compile optimized)
'''
import units
import numpy as np
import torch
import logging

# Suppress torch.compile warnings that break TUI output
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)

def _compute_forces_impl(positions, frequencies):
    if positions.shape[0] == 0:
        return torch.zeros_like(positions)

    forces = -(frequencies**2) * positions

    delta_r = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.norm(delta_r, dim=-1)
    dist_pow3 = distances.pow(3)

    # Avoid zero division and self-interaction safely without in-place mutation
    eye = torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device)
    dist_pow3 = torch.where(eye, 1.0, dist_pow3)

    inv_dist3 = 1.0 / dist_pow3
    inv_dist3 = torch.where(eye, 0.0, inv_dist3)

    coulomb_forces = torch.sum(delta_r * inv_dist3.unsqueeze(-1), dim=1)
    return forces + coulomb_forces

@torch.compile(mode="reduce-overhead")
def _update_n_steps_compiled(steps: int, dt: float, gamma_eff: float, random_force_std: float, positions: torch.Tensor, velocities: torch.Tensor, frequencies: torch.Tensor, stored_forces: torch.Tensor):
    dt_sqrt = dt**0.5
    for _ in range(steps):
        noise = (random_force_std / dt_sqrt) * torch.randn_like(positions)

        f_total = stored_forces + noise
        v_half = velocities * (1 - 0.5 * gamma_eff * dt) + 0.5 * f_total * dt
        positions = positions + v_half * dt

        f_det_new = _compute_forces_impl(positions, frequencies)
        
        f_total_new = f_det_new + noise
        stored_forces = f_det_new

        velocities = (v_half + 0.5 * f_total_new * dt) / (1 + 0.5 * gamma_eff * dt)

    num_ions = positions.shape[0]
    real_temp = torch.sum(velocities**2) / (3 * num_ions)
    return positions, velocities, stored_forces, real_temp


class PaulTrap:
    def __init__(self, num_ions, frequencies, gamma_laser, gamma_thermal, temperature, precision='float64'):
        self.num_ions = num_ions
        # Store initial params for reset
        self.current_time = 0.0

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64 if precision == 'float64' else torch.float32

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
        self._positions = torch.zeros((self.num_ions, 3), dtype=self.dtype, device=self.device)
        self._velocities = torch.zeros((self.num_ions, 3), dtype=self.dtype, device=self.device)
        self._frequencies = torch.tensor(frequencies, dtype=self.dtype, device=self.device)

        # Real temperature tracking (scalar tensor to avoid sync in loop)
        self._real_temperature = torch.tensor(temperature, dtype=self.dtype, device=self.device)

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
        self._positions = torch.tensor(value, dtype=self.dtype, device=self.device)

    @property
    def velocities(self):
        """Return velocities as discrete numpy array (CPU). Copy."""
        return self._velocities.cpu().numpy()

    @velocities.setter
    def velocities(self, value):
        """Set velocities from array-like."""
        self._velocities = torch.tensor(value, dtype=self.dtype, device=self.device)

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
        self._real_temperature = torch.tensor(value, dtype=self.dtype, device=self.device)

    # --- Methods ---

    def reset(self):
        # Reset current time
        self.current_time = 0.0
        self.stored_forces = None

        if self.num_ions == 0:
            self._positions = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            self._velocities = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            return

        # Initialize positions based on Boltzmann distribution in harmonic trap
        # P(x) ~ exp(-omega^2 * x^2 / (2T)) => sigma = sqrt(T) / omega
        if self.temperature > 0:
            # Avoid sync: keep freq on device
            sigma_pos = np.sqrt(self.temperature) / self._frequencies.cpu().numpy()
            sigma_tensor = torch.tensor(sigma_pos, device=self.device, dtype=self.dtype)
            # Use broadcasting: (N, 3) * (3,)
            self._positions = torch.randn((self.num_ions, 3), device=self.device, dtype=self.dtype) * sigma_tensor
        else:
            # Fallback to small random displacements if T=0
            self._positions = 2.0 * torch.rand((self.num_ions, 3), device=self.device, dtype=self.dtype) - 1.0

        # Initialize velocities with Maxwell-Boltzmann distribution (Gaussian)
        # P(v) ~ exp(-v^2 / (2T))  => sigma = sqrt(T)
        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            self._velocities = torch.normal(mean=0.0, std=float(sigma), size=(self.num_ions, 3), device=self.device, dtype=self.dtype)
        else:
            self._velocities = torch.zeros((self.num_ions, 3), device=self.device, dtype=self.dtype)

        # Reset real temperature tracking (scalar tensor)
        self._real_temperature = torch.tensor(self.temperature, dtype=self.dtype, device=self.device)

    def add_ion(self, n=1):
        self.num_ions += n

        # Invalidate cache
        self.stored_forces = None

        # Initialize new position similar to reset but for one particle
        if self.temperature > 0:
            sigma_pos = np.sqrt(self.temperature) / self._frequencies.cpu().numpy()
            sigma_tensor = torch.tensor(sigma_pos, device=self.device, dtype=self.dtype)
            new_pos = torch.randn((n, 3), device=self.device, dtype=self.dtype) * sigma_tensor
        else:
            new_pos = 2.0 * torch.rand((n, 3), device=self.device, dtype=self.dtype) - 1.0

        self._positions = torch.vstack((self._positions, new_pos))

        # Initialize new velocity
        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            new_vel = torch.normal(mean=0.0, std=float(sigma), size=(n, 3), device=self.device, dtype=self.dtype)
        else:
            new_vel = torch.zeros((n, 3), device=self.device, dtype=self.dtype)

        self._velocities = torch.vstack((self._velocities, new_vel))

    def remove_all_ions(self):
        self.num_ions = 0
        self._positions = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self._velocities = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.current_time = 0.0
        self._real_temperature = torch.tensor(0.0, device=self.device, dtype=self.dtype)
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
            return torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        return _compute_forces_impl(self._positions, self._frequencies)

    def update(self, dt):
        """
        Updates the simulation for 1 step.
        Returns the internal positions tensor to avoid CPU sync.
        """
        return self.update_n_steps(dt, 1)

    def update_n_steps(self, dt, steps):
        """
        Updates the simulation for N steps using torch.compile caching.
        """
        if self.num_ions == 0 or steps <= 0:
            return self._positions

        if self.stored_forces is None or self.stored_forces.shape != self._positions.shape:
            self.stored_forces = self.compute_forces()

        with torch._dynamo.config.patch(suppress_errors=True):
            self._positions, self._velocities, self.stored_forces, self._real_temperature = _update_n_steps_compiled(
                int(steps), dt, float(self.gamma_eff), float(self.random_force_std),
                self._positions, self._velocities, self._frequencies, self.stored_forces
            )

        self.current_time += dt * steps
        
        # Perform implicit wait. Wait implicitly for the queued ops.
        # This keeps the host aligned with performance trackers without blocking if possible
        if self.device.type == 'cuda':
            torch.cuda.synchronize(device=self.device)
            
        return self._positions


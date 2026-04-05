'''
PaulTrap Physics Core (JAX-Accelerated)
'''
import units
import numpy as np

import jax
from jax import config
import jax.numpy as jnp


@jax.jit
def _compute_forces_impl(positions, frequencies):
    if positions.shape[0] == 0:
        return jnp.zeros((0, 3), dtype=positions.dtype)

    forces = -(frequencies**2) * positions

    delta_r = positions[:, None, :] - positions[None, :, :]
    distances = jnp.linalg.norm(delta_r, axis=-1)
    dist_pow3 = distances**3

    # Remove self-interaction safely on the diagonal.
    eye = jnp.eye(positions.shape[0], dtype=bool)
    dist_pow3 = jnp.where(eye, 1.0, dist_pow3)
    inv_dist3 = 1.0 / dist_pow3
    inv_dist3 = jnp.where(eye, 0.0, inv_dist3)

    coulomb_forces = jnp.sum(delta_r * inv_dist3[:, :, None], axis=1)
    return forces + coulomb_forces

@jax.jit
def _update_n_steps_jit(steps, dt, gamma_eff, random_force_std, positions, velocities, frequencies, rng_key, stored_forces):
    def cond_fn(state):
        i, *rest = state
        return i < steps

    def body_fn(state):
        i, pos, vel, f_det, key = state
        key, subkey = jax.random.split(key)
        noise = random_force_std * jax.random.normal(subkey, shape=pos.shape, dtype=pos.dtype) / jnp.sqrt(dt)
        f_total = f_det + noise
        
        v_half = vel * (1 - 0.5 * gamma_eff * dt) + 0.5 * f_total * dt
        pos_new = pos + v_half * dt
        
        f_det_new = _compute_forces_impl(pos_new, frequencies)
        f_total_new = f_det_new + noise
        vel_new = (v_half + 0.5 * f_total_new * dt) / (1 + 0.5 * gamma_eff * dt)
        
        return (i + 1, pos_new, vel_new, f_det_new, key)

    init_state = (0, positions, velocities, stored_forces, rng_key)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    _, final_pos, final_vel, final_f_det, final_key = final_state
    return final_pos, final_vel, final_f_det, final_key


class PaulTrap:
    def __init__(self, num_ions, frequencies, gamma_laser, gamma_thermal, temperature, precision='float64'):
        # Keep numerical behavior aligned with the torch backend.
        if precision == 'float64':
            config.update('jax_enable_x64', True)
        self.dtype = jnp.float64 if precision == 'float64' else jnp.float32
        
        self.num_ions = num_ions
        self.current_time = 0.0

        # Keep a device field for UI compatibility with the torch backend.
        self.device = jax.devices()[0]

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

        # Internal State Arrays (prefixed with _)
        self._positions = jnp.zeros((self.num_ions, 3), dtype=self.dtype)
        self._velocities = jnp.zeros((self.num_ions, 3), dtype=self.dtype)
        self._frequencies = jnp.array(frequencies, dtype=self.dtype)
        self._real_temperature = jnp.array(temperature, dtype=self.dtype)

        # RNG state for JAX random generation.
        self._rng_key = jax.random.PRNGKey(0)

        # Precompute effective gamma and noise
        self.gamma_eff = 0.0
        self.recompute_gamma_eff()
        self.recompute_noise_std()

        # Cache for deterministic forces
        self.stored_forces = None

        self.reset()

    @property
    def positions(self):
        """Return positions as discrete numpy array (CPU). Copy."""
        return np.asarray(self._positions)

    @positions.setter
    def positions(self, value):
        """Set positions from array-like."""
        self._positions = jnp.array(value, dtype=self.dtype)

    @property
    def velocities(self):
        """Return velocities as discrete numpy array (CPU). Copy."""
        return np.asarray(self._velocities)

    @velocities.setter
    def velocities(self, value):
        """Set velocities from array-like."""
        self._velocities = jnp.array(value, dtype=self.dtype)

    @property
    def frequencies(self):
        """Return frequencies as discrete numpy array (CPU). Copy."""
        return np.asarray(self._frequencies)

    @property
    def real_temperature(self):
        """Return scalar real temperature."""
        return float(self._real_temperature)

    @real_temperature.setter
    def real_temperature(self, value):
        """Set real temperature manually."""
        self._real_temperature = jnp.array(value, dtype=self.dtype)

    def _split_key(self):
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def reset(self):
        self.current_time = 0.0
        self.stored_forces = None

        if self.num_ions == 0:
            self._positions = jnp.empty((0, 3), dtype=self.dtype)
            self._velocities = jnp.empty((0, 3), dtype=self.dtype)
            return

        if self.temperature > 0:
            sigma_pos = jnp.sqrt(self.temperature) / self._frequencies
            key_pos = self._split_key()
            self._positions = jax.random.normal(
                key_pos,
                shape=(self.num_ions, 3),
                dtype=self.dtype,
            ) * sigma_pos
        else:
            key_pos = self._split_key()
            self._positions = 2.0 * jax.random.uniform(
                key_pos,
                shape=(self.num_ions, 3),
                dtype=self.dtype,
            ) - 1.0

        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            key_vel = self._split_key()
            self._velocities = jax.random.normal(
                key_vel,
                shape=(self.num_ions, 3),
                dtype=self.dtype,
            ) * float(sigma)
        else:
            self._velocities = jnp.zeros((self.num_ions, 3), dtype=self.dtype)

        self._real_temperature = jnp.array(self.temperature, dtype=self.dtype)

    def add_ion(self, n=1):
        self.num_ions += n
        self.stored_forces = None

        if self.temperature > 0:
            sigma_pos = jnp.sqrt(self.temperature) / self._frequencies
            key_pos = self._split_key()
            new_pos = jax.random.normal(key_pos, shape=(n, 3), dtype=self.dtype) * sigma_pos
        else:
            key_pos = self._split_key()
            new_pos = 2.0 * jax.random.uniform(key_pos, shape=(n, 3), dtype=self.dtype) - 1.0

        self._positions = jnp.vstack((self._positions, new_pos))

        if self.temperature > 0:
            sigma = np.sqrt(self.temperature)
            key_vel = self._split_key()
            new_vel = jax.random.normal(key_vel, shape=(n, 3), dtype=self.dtype) * float(sigma)
        else:
            new_vel = jnp.zeros((n, 3), dtype=self.dtype)

        self._velocities = jnp.vstack((self._velocities, new_vel))

    def remove_all_ions(self):
        self.num_ions = 0
        self._positions = jnp.empty((0, 3), dtype=self.dtype)
        self._velocities = jnp.empty((0, 3), dtype=self.dtype)
        self.current_time = 0.0
        self._real_temperature = jnp.array(0.0, dtype=self.dtype)
        self.stored_forces = None

    def catch_ions(self, n):
        self.num_ions = n
        self.reset()

    def recompute_gamma_eff(self):
        self.gamma_eff = 0.0
        if self.is_laser_on:
            self.gamma_eff += self.gamma_laser
        if self.is_thermal_on:
            self.gamma_eff += self.gamma_thermal

    def recompute_noise_std(self):
        if not self.is_stochastic_on:
            self.random_force_std = 0.0
            return

        term_thermal = 0.0
        if self.is_thermal_on:
            term_thermal = 2 * self.gamma_thermal * self.temperature

        term_recoil = 0.0
        if self.is_laser_on:
            term_recoil = units.S / (1 + units.S) * units.GAMMA * (units.HBAR * units.WAVENUMBER) ** 2

        self.random_force_std = np.sqrt(term_thermal + term_recoil)

    def set_switches(self, laser=None, thermal=None, stochastic=None):
        if laser is not None:
            self.is_laser_on = laser
        if thermal is not None:
            self.is_thermal_on = thermal
        if stochastic is not None:
            self.is_stochastic_on = stochastic

        self.recompute_gamma_eff()
        self.recompute_noise_std()
        self.stored_forces = None

    def set_params(self, freq_x=None, freq_y=None, freq_z=None, gamma_laser=None, gamma_thermal=None, temperature=None):
        freqs = self._frequencies
        if freq_x is not None:
            freqs = freqs.at[0].set(float(freq_x))
        if freq_y is not None:
            freqs = freqs.at[1].set(float(freq_y))
        if freq_z is not None:
            freqs = freqs.at[2].set(float(freq_z))
        self._frequencies = freqs

        if gamma_laser is not None:
            self.gamma_laser = float(gamma_laser)
        if gamma_thermal is not None:
            self.gamma_thermal = float(gamma_thermal)
        if temperature is not None:
            self.temperature = float(temperature)

        self.recompute_gamma_eff()
        self.recompute_noise_std()
        self.stored_forces = None

    def get_total_gamma(self):
        g = 0.0
        if self.is_laser_on:
            g += self.gamma_laser
        if self.is_thermal_on:
            g += self.gamma_thermal
        return g

    def compute_forces(self):
        return _compute_forces_impl(self._positions, self._frequencies)

    def update(self, dt):
        """
        Updates the simulation.
        Returns the internal positions array to keep interface parity with torch backend.
        """
        if self.num_ions == 0:
            return self._positions

        gamma_eff = self.gamma_eff

        key_noise = self._split_key()
        random_forces = self.random_force_std * jax.random.normal(
            key_noise,
            shape=self._positions.shape,
            dtype=self.dtype,
        ) / np.sqrt(dt)

        if self.stored_forces is None or self.stored_forces.shape != self._positions.shape:
            self.stored_forces = self.compute_forces()

        forces_t = self.stored_forces + random_forces
        v_half = self._velocities * (1 - 0.5 * gamma_eff * dt) + 0.5 * forces_t * dt
        self._positions = self._positions + v_half * dt

        new_deterministic_forces = self.compute_forces()
        forces_new = new_deterministic_forces + random_forces
        self.stored_forces = new_deterministic_forces

        self._velocities = (v_half + 0.5 * forces_new * dt) / (1 + 0.5 * gamma_eff * dt)

        self._real_temperature = jnp.sum(self._velocities**2) / (3 * self.num_ions)
        self.current_time += dt

        return self._positions

    def update_n_steps(self, dt, steps):
        """
        Updates the simulation for N steps without passing data back to host.
        """
        if self.num_ions == 0 or steps <= 0:
            return self._positions

        if self.stored_forces is None or self.stored_forces.shape != self._positions.shape:
            self.stored_forces = self.compute_forces()

        final_pos, final_vel, final_f_det, final_key = _update_n_steps_jit(
            jnp.int32(steps),
            dt,
            self.gamma_eff,
            self.random_force_std,
            self._positions,
            self._velocities,
            self._frequencies,
            self._rng_key,
            self.stored_forces
        )
        
        self._positions = final_pos
        self._velocities = final_vel
        self.stored_forces = final_f_det
        self._rng_key = final_key
        
        self._real_temperature = jnp.sum(self._velocities**2) / (3 * self.num_ions)
        self.current_time += dt * steps
        
        # 强制等待异步计算完成，确保外部性能分析工具(如TUI)能测到真实的计算时间
        self._positions.block_until_ready()
        
        return self._positions


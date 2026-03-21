import curses
import time
import numpy as np
import units

try:
    from paul_trap_torch import PaulTrap
    print("Using PyTorch backend.")
except ImportError:
    print("PyTorch backend unavailable. Using NumPy backend.")
    from paul_trap import PaulTrap

def main(stdscr):
    # Setup curses
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0) # Non-blocking getch
    stdscr.keypad(True) # Enable special keys (arrows)

    # --- Simulation Parameters (Hardcoded config) ---
    num_ions = 10
    dt = 0.01
    init_freq = 8.0
    init_gamma_laser = 2.0
    init_gamma_thermal = 0.1
    init_temp = 0.0020

    # Initialize Trap
    trap = PaulTrap(num_ions=num_ions, frequencies=(init_freq, init_freq, 1.0),
                    gamma_laser=init_gamma_laser, gamma_thermal=init_gamma_thermal,
                    temperature=init_temp)

    # State variables for switches (matching gui.py defaults: Laser=True, Real=False, Vacuum=True, Stochastic=True)
    state = {
        'laser': True,
        'real_laser': False,
        'vacuum': True,
        'stochastic': True
    }

    # Helper to apply logic
    def apply_state():
        # Determine Gamma Laser Value
        current_gamma_l = init_gamma_laser
        if state['real_laser']:
            current_gamma_l = units.GAMMA_LASER

        # Apply parameters
        trap.set_params(gamma_laser=current_gamma_l)

        # Apply switches
        trap.set_switches(laser=state['laser'], thermal=(not state['vacuum']), stochastic=state['stochastic'])

    # Initial Application
    apply_state()

    # Performance tracking
    last_render_time = time.time()
    steps_accumulated = 0
    # Target frame rate for UI updates (e.g., 15 FPS to reduce flickering and CPU for rendering)
    target_fps = 15.0
    min_render_interval = 1.0 / target_fps

    # "Batch size" for simulation between input checks.
    # This was "steps_per_frame", now it means "steps per batch".
    steps_batch_size = 10

    # Auto-tune variables
    TARGET_BATCH_DURATION = 0.005 # Target 5ms per batch check for responsive input
    batch_tuner_history = []

    # --- Lindemann Index Variables ---
    LINDEMANN_N_EQ = 20000    # Equilibration steps
    LINDEMANN_N_PROD = 50000  # Production steps
    LINDEMANN_N_SAMPLE = 100  # Sample every N steps

    lindemann_state = {
        'status': 'IDLE', # IDLE, EQ, PROD, DONE
        'step_counter': 0,
        'samples': 0,
        'sum_rij': None,
        'sum_rij_sq': None,
        'result': 0.0
    }

    # Initial Calculation Variables
    sim_speed = 0.0
    sim_speed_us = 0.0
    fps = 0.0

    while True:
        try:
            batch_start_time = time.time()

            # --- Input Handling ---
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                break

            update_needed = False

            # FPS Control
            if key == curses.KEY_UP:
                target_fps += 5.0
                if target_fps > 200.0: target_fps = 200.0
                min_render_interval = 1.0 / target_fps
            elif key == curses.KEY_DOWN:
                target_fps -= 5.0
                if target_fps < 5.0: target_fps = 5.0
                min_render_interval = 1.0 / target_fps

            if key == ord('l') or key == ord('L'): # Toggle Laser
                state['laser'] = not state['laser']
                # Constraint: if Laser becomes OFF and Real Laser was ON, turn off Real Laser
                if not state['laser'] and state['real_laser']:
                    state['real_laser'] = False
                update_needed = True

            elif key == ord('k') or key == ord('K'): # Toggle Real Laser (K instead of R to avoid conflict with Reset)
                state['real_laser'] = not state['real_laser']
                # Constraint: if Real Laser becomes ON and Laser was OFF, turn on Laser
                if state['real_laser'] and not state['laser']:
                    state['laser'] = True
                update_needed = True

            elif key == ord('v') or key == ord('V'): # Toggle Vacuum
                state['vacuum'] = not state['vacuum']
                update_needed = True

            elif key == ord('s') or key == ord('S'): # Toggle Stochastic
                state['stochastic'] = not state['stochastic']
                update_needed = True

            elif key == ord('r') or key == ord('R'): # Reset
                trap.reset()
                lindemann_state['status'] = 'IDLE'

            elif key == ord('i') or key == ord('I'): # Start Lindemann Index Calculation
                if lindemann_state['status'] == 'IDLE' or lindemann_state['status'] == 'DONE':
                    lindemann_state['status'] = 'EQ'
                    lindemann_state['step_counter'] = 0
                    lindemann_state['samples'] = 0
                    lindemann_state['result'] = 0.0
                    # Initialize accumulators
                    n_ions = trap.num_ions
                    lindemann_state['sum_rij'] = np.zeros((n_ions, n_ions))
                    lindemann_state['sum_rij_sq'] = np.zeros((n_ions, n_ions))
            
            # Manual tuning removed in favor of auto-tuning, but keep keys for bias adjustment if needed?
            # Let's replace UP/DOWN with bias adjustment for "aggressiveness"
            # Or just let it be fully automatic. Let's keep manual override as bias.

            if update_needed:
                apply_state()

            # --- Physics Update (Run a batch) ---
            # Measure time taken for physics only
            p_start = time.time()

            # Determine how to run based on Lindemann status
            if lindemann_state['status'] == 'IDLE' or lindemann_state['status'] == 'DONE':
                # Normal fast loop
                for _ in range(steps_batch_size):
                    trap.update(dt)
            else:
                # Lindemann Calculation Active - we need to count steps carefully
                steps_remaining = steps_batch_size
                while steps_remaining > 0:
                    trap.update(dt)
                    lindemann_state['step_counter'] += 1
                    steps_remaining -= 1

                    if lindemann_state['status'] == 'EQ':
                        if lindemann_state['step_counter'] >= LINDEMANN_N_EQ:
                            lindemann_state['status'] = 'PROD'
                            lindemann_state['step_counter'] = 0 # Reset for production phase

                    elif lindemann_state['status'] == 'PROD':
                        # Sample logic
                        if lindemann_state['step_counter'] % LINDEMANN_N_SAMPLE == 0:
                            # Calculate all pairs distances
                            pos = trap.positions # (N, 3)
                            # Efficient distance matrix calculation:
                            # dist_matrix[i, j] = norm(pos[i] - pos[j])
                            # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
                            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] 
                            rij = np.linalg.norm(diff, axis=2)

                            lindemann_state['sum_rij'] += rij
                            lindemann_state['sum_rij_sq'] += rij**2
                            lindemann_state['samples'] += 1

                        if lindemann_state['step_counter'] >= LINDEMANN_N_PROD:
                            # Calculate Final Result
                            lindemann_state['status'] = 'DONE'

                            N = trap.num_ions
                            if N > 1 and lindemann_state['samples'] > 0:
                                avg_rij = lindemann_state['sum_rij'] / lindemann_state['samples']
                                avg_rij_sq = lindemann_state['sum_rij_sq'] / lindemann_state['samples']

                                # Variance term: <r^2> - <r>^2
                                var_rij = avg_rij_sq - avg_rij**2
                                # Avoid negative due to float precision
                                var_rij = np.maximum(var_rij, 0)
                                std_rij = np.sqrt(var_rij)

                                # Term for each pair: sqrt(...) / <rij>
                                # Avoid division by zero (diagonal is 0)
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    term = std_rij / avg_rij
                                    term[np.isnan(term)] = 0.0 # Diagonal nan
                                    term[np.isinf(term)] = 0.0 # Should not happen for distinct ions

                                # Sum i < j
                                # np.triu gives upper triangle, k=1 excludes diagonal
                                sum_terms = np.sum(np.triu(term, k=1))

                                delta_L = (2.0 / (N * (N - 1))) * sum_terms
                                lindemann_state['result'] = delta_L
                            else:
                                lindemann_state['result'] = 0.0

            p_end = time.time()

            p_duration = p_end - p_start
            steps_accumulated += steps_batch_size

            # --- Auto-Tuning Logic ---
            # We want p_duration to be around TARGET_BATCH_DURATION (e.g. 5ms)
            # This ensures we check for input ~200 times a second.
            if p_duration > 0:
                # Calculate steps per second capacity
                current_sps = steps_batch_size / p_duration
                # Suggest new batch size
                suggested_batch = int(current_sps * TARGET_BATCH_DURATION)

                # Apply with smoothing (Exponential Moving Average)
                # steps_batch_size = 0.9 * steps_batch_size + 0.1 * suggested_batch
                # Using integer logic for stability
                if suggested_batch > steps_batch_size:
                    steps_batch_size += 1
                elif suggested_batch < steps_batch_size and steps_batch_size > 1:
                    steps_batch_size -= 1

                # Hard clamps
                steps_batch_size = max(1, min(steps_batch_size, 5000))

            # --- Rendering Check ---
            current_time = time.time()
            dt_render = current_time - last_render_time

            if dt_render >= min_render_interval:
                # 1. Calculate Statistics
                if dt_render > 0:
                    fps = 1.0 / dt_render
                    sim_speed = (dt * steps_accumulated) / dt_render
                    sim_speed_us = sim_speed * units.TIME * 1e6
                else:
                    fps = 0.0
                    sim_speed = 0.0
                    sim_speed_us = 0.0

                # 2. Reset Counters (Critical: Do this BEFORE drawing to ensure Loop timing logic holds even if drawing fails)
                last_render_time = current_time
                steps_accumulated = 0

                # 3. Draw
                try:
                    stdscr.erase()

                    # Header
                    stdscr.addstr(0, 0, "Paul Trap Simulation (TUI Mode - Free Running)", curses.A_BOLD)
                    stdscr.addstr(1, 0, "=" * 50)

                    # calculations
                    sim_time_us = trap.current_time * units.TIME * 1e6
                    real_temp_k = trap.real_temperature * units.TEMPERATURE

                    # Data Display
                    stdscr.addstr(2, 0, f"Ion Count:  {trap.num_ions}")
                    stdscr.addstr(3, 0, f"Sim Time:   {trap.current_time:8.2f} ({sim_time_us:8.2f} µs)")
                    stdscr.addstr(4, 0, f"Real Temp:  {trap.real_temperature:8.4e} ({real_temp_k:8.4e} K)")

                    # Configuration Section
                    stdscr.addstr(6, 0, "Current Configuration (Modify in code):", curses.A_BOLD | curses.A_UNDERLINE)
                    stdscr.addstr(7, 0, f"Frequencies (x,y,z): {trap.frequencies}")
                    
                    laser_status = "ACTIVE" if trap.is_laser_on else "OFF"
                    thermal_status = "ACTIVE" if trap.is_thermal_on else "OFF (Vacuum)"

                    stdscr.addstr(8, 0, f"Gamma Laser:   {trap.gamma_laser:.4f} [{laser_status}]")
                    stdscr.addstr(9, 0, f"Gamma Thermal: {trap.gamma_thermal:.4f} [{thermal_status}]")
                    stdscr.addstr(10, 0, f"Target Temp:   {trap.temperature:.4f}")
                    stdscr.addstr(11, 0, f"Noise Std Dev: {trap.random_force_std:.4e}")

                    # Controls Section
                    stdscr.addstr(13, 0, "Controls (Press key to toggle):", curses.A_BOLD | curses.A_UNDERLINE)

                    def checkbox(label, is_on, key_char):
                        mark = "[X]" if is_on else "[ ]"
                        return f"{key_char.upper()}: {mark} {label}"

                    stdscr.addstr(14, 0,  checkbox("Laser", state['laser'], 'l'))
                    stdscr.addstr(15, 0, checkbox("Real Laser Params", state['real_laser'], 'k'))
                    stdscr.addstr(16, 0, checkbox("Vacuum (No Thermal)", state['vacuum'], 'v'))
                    stdscr.addstr(17, 0, checkbox("Stochastic Noise", state['stochastic'], 's'))
                    stdscr.addstr(18, 0, "R: Reset Simulation")
                    stdscr.addstr(19, 0, "I: Start Lindemann Index Calc")
                    stdscr.addstr(20, 0, "UP/DOWN: Adjust Target FPS")
                    stdscr.addstr(21, 0, "Q: Quit")

                    # Lindemann Status
                    l_status = lindemann_state['status']
                    l_info = "Idle (Press 'I' to start)"
                    if l_status == 'EQ':
                        prog = (lindemann_state['step_counter'] / LINDEMANN_N_EQ) * 100
                        l_info = f"Equilibrating... {prog:.1f}% ({lindemann_state['step_counter']}/{LINDEMANN_N_EQ})"
                    elif l_status == 'PROD':
                        prog = (lindemann_state['step_counter'] / LINDEMANN_N_PROD) * 100
                        l_info = f"Sampling... {prog:.1f}% ({lindemann_state['step_counter']}/{LINDEMANN_N_PROD}) Samples: {lindemann_state['samples']}"
                    elif l_status == 'DONE':
                        l_info = f"Completed. Index: {lindemann_state['result']:.5f}"

                    stdscr.addstr(23, 0, "Lindemann Index:", curses.A_BOLD | curses.A_UNDERLINE)
                    stdscr.addstr(24, 0, l_info)

                    # Performance Stats
                    stdscr.addstr(26, 0, "Performance Stats:", curses.A_BOLD | curses.A_UNDERLINE)
                    stdscr.addstr(27, 0, f"FPS:        {fps:5.1f} / {target_fps:3.0f} (Target) | Batch: {steps_batch_size:4d}") 
                    stdscr.addstr(28, 0, f"Sim Speed:  {sim_speed:8.2f}/s ({sim_speed_us:8.2f} µs/sec) | dt: {dt}")
                    stdscr.addstr(29, 0, f"Perf:       {p_duration*1000:6.3f} ms/batch (Target: {TARGET_BATCH_DURATION*1000:.1f} ms)")

                    stdscr.refresh()

                except curses.error:
                    pass

            # No Sleep! Maximize CPU usage for calculation.

        except curses.error:
            pass
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    curses.wrapper(main)

'''
GUI and Visualization for Paul Trap Simulation
'''

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons

import units
from paul_trap import PaulTrap

# --- Simulation Parameters ---
num_ions = 10
dt = 0.01
init_freq = 8.0
init_gamma_laser = 2.0
init_gamma_thermal = 0.1

# Initialize Trap with new parameters
trap = PaulTrap(num_ions=num_ions, frequencies=(init_freq, init_freq, 1.0),
                gamma_laser=init_gamma_laser, gamma_thermal=init_gamma_thermal,
                temperature=0.0020)

# --- Visualization Setup ---
fig = plt.figure(figsize=(14, 8)) # Increased width for controls
plt.subplots_adjust(left=0.05, right=0.80, bottom=0.35)

ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Real-time Paul Trap Simulation")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Base limit
base_limit = 3.0
# We will set limits in the update_zoom callback or initially
ax.set_xlim(-base_limit, base_limit)
ax.set_ylim(-base_limit, base_limit)
ax.set_zlim(-base_limit, base_limit)

# Initial Plot Objects
colors = plt.cm.jet(np.linspace(0, 1, max(1, num_ions)))
# 3D scatter plot returns a Path3DCollection
if num_ions > 0:
    particles = ax.scatter(trap.positions[:, 0], trap.positions[:, 1], trap.positions[:, 2], c=colors, s=50)
else:
    particles = ax.scatter([], [], [], c=[], s=50)

# Real-time text display
fps_text = fig.text(0.05, 0.95, '', transform=fig.transFigure)
time_text = fig.text(0.05, 0.92, '', transform=fig.transFigure)
temp_text = fig.text(0.05, 0.89, '', transform=fig.transFigure)
count_text = fig.text(0.80, 0.95, '', transform=fig.transFigure, fontsize=12, fontweight='bold')

# --- Interactive Widgets ---
# Vertical Zoom Slider on the Right
ax_zoom = plt.axes([0.92, 0.25, 0.03, 0.5])
slider_zoom = Slider(
    ax_zoom,
    'Zoom',
    -2.0, # 10^-2 = 0.01x
    2.0,  # 10^2  = 100x
    valinit=0.0,
    orientation='vertical'
)

# Axes for sliders [left, bottom, width, height]
# Adjusted layout for more controls
# Left Column
ax_temp_text   = plt.axes([0.10, 0.24, 0.15, 0.03]) # Temp Text box
ax_freq_x      = plt.axes([0.10, 0.20, 0.35, 0.03])
ax_freq_y      = plt.axes([0.10, 0.16, 0.35, 0.03])
ax_freq_z      = plt.axes([0.10, 0.12, 0.35, 0.03])
ax_gamma_L     = plt.axes([0.10, 0.08, 0.35, 0.03])
ax_gamma_T     = plt.axes([0.10, 0.04, 0.35, 0.03])

# Switches (CheckButtons) - Right Column Bottom
# [left, bottom, width, height]
ax_switches    = plt.axes([0.55, 0.04, 0.12, 0.15])

# Button Controls (Far Right Column Bottom)
ax_reset       = plt.axes([0.80, 0.04, 0.08, 0.04])
ax_discard     = plt.axes([0.80, 0.10, 0.08, 0.04])
ax_catch_txt   = plt.axes([0.89, 0.10, 0.05, 0.04])
ax_add         = plt.axes([0.80, 0.16, 0.08, 0.04])
ax_add_txt     = plt.axes([0.89, 0.16, 0.05, 0.04])

# Sliders Initialization
slider_freq_x  = Slider(ax_freq_x, 'Freq X', 0.1, 20.0, valinit=trap.frequencies[0])
slider_freq_y  = Slider(ax_freq_y, 'Freq Y', 0.1, 20.0, valinit=trap.frequencies[1])
slider_freq_z  = Slider(ax_freq_z, 'Freq Z', 0.1, 20.0, valinit=trap.frequencies[2])
slider_gamma_L = Slider(ax_gamma_L, 'Laser Damp', 0.0, 5.0, valinit=trap.gamma_laser, valstep=0.01, color='cornflowerblue')
slider_gamma_T = Slider(ax_gamma_T, 'Therm Damp', 0.0, 0.5, valinit=trap.gamma_thermal, valstep=0.001, color='cornflowerblue')

# Text Box for Temperature
text_temp = TextBox(ax_temp_text, 'Temp T', initial=f'{trap.temperature:.4f}')
text_temp.label.set_horizontalalignment('right')

# Switches Initialization
check_switches = CheckButtons(ax_switches, ['Laser', 'Real Laser', 'Vacuum', 'Stochastic'], [True, False, True, True])

# SPF Slider (Steps Per Frame) - Optimization
# Placed vertically to the left of the Zoom slider
ax_spf = plt.axes([0.88, 0.25, 0.03, 0.5])
slider_spf = Slider(ax_spf, 'Steps', 1, 100, valinit=10, valstep=1, orientation='vertical')

# Buttons Initialization
button_add     = Button(ax_add,    'Add Ion', hovercolor='0.975')
text_add       = TextBox(ax_add_txt, '', initial='1')

# Toggle Button for Discard/Catch
button_discard = Button(ax_discard, 'Discard', hovercolor='0.975')
# Helper variable to track state
button_discard.cli_mode = 'discard'

text_catch     = TextBox(ax_catch_txt, '', initial='10')
button_reset   = Button(ax_reset,  'Reset', hovercolor='0.975')

# Hide Catch text initially
ax_catch_txt.set_visible(False)

# Labels Alignment
slider_freq_x.label.set_horizontalalignment('right')
slider_freq_y.label.set_horizontalalignment('right')
slider_freq_z.label.set_horizontalalignment('right')
slider_gamma_L.label.set_horizontalalignment('right')
slider_gamma_T.label.set_horizontalalignment('right')

# --- Callbacks ---

def update_params(val):
    trap.set_params(
        freq_x=slider_freq_x.val,
        freq_y=slider_freq_y.val,
        freq_z=slider_freq_z.val,
        gamma_laser=slider_gamma_L.val,
        gamma_thermal=slider_gamma_T.val
    )

def update_switches_callback(label):
    # Get current status of all checkbuttons
    status = check_switches.get_status() # (bool, bool, bool, bool)
    is_laser, is_real_laser, is_vacuum, is_stochastic = status

    # Constraint Logic: Link Laser and Real Laser
    if label == 'Laser' and not is_laser and is_real_laser:
        check_switches.set_active(1) # Toggle Real Laser to False
        is_real_laser = False
    elif label == 'Real Laser' and is_real_laser and not is_laser:
        check_switches.set_active(0) # Toggle Laser to True
        is_laser = True

    # Meaning of switches:
    # Laser: Checked = ON
    # Real Laser: Checked = Use units.GAMMA_LASER
    # Vacuum: Checked = ON (Environment is Vacuum) => Thermal Coupling is OFF.
    # Stochastic: Checked = ON

    if is_real_laser:
        # Override slider and physics value
        slider_gamma_L.set_val(units.GAMMA_LASER)

    trap.set_switches(laser=is_laser, thermal=(not is_vacuum), stochastic=is_stochastic)

    # UI Feedback logic
    # 1. Laser OFF -> Laser Damp fixed (disable interaction) and black
    if not is_laser:
        slider_gamma_L.set_active(False)
        if hasattr(slider_gamma_L, 'poly'):
             slider_gamma_L.poly.set_fc('black')
        slider_gamma_L.valtext.set_color('gray')
        slider_gamma_L.label.set_color('gray')
    elif is_real_laser:
        # Laser ON + Real Laser ON -> Fixed to unit value
        slider_gamma_L.set_active(False)
        if hasattr(slider_gamma_L, 'poly'):
             slider_gamma_L.poly.set_fc('orange') # Highlight special state
        slider_gamma_L.valtext.set_color('orange')
        slider_gamma_L.label.set_color('orange')
    else:
        # Laser ON + Real Laser OFF -> Adjustable
        slider_gamma_L.set_active(True)
        if hasattr(slider_gamma_L, 'poly'):
             slider_gamma_L.poly.set_fc('cornflowerblue') 
        slider_gamma_L.valtext.set_color('black')
        slider_gamma_L.label.set_color('black')

    # 2. Vacuum ON -> Thermal OFF -> Therm Damp and Temp T fixed and gray
    if is_vacuum: # Vacuum is ON, so Thermal connection is OFF
        slider_gamma_T.set_active(False)
        if hasattr(slider_gamma_T, 'poly'):
             slider_gamma_T.poly.set_fc('gray')
        slider_gamma_T.valtext.set_color('gray')
        slider_gamma_T.label.set_color('gray')
    else: # Vacuum is OFF, so Thermal connection is ON
        slider_gamma_T.set_active(True)
        if hasattr(slider_gamma_T, 'poly'):
             slider_gamma_T.poly.set_fc('cornflowerblue')
        slider_gamma_T.valtext.set_color('black')
        slider_gamma_T.label.set_color('black')

    # Temp T Logic
    # Disable if (Vacuum ON) OR (Stochastic OFF)
    if is_vacuum or (not is_stochastic):
        text_temp.set_active(False)
        text_temp.label.set_color('gray')
        text_temp.text_disp.set_color('gray')
        text_temp.ax.set_facecolor('lightgray')
    else:
        text_temp.set_active(True)
        text_temp.label.set_color('black')
        text_temp.text_disp.set_color('black')
        text_temp.ax.set_facecolor('white')

    # Force redraw to apply style changes immediately
    fig.canvas.draw_idle()

check_switches.on_clicked(update_switches_callback)

# Initial sync of UI state with CheckButton state
# (Since defaults are [True, True, True], Vacuum=True => Thermal=False initially)
update_switches_callback(None)

def submit_temp(text):
    try:
        val = float(text)
        val = max(0.0, min(10.0, val))
        trap.set_params(temperature=val)
        # Update text box to show valid/clamped value formatted
        text_temp.set_val(f'{val:.4f}')
    except ValueError:
        # Revert to current valid value
        text_temp.set_val(f'{trap.temperature:.4f}')

text_temp.on_submit(submit_temp)

slider_freq_x.on_changed(update_params)
slider_freq_y.on_changed(update_params)
slider_freq_z.on_changed(update_params)
slider_gamma_L.on_changed(update_params)
slider_gamma_T.on_changed(update_params)

def update_zoom(val):
    # val is log10(magnification)
    # val = 2.0 -> mag = 100 -> Zoom In -> Limit = Base / 100
    # val = -2.0 -> mag = 0.01 -> Zoom Out -> Limit = Base / 0.01
    mag = 10.0 ** val
    limit = base_limit / mag
    
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Update label to show magnification factor
    slider_zoom.valtext.set_text(f"x{mag:.2f}")

slider_zoom.on_changed(update_zoom)
# Initial call to set text
update_zoom(0.0)

# Matplotlib TextBox cursor position is tricky to access and manipulate reliably across versions.
# We will implement a simplified version: Pressing Up/Down increments/decrements by a fixed step (0.0001) if no specific cursor logic works perfectly,
# OR we try to be smart about cursor position if available.

def on_key_press(event):
    if event.inaxes == ax_temp_text and event.key in ['up', 'down']:
        try:
            current_str = text_temp.text
            current_val = float(current_str)

            # Use a fixed small step for high precision as requested by "0.0000" format
            # Trying to map cursor position to decimal place is unstable in MPL widgets (cursor index methods are private/undocumented)
            # So satisfying "increase value" is best done with fine control.
            # To simulate "changing that digit", users can just type the digit.
            # But the request says "change that digit". Let's try one more time to find cursor.

            # Private API access for matplotlib < 3.8
            cursor_idx = getattr(text_temp, '_cursor', None)
            # Public API for matplotlib >= 3.8
            if cursor_idx is None:
                cursor_idx = getattr(text_temp, 'cursor_index', len(current_str))

            step = 0.0001 # Default fallback

            if cursor_idx is not None:
                # Modify digit to the LEFT of cursor
                decimal_pos = current_str.find('.')
                if decimal_pos == -1:
                    decimal_pos = len(current_str)

                if cursor_idx > decimal_pos:
                    # Cursor in fractional part (right of dot)
                    # e.g. 12.3|4 (pos 4, dec 2). Left is '3'. 10^-1.
                    # 2 - 4 + 1 = -1.
                    power = decimal_pos - cursor_idx + 1
                else:
                    # Cursor in integer part (left of or at dot)
                    # e.g. 1|2.34 (pos 1, dec 2). Left is '1'. 10^1.
                    # 2 - 1 = 1.
                    power = decimal_pos - cursor_idx

                step = 10.0**power

            new_val = current_val
            if event.key == 'up':
                new_val = min(10.0, current_val + step)
            elif event.key == 'down':
                new_val = max(0.0, current_val - step)

            text_temp.set_val(f'{new_val:.4f}')
            trap.set_params(temperature=new_val)

            # Restore cursor (Best effort)
            if hasattr(text_temp, '_cursor'):
                text_temp._cursor = cursor_idx
                if hasattr(text_temp, '_rendercursor'):
                     text_temp._rendercursor()

        except (ValueError, AttributeError):
            pass

# Connect key press globally, but check inaxes inside
fig.canvas.mpl_connect('key_press_event', on_key_press)

slider_freq_x.on_changed(update_params)
slider_freq_y.on_changed(update_params)
slider_freq_z.on_changed(update_params)
slider_gamma_L.on_changed(update_params)
slider_gamma_T.on_changed(update_params)

def update_scatter():
    global particles, colors
    
    if particles is not None:
        particles.remove()

    colors = plt.cm.jet(np.linspace(0, 1, max(1, trap.num_ions)))
    if trap.num_ions > 0:
        particles = ax.scatter(trap.positions[:, 0], trap.positions[:, 1], trap.positions[:, 2], c=colors, s=50)
    else:
        particles = ax.scatter([], [], [], c=[], s=50)

def add_ion_callback(event):
    try:
        n = int(text_add.text)
        trap.add_ion(n)

        # If adding ions while in "Catch" mode (i.e. discarded/empty), switch button back to "Discard"
        if button_discard.cli_mode == 'catch':
             button_discard.label.set_text('Discard')
             ax_catch_txt.set_visible(False)
             button_discard.cli_mode = 'discard'

        update_scatter()
    except ValueError:
        pass

button_add.on_clicked(add_ion_callback)


def toggle_discard_catch(event):
    if button_discard.cli_mode == 'discard':
        # Switch to Catch mode UI
        button_discard.label.set_text('Catch')
        ax_catch_txt.set_visible(True)
        button_discard.cli_mode = 'catch'

        # Action: Clear ions
        trap.remove_all_ions()
        update_scatter()
    else:
        # Action: Catch ions
        try:
            n = int(text_catch.text)
            trap.catch_ions(n)

            # Switch back to Discard mode UI
            button_discard.label.set_text('Discard')
            ax_catch_txt.set_visible(False)
            button_discard.cli_mode = 'discard'
            update_scatter()
        except ValueError:
            pass

button_discard.on_clicked(toggle_discard_catch)

def reset_sim_callback(event):
    trap.reset()
    update_scatter()

button_reset.on_clicked(reset_sim_callback)

last_frame_time = time.time()

def update_frame(frame):
    global last_frame_time
    current_real_time = time.time()
    dt_real = current_real_time - last_frame_time
    last_frame_time = current_real_time

    steps = int(slider_spf.val)
    if dt_real > 0:
        fps = 1.0 / dt_real
        sim_speed = (dt * steps) / dt_real # simulation time units per real second
        sim_speed_us = sim_speed * units.TIME * 1e6 # microseconds per real second
    else:
        fps = 0.0
        sim_speed = 0.0
        sim_speed_us = 0.0

    fps_text.set_text(f'FPS: {fps:.1f} | Speed: {sim_speed:.2f}/s ({sim_speed_us:.2f} µs/s) | dt: {dt} x {steps}')

    # Physics Update: Multiple steps per frame for speed
    for _ in range(steps):
        trap.update(dt)

    # Visualization Update
    if trap.num_ions > 0 and particles is not None:
        particles._offsets3d = (trap.positions[:, 0], trap.positions[:, 1], trap.positions[:, 2])

        real_temp_k = trap.real_temperature * units.TEMPERATURE
        temp_text.set_text(f'Real Temp: {trap.real_temperature:.4e} | {real_temp_k:.4e} K')
    else:
         # If no ions, ensure plot is empty
        particles._offsets3d = ([], [], [])
        temp_text.set_text('Trap Empty')

    # Update real-time text
    real_time_us = trap.current_time * units.TIME * 1e6
    time_text.set_text(f'Sim Time: {trap.current_time:.2f} | {real_time_us:.2f} µs')

    count_text.set_text(f'Ion Count: {trap.num_ions}')

    return particles, time_text, temp_text, fps_text, count_text

ani = FuncAnimation(fig, update_frame, frames=200, interval=10, blit=False)

plt.show()

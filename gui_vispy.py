'''
VisPy + PyQt5 GUI for Paul Trap Simulation (GPU Accelerated)
Matches functionality of the original Matplotlib GUI.
'''

import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton, 
                             QLineEdit, QGroupBox, QFrame)
from PyQt5.QtCore import QTimer, Qt
from vispy import scene
from vispy.color import get_colormap

def select_backend():
    # Priority: JAX -> PyTorch -> NumPy
    try:
        import jax
        from paul_trap_jax import PaulTrap
        platform = jax.devices()[0].platform.lower()
        device = "GPU" if platform == "gpu" else "CPU"
        return PaulTrap, "JAX", device
    except Exception:
        pass

    try:
        import torch
        from paul_trap_torch import PaulTrap
        device = "GPU" if torch.cuda.is_available() else "CPU"
        return PaulTrap, "PyTorch", device
    except Exception:
        pass

    from paul_trap import PaulTrap
    return PaulTrap, "NumPy", "CPU"


PaulTrap, BACKEND_NAME, DEVICE_NAME = select_backend()
print(f"Using backend: {BACKEND_NAME} ({DEVICE_NAME})")

import units

# --- Configuration ---
DT = 0.01
PRECISION = 'float64' # 'float32' or 'float64'

class NumericLineEdit(QLineEdit):
    """
    QLineEdit that handles Up/Down arrow keys to increment/decrement the digit 
    at the cursor position (or the digit to the left of the cursor).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            self._adjust_value(1 if event.key() == Qt.Key_Up else -1)
        else:
            super().keyPressEvent(event)

    def _adjust_value(self, direction):
        text = self.text()
        cursor_pos = self.cursorPosition()

        # Don't do anything if empty
        if not text:
            return

        # Find the power of 10 to adjust based on cursor position
        # We need to parse where the decimal point is
        try:
            val = float(text)
        except ValueError:
            return

        if '.' in text:
            dot_pos = text.find('.')
            if cursor_pos <= dot_pos:
                # Cursor is before decimal point: 10^(dot_pos - cursor_pos)
                exp = dot_pos - cursor_pos
                if exp < 0: exp = 0
                # e.g. "123.45", dot at 3.
                # cursor at 3 (after '3'): pos=3 -> exp=0 -> 10^0=1 (change units)
                # cursor at 2 (after '2'): pos=2 -> exp=1 -> 10^1=10 (change tens)
            else:
                # Cursor is after decimal point: 10^(dot_pos - cursor_pos)
                exp = dot_pos - cursor_pos
                # e.g. "123.456" dot at 3.
                # cursor at 4 (after '4'): pos=4 -> exp=-1 -> 10^-1=0.1
        else:
            # No decimal point, integer logic
            exp = len(text) - cursor_pos

        # If cursor is at the very beginning (pos=0), treat same as pos=1 if number is positive?
        # Actually standard behavior: cursor is "between" characters.
        # If cursor is to the RIGHT of a digit, we want to modify that digit.

        # Adjust logic: cursor position i means we want to modify the digit at i-1?
        # Let's stick to power of 10 logic.

        step = 10.0 ** exp

        # Update value
        new_val = val + direction * step

        # Limit to reasonable range (0 to 10 based on original constraint)
        new_val = max(0.0, min(10.0, new_val))

        # Format back to string, preserving precision roughly
        # If we are changing small decimals, ensure we show enough digits
        if abs(step) < 1e-4:
            fmt = "{:.5f}"
        elif abs(step) < 1e-3:
            fmt = "{:.4f}"
        else:
            fmt = "{:.4f}" # Default to 4

        new_text = fmt.format(new_val)
        self.setText(new_text)

        # Restore cursor position (or adjust if length changed?)
        # Simple approach: keep cursor at same offset from end if length changed?
        # Or just try to put it back at numerical position.
        # Let's just try to keep simple index, clamped to length
        new_pos = min(cursor_pos, len(new_text))
        self.setCursorPosition(new_pos)

        # Emit returnPressed to trigger update logic in main window
        self.returnPressed.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Paul Trap Simulation (VisPy) - {BACKEND_NAME} / {DEVICE_NAME}")
        self.resize(1600, 900)

        # --- Physics Initialization ---
        self.num_ions = 10
        init_freq = 8.0
        init_gamma_laser = 2.0
        init_gamma_thermal = 0.1
        self.trap = PaulTrap(
            num_ions=self.num_ions,
            frequencies=(init_freq, init_freq, 1.0),
            gamma_laser=init_gamma_laser,
            gamma_thermal=init_gamma_thermal,
            temperature=0.0020,
            precision=PRECISION
        )

        # Performance Tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.spf = 10  # Steps per frame

        # --- UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. VisPy Canvas (Left Side)
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 25

        # Visuals
        # XYZ Axis
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)

        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.cmap = get_colormap('jet')

        # Scale Bar (Text)
        self.scale_text = scene.visuals.Text(
            text='',
            pos=(100, 50),
            color='black',
            font_size=16,
            anchor_x='center',
            anchor_y='bottom',
            parent=self.canvas.scene
        )
        self.scale_bar = scene.visuals.Line(
            pos=np.array([[0, 0], [100, 0]]),
            color='black',
            width=2,
            parent=self.canvas.scene
        )
        # We'll update position and content in resize event or update loop
        self.canvas.events.resize.connect(self.on_canvas_resize)

        # Add canvas to layout
        # We need to wrap VisPy canvas in a widget if using PyQt
        main_layout.addWidget(self.canvas.native, stretch=3)

        # 2. Controls Panel (Right Side)
        controls_panel = QWidget()
        controls_panel.setFixedWidth(400)
        controls_layout = QVBoxLayout(controls_panel)
        main_layout.addWidget(controls_panel, stretch=1)

        # --- Ion Count (Prominent Display) ---
        self.lbl_count = QLabel(f"Ion Count: {self.num_ions}")
        self.lbl_count.setStyleSheet("font-size: 24px; font-weight: bold; color: black; padding: 10px;")
        self.lbl_count.setAlignment(Qt.AlignCenter)
        self.lbl_count.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        controls_layout.addWidget(self.lbl_count)

        # --- Info Section ---
        self.info_group = QGroupBox("Status")
        info_layout = QVBoxLayout()
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_speed = QLabel("Speed: 0.0/s")
        self.lbl_time = QLabel("Sim Time: 0.00 µs")
        self.lbl_real_temp = QLabel("Real Temp: 0.0000 K")
        self.lbl_device_and_backend = QLabel(f"Device: {DEVICE_NAME} | Backend: {BACKEND_NAME} | Precision: {PRECISION}")

        info_layout.addWidget(self.lbl_fps)
        info_layout.addWidget(self.lbl_speed)
        info_layout.addWidget(self.lbl_time)
        info_layout.addWidget(self.lbl_real_temp)
        info_layout.addWidget(self.lbl_device_and_backend)
        self.info_group.setLayout(info_layout)
        controls_layout.addWidget(self.info_group)

        # --- Parameters Section ---
        self.params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        # Helper to create slider
        def create_slider(label, min_val, max_val, init_val, scale=100.0, callback=None):
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(f"{label}: {init_val:.2f}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * scale))
            slider.setMaximum(int(max_val * scale))
            slider.setValue(int(init_val * scale))

            def on_change(val):
                real_val = val / scale
                lbl.setText(f"{label}: {real_val:.2f}")
                if callback: callback(real_val)

            slider.valueChanged.connect(on_change)

            layout.addWidget(lbl)
            layout.addWidget(slider)

            # Store references to update programmatically later
            slider.lbl_ref = lbl
            slider.label_template = label
            slider.scale_factor = scale

            return container, slider

        # Sliders
        # Using lambda wrapper to update all params when one changes (simplifies sync)
        c1, self.sl_freq_x = create_slider("Freq X", 0.1, 20.0, init_freq, callback=lambda v: self.update_physics_params())
        c2, self.sl_freq_y = create_slider("Freq Y", 0.1, 20.0, init_freq, callback=lambda v: self.update_physics_params())
        c3, self.sl_freq_z = create_slider("Freq Z", 0.1, 20.0, 1.0, callback=lambda v: self.update_physics_params())
        c4, self.sl_gamma_L = create_slider("Laser Damp", 0.0, 5.0, init_gamma_laser, scale=10000.0, callback=lambda v: self.update_physics_params())
        c5, self.sl_gamma_T = create_slider("Therm Damp", 0.0, 0.5, init_gamma_thermal, scale=1000.0, callback=lambda v: self.update_physics_params())

        params_layout.addWidget(c1)
        params_layout.addWidget(c2)
        params_layout.addWidget(c3)
        params_layout.addWidget(c4)
        params_layout.addWidget(c5)

        # Target Temperature Input
        temp_container = QWidget()
        temp_layout = QHBoxLayout(temp_container)
        temp_layout.setContentsMargins(0, 0, 0, 0)
        temp_layout.addWidget(QLabel("Target Temp:"))

        # Custom input handling logic
        class TempInput(QLineEdit):
            def __init__(self, parent_gui, init_text):
                super().__init__(init_text)
                self.parent_gui = parent_gui

            def keyPressEvent(self, event):
                if event.key() in (Qt.Key_Up, Qt.Key_Down):
                    self._adjust_value(1 if event.key() == Qt.Key_Up else -1)
                else:
                    super().keyPressEvent(event)

            def _adjust_value(self, direction):
                text = self.text()
                cursor_pos = self.cursorPosition()
                try:
                    val = float(text)
                except ValueError:
                    return

                # Determine power of 10 based on cursor position relative to decimal point
                if '.' in text:
                    dot_pos = text.find('.')
                    if cursor_pos <= dot_pos:
                        # Left of decimal: 10^(dot_pos - cursor_pos)
                        # e.g. "12.34", dot=2. pos=2 -> exp=0 -> 10^0=1
                        # pos=1 -> exp=1 -> 10^1=10
                        exp = dot_pos - cursor_pos
                    else:
                        # Right of decimal: 10^(dot_pos - cursor_pos + 1) NO
                        # e.g. "12.34", dot=2. pos=3 (after '3') -> exp=-1 -> 10^-1=0.1
                        # pos=4 (after '4') -> exp=-2 -> 10^-2=0.01
                        exp = dot_pos - cursor_pos + 1
                else:
                    # No decimal, treat end as dot position
                    exp = len(text) - cursor_pos

                # Prevent overflow or underflow of exp if cursor is far away
                step = 10.0 ** exp
                new_val = val + direction * step

                # Clamp (0 to 10.0 per original logic)
                new_val = max(0.0, min(10.0, new_val))

                # Format
                new_text = f"{new_val:.4f}"
                self.setText(new_text)

                # Restore cursor position (clamp to new length)
                # IMPORTANT: Do not reset cursor to end. Keep it where it was.
                self.setCursorPosition(min(cursor_pos, len(new_text)))

                # Trigger update
                # self.returnPressed.emit() # Cancel auto-submit. User must press Enter manually.

        self.txt_temp = TempInput(self, f"{self.trap.temperature:.4f}")
        self.txt_temp.returnPressed.connect(self.on_temp_submit)
        temp_layout.addWidget(self.txt_temp)
        params_layout.addWidget(temp_container)

        self.params_group.setLayout(params_layout)
        controls_layout.addWidget(self.params_group)

        # --- Speed Control ---
        speed_group = QGroupBox("Performance")
        speed_layout = QVBoxLayout()
        # Scale=1.0 because SPF is int
        c_spf, self.sl_spf = create_slider("Steps/Frame", 1, 100, 10, scale=1.0, callback=lambda v: self.on_spf_change(v))
        speed_layout.addWidget(c_spf)
        speed_group.setLayout(speed_layout)
        controls_layout.addWidget(speed_group)

        # --- Switches ---
        self.switches_group = QGroupBox("Switches")
        switches_layout = QVBoxLayout()

        self.chk_laser = QCheckBox("Laser Cooling")
        self.chk_laser.setChecked(True)
        self.chk_laser.toggled.connect(self.update_switches)

        self.chk_real_laser = QCheckBox("Real Laser Params")
        self.chk_real_laser.setChecked(False)
        self.chk_real_laser.toggled.connect(self.update_switches)

        self.chk_vacuum = QCheckBox("Vacuum (No Thermal)")
        self.chk_vacuum.setChecked(True)
        self.chk_vacuum.toggled.connect(self.update_switches)

        self.chk_stochastic = QCheckBox("Stochastic Noise")
        self.chk_stochastic.setChecked(True)
        self.chk_stochastic.toggled.connect(self.update_switches)

        switches_layout.addWidget(self.chk_laser)
        switches_layout.addWidget(self.chk_real_laser)
        switches_layout.addWidget(self.chk_vacuum)
        switches_layout.addWidget(self.chk_stochastic)
        self.switches_group.setLayout(switches_layout)
        controls_layout.addWidget(self.switches_group)

        # --- Actions ---
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()

        # Add Ion
        add_container = QWidget()
        add_layout = QHBoxLayout(add_container)
        self.btn_add = QPushButton("Add Ion")
        self.btn_add.clicked.connect(self.on_add_ion)
        self.txt_add_count = QLineEdit("1")
        self.txt_add_count.setFixedWidth(50)
        add_layout.addWidget(self.btn_add)
        add_layout.addWidget(self.txt_add_count)
        actions_layout.addWidget(add_container)

        # Catch / Discard
        catch_container = QWidget()
        catch_layout = QHBoxLayout(catch_container)
        # Default state: Discard mode (button says Discard)
        self.btn_discard = QPushButton("Discard")
        self.btn_discard.clicked.connect(self.on_discard_catch)
        self.txt_catch_count = QLineEdit("10")
        self.txt_catch_count.setFixedWidth(50)
        self.txt_catch_count.setVisible(False)
        catch_layout.addWidget(self.btn_discard)
        catch_layout.addWidget(self.txt_catch_count)
        actions_layout.addWidget(catch_container)

        # Reset
        self.btn_reset = QPushButton("Reset Simulation")
        self.btn_reset.clicked.connect(self.on_reset)
        actions_layout.addWidget(self.btn_reset)

        actions_group.setLayout(actions_layout)
        controls_layout.addWidget(actions_group)

        # Add spacer at bottom
        controls_layout.addStretch()

        # --- Initial Logic Sync ---
        self.discard_mode = True # True = Discard Button shown, False = Catch Button shown
        # IMPORTANT: Trigger initial param set to match sliders
        self.update_physics_params()
        self.update_switches()

        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(10) # 10ms

    # --- Callbacks ---

    def on_canvas_resize(self, event):
        """Update scale text position when canvas resizes."""
        # VisPy canvas coords: (0,0) is top-left usually?
        # Actually in SceneCanvas scene, it matches underlying backend coords.
        # But SceneCanvas.size gives (w, h).
        w, h = self.canvas.size
        # Position at top-right with margin
        self.scale_text.pos = (w - 180, 30)

    def update_physics_params(self):
        # Read from sliders directly
        g_laser = float(self.sl_gamma_L.value() / self.sl_gamma_L.scale_factor)

        # Override with exact Real Laser value if checked
        if self.chk_real_laser.isChecked():
            g_laser = units.GAMMA_LASER

        self.trap.set_params(
            freq_x=float(self.sl_freq_x.value() / self.sl_freq_x.scale_factor),
            freq_y=float(self.sl_freq_y.value() / self.sl_freq_y.scale_factor),
            freq_z=float(self.sl_freq_z.value() / self.sl_freq_z.scale_factor),
            gamma_laser=g_laser,
            gamma_thermal=float(self.sl_gamma_T.value() / self.sl_gamma_T.scale_factor)
        )

    def on_spf_change(self, val):
        self.spf = int(val)

    def on_temp_submit(self):
        try:
            val = float(self.txt_temp.text())
            val = max(0.0, min(10.0, val))
            self.trap.set_params(temperature=float(val))
            self.txt_temp.setText(f'{val:.4f}')
        except ValueError:
            self.txt_temp.setText(f'{self.trap.temperature:.4f}')

    def update_switches(self):
        # Sync Checks logic (Laser/Real Laser Constraint)
        sender = self.sender()

        if sender == self.chk_laser:
            # If Laser turned OFF but Real Laser was ON, turn Real OFF
            # But the logic in gui.py was:
            # if label == 'Laser' and not is_laser and is_real_laser: ...
            # Actually, gui.py logic:
            # Laser=OFF -> Slider Disabled.
            # Real=ON -> Slider Disabled & Set Value.
            pass

        elif sender == self.chk_real_laser:
            # If Real Laser turned ON, ensure Laser is ON
            if self.chk_real_laser.isChecked() and not self.chk_laser.isChecked():
                self.chk_laser.setChecked(True) # Block signal? No, let it cascade.

        is_laser = self.chk_laser.isChecked()
        is_real = self.chk_real_laser.isChecked()
        is_vacuum = self.chk_vacuum.isChecked()
        is_stochastic = self.chk_stochastic.isChecked()

        # Additional constraint implementation from gui.py:
        # if label == 'Laser' and not is_laser and is_real_laser: check_switches.set_active(1) # Toggle Real Laser to False
        if not is_laser and is_real:
            self.chk_real_laser.setChecked(False)
            is_real = False

        # Apply Real Laser Value
        if is_real:
            val = units.GAMMA_LASER
            # Update slider UI to reflect the fixed value
            # Block signals to prevent recursion loop
            self.sl_gamma_L.blockSignals(True)
            self.sl_gamma_L.setValue(int(val * self.sl_gamma_L.scale_factor))
            self.sl_gamma_L.lbl_ref.setText(f"{self.sl_gamma_L.label_template}: {val:.4f}")
            self.sl_gamma_L.blockSignals(False)

        self.trap.set_switches(
            laser=is_laser,
            thermal=(not is_vacuum),
            stochastic=is_stochastic
        )

        # Enable/Disable UI elements
        self.sl_gamma_L.setEnabled(is_laser and not is_real)
        self.sl_gamma_T.setEnabled(not is_vacuum)
        self.txt_temp.setEnabled(not is_vacuum and is_stochastic)

        # Re-apply params to ensure correct state if slider was overridden
        if is_real:
            self.update_physics_params()

    def on_add_ion(self):
        try:
            n = int(self.txt_add_count.text())
            self.trap.add_ion(n)
            self.lbl_count.setText(f"Ion Count: {self.trap.num_ions}")

            # If in Catch mode (showing Catch button, meaning empty), switch back to Discard
            if not self.discard_mode:
                self.discard_mode = True
                self.btn_discard.setText("Discard")
                self.txt_catch_count.setVisible(False)

                # Reconstruct markers if needed (VisPy handles size change well usually but let's be safe)
                # Nothing special needed, next update loop handles it.
        except ValueError:
            pass

    def on_discard_catch(self):
        if self.discard_mode:
            # Switch to Catch Mode
            # Action: Clear ions
            self.trap.remove_all_ions()
            self.lbl_count.setText("Ion Count: 0")

            self.discard_mode = False
            self.btn_discard.setText("Catch")
            self.txt_catch_count.setVisible(True)

            # Clear plot immediately
            self.markers.set_data(pos=np.zeros((0, 3)))
            self.canvas.update()

        else:
            # Action: Catch Ions
            try:
                n = int(self.txt_catch_count.text())
                self.trap.catch_ions(n)
                self.lbl_count.setText(f"Ion Count: {self.trap.num_ions}")

                self.discard_mode = True
                self.btn_discard.setText("Discard")
                self.txt_catch_count.setVisible(False)
            except ValueError:
                pass

    def on_reset(self):
        self.trap.reset()
        self.lbl_count.setText(f"Ion Count: {self.trap.num_ions}")

    def update_loop(self):
        # Physics Step
        self.trap.update_n_steps(DT, self.spf)

        # Performance Calc
        self.frame_count += 1
        current_time = time.time()
        dt_render = current_time - self.last_frame_time
        if dt_render >= 0.2: # Update text every ~200ms to avoid flicker
            fps = self.frame_count / dt_render if dt_render > 0 else 0
            self.lbl_fps.setText(f"FPS: {fps:.1f}")

            sim_speed = (DT * self.spf * self.frame_count) / dt_render if dt_render > 0 else 0
            sim_speed_us = sim_speed * units.TIME * 1e6
            self.lbl_speed.setText(f"Speed: {sim_speed:.1f}/s ({sim_speed_us:.1f} µs/s) | dt: {DT}x{self.spf}")

            real_time_us = self.trap.current_time * units.TIME * 1e6
            self.lbl_time.setText(f"Sim Time: {self.trap.current_time:.2f} ({real_time_us:.2f} µs)")

            real_temp_k = self.trap.real_temperature * units.TEMPERATURE
            self.lbl_real_temp.setText(f"Real Temp: {self.trap.real_temperature:.4e} ({real_temp_k:.4e} K)")

            self.last_frame_time = current_time
            self.frame_count = 0

        # Visualization Update
        # Update Scale Bar
        # Screen dimensions
        w, h = self.canvas.size
        # Calculation: Visible height at focus distance
        # For perspective camera: 2 * dist * tan(fov/2)
        fov_rad = np.deg2rad(self.view.camera.fov)
        dist = self.view.camera.distance
        if dist < 0.1: dist = 0.1 # Avoid near-zero distance issues

        visible_h = 2 * dist * np.tan(fov_rad / 2)
        if visible_h > 1e-9:
            pixels_per_unit = h / visible_h
        else:
            pixels_per_unit = 1000.0 # Fallback

        if w > 0:
            # Adaptive Scale Logic
            # Goal: Find a "nice" length (1, 2, 5) in microns that fits well in ~100-150 pixels
            target_px = 150.0 
            target_sim = target_px / pixels_per_unit
            target_um = target_sim * units.LENGTH * 1e6

            if target_um <= 0: target_um = 1.0 # Safety

            # Find nearest power of 10
            exponent = np.floor(np.log10(target_um))
            mantissa = target_um / (10 ** exponent)

            # Snap to 1, 2, 5
            if mantissa < 1.5:
                nice_mantissa = 1.0
            elif mantissa < 3.5:
                nice_mantissa = 2.0
            elif mantissa < 7.5:
                nice_mantissa = 5.0
            else:
                nice_mantissa = 10.0

            nice_um = nice_mantissa * (10 ** exponent)

            # Convert back to pixels
            nice_sim = nice_um * 1e-6 / units.LENGTH
            bar_px = nice_sim * pixels_per_unit

            # Formatting text
            if nice_um >= 1000:
                txt = f"{nice_um/1000:.0f} mm" if (nice_um/1000).is_integer() else f"{nice_um/1000:.1f} mm"
            elif nice_um < 1:
                txt = f"{nice_um * 1000:.0f} nm"
            else:
                txt = f"{nice_um:.0f} µm" if nice_um.is_integer() else f"{nice_um:.1f} µm"

            self.scale_text.text = txt
            self.scale_text.font_size = 10 # Screen points, fixed

            # Position: Bottom Right
            # Let's put the bar 20px from bottom, 20px from right
            margin_right = 30
            bar_y = 30

            bar_end_x = w - margin_right
            bar_start_x = bar_end_x - bar_px

            # Text centered over bar
            self.scale_text.pos = (bar_start_x + bar_px/2, bar_y + 5)

            self.scale_bar.set_data(pos=np.array([
                [bar_start_x, bar_y],
                [bar_end_x, bar_y]
            ]))

        pos = self.trap.positions
        if len(pos) > 0:
            # Map colors based on index to keep consistent color per particle
            indices = np.linspace(0, 1, len(pos))
            colors = self.cmap.map(indices)

            self.markers.set_data(
                pos=pos,
                face_color=colors,
                size=12,
                edge_width=0
            )
        else:
            self.markers.set_data(pos=np.zeros((0, 3)))

        self.canvas.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

#Constants
C = 3e8  # Speed of light (m/s)
K_B = 1.38e-23  # Boltzmann constant

class Target:
    """Represents a single target with position, velocity, and RCS."""
    def __init__(self, target_id, x, y, vx, vy, rcs, is_clutter=False):
        self.id = target_id
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.rcs = rcs
        self.is_clutter = is_clutter
        self.accel = np.array([0.0, 0.0], dtype=float)

    def update(self, dt):
        """Update target position based on its velocity and acceleration."""
        if not self.is_clutter:
            if np.random.rand() < 0.01:
                self.accel = (np.random.rand(2) - 0.5) * 20
            self.vel += self.accel * dt
            self.pos += self.vel * dt

#RRE and MTI Functions
def calculate_rre_max_range(radar_params):
    """Calculates the maximum detection range based on the RRE"""
    power_tx = radar_params['tx_power']
    gain_db = radar_params['antenna_gain']
    wavelength = radar_params['wavelength']
    rcs = radar_params['rcs']
    snr_min_db = radar_params['snr_min_db']
    #System noise parameters
    system_temp, bandwidth, noise_figure_db = 290.0, 1.0e6, 3.0
    #Convert dB values to linear scale
    gain_linear = 10**(gain_db / 10.0)
    snr_min_linear = 10**(snr_min_db / 10.0)
    noise_figure_linear = 10**(noise_figure_db / 10.0)
    #Calculate receiver noise floor
    noise_power = K_B * system_temp * bandwidth * noise_figure_linear
    #Calculate minimum detectable signal (S_min)
    s_min = noise_power * snr_min_linear
    #Calculate max range from the RRE formula
    numerator = power_tx * gain_linear**2 * wavelength**2 * rcs
    denominator = (4 * np.pi)**3 * s_min
    return (numerator/denominator)**0.25 if denominator else np.inf

def moving_target_indicator(target):
    """Performs MTI by checking the target's radial velocity"""
    range_mag = np.linalg.norm(target.pos)
    if range_mag == 0: return 0, False
    radial_velocity = np.dot(target.vel, target.pos) / range_mag
    is_moving = abs(radial_velocity) > 1.0
    return radial_velocity, is_moving

#GUI

class RadarApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Radar Simulation")
        self.geometry("1200x1000") #window size
        self.is_running = False
        self.beam_angle = 0.0
        self.time_step = 0.1
        self.sim_time = 0.0
        self._setup_gui()
        self.reset_simulation()

    def _setup_gui(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame, width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        sim_controls = ttk.LabelFrame(left_panel, text="Simulation Controls", padding=1)
        sim_controls.pack(fill=tk.X, pady=5)
        self.start_button = ttk.Button(sim_controls, text="Start", command=self.toggle_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_controls, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=5)

        target_gen_frame = ttk.LabelFrame(left_panel, text="Target Generation", padding=10)
        target_gen_frame.pack(fill=tk.X, pady=5)
        self.num_targets_var = tk.IntVar(value=8)
        self.num_clutter_var = tk.IntVar(value=5)
        self._create_entry(target_gen_frame, "Aircraft Targets:", self.num_targets_var, 0)
        self._create_entry(target_gen_frame, "Mountains (Clutter):", self.num_clutter_var, 1)

        radar_params_frame = ttk.LabelFrame(left_panel, text="Radar Parameters", padding=1)
        radar_params_frame.pack(fill=tk.X, pady=5)
        self.tx_power_var = tk.DoubleVar(value=1.5e6)
        self.antenna_gain_var = tk.DoubleVar(value=40)
        self.frequency_var = tk.DoubleVar(value=1.3e9)
        self.snr_min_db_var = tk.DoubleVar(value=14.0)
        self.mti_enabled_var = tk.BooleanVar(value=True)
        self._create_entry(radar_params_frame, "Tx Power (W):", self.tx_power_var, 0)
        self._create_entry(radar_params_frame, "Antenna Gain (dBi):", self.antenna_gain_var, 1)
        self._create_entry(radar_params_frame, "Frequency (Hz):", self.frequency_var, 2)
        self._create_entry(radar_params_frame, "Min SNR for Detection (dB):", self.snr_min_db_var, 3)
        ttk.Checkbutton(radar_params_frame, text="Enable MTI", variable=self.mti_enabled_var).grid(row=4, columnspan=2, pady=5)
        
        log_frame = ttk.LabelFrame(left_panel, text="Detection Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=45, width=45, font=("Courier New", 9), bg="black", fg="lime")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        self.fig_ppi, self.ax_ppi = plt.subplots(figsize=(8, 8))
        self.canvas_ppi = FigureCanvasTkAgg(self.fig_ppi, master=main_frame)
        self.canvas_ppi.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _create_entry(self, parent, text, var, row):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=15).grid(row=row, column=1, sticky="w")

    def toggle_simulation(self):
        self.is_running = not self.is_running
        self.start_button.config(text="Pause" if self.is_running else "Start")
        if self.is_running:
            self.run_simulation_step()

    def reset_simulation(self):
        self.is_running = False
        self.start_button.config(text="Start")
        self.sim_time = 0.0
        self.display_range = 300e3
        self.targets = self.generate_targets(self.num_targets_var.get(), self.num_clutter_var.get(), self.display_range)
        self.draw_ppi_and_log()
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)

    def generate_targets(self, num_targets, num_clutter, max_range):
        targets = []
        for i in range(num_targets):
            angle, r = np.random.rand() * 2 * np.pi, np.sqrt(np.random.rand()) * max_range * 0.95
            x, y = r * np.cos(angle), r * np.sin(angle)
            vx, vy = (np.random.rand(2) - 0.5) * 600
            rcs = 10**(np.random.uniform(-1, 1.5))
            targets.append(Target(f"T{i}", x, y, vx, vy, rcs))
        for i in range(num_clutter):
            angle, r = np.random.rand() * 2 * np.pi, np.random.uniform(0.1, 0.9) * max_range
            x, y = r * np.cos(angle), r * np.sin(angle)
            rcs = np.random.uniform(50, 500)
            targets.append(Target(f"C{i}", x, y, 0, 0, rcs, is_clutter=True))
        return targets

    def run_simulation_step(self):
        if not self.is_running: return
        
        self.sim_time += self.time_step
        self.beam_angle = (self.beam_angle + 2.0) % 360
        
        for target in self.targets:
            target.update(self.time_step)
            if np.linalg.norm(target.pos) > self.display_range * 1.1:
                target.pos *= -0.95

        self.draw_ppi_and_log()
        self.after(50, self.run_simulation_step)

    def draw_ppi_and_log(self):
        self.ax_ppi.clear()
        self.ax_ppi.set_facecolor('black')
        self.ax_ppi.set_aspect('equal')
        self.ax_ppi.set_title("PPI Display (RRE Detection)")
        self.ax_ppi.set_xlim(-self.display_range, self.display_range)
        self.ax_ppi.set_ylim(-self.display_range, self.display_range)

        for r in np.linspace(0.25, 1.0, 4) * self.display_range:
            self.ax_ppi.add_patch(Circle((0, 0), r, fill=False, edgecolor='lime', linestyle='--', alpha=0.5))
            self.ax_ppi.text(5000, r + 5000, f'{r/1000:.0f} km', color='lime', fontsize=12)

        bx, by = self.display_range * np.cos(np.deg2rad(self.beam_angle)), self.display_range * np.sin(np.deg2rad(self.beam_angle))
        self.ax_ppi.plot([0, bx], [0, by], color='lime', linewidth=1)

        radar_params = {'tx_power': self.tx_power_var.get(), 'antenna_gain': self.antenna_gain_var.get(),
                        'wavelength': C / self.frequency_var.get(), 'rcs': 1.0, 'snr_min_db': self.snr_min_db_var.get()}
        
        log_string = f"Time: {self.sim_time:.1f}s | Beam: {self.beam_angle:.0f}°\n"
        log_string += "----------------------------------------\n"
        log_string += "ID   | Status   | Range(km) | R_max(km)\n"
        log_string += "----------------------------------------\n"

        for target in self.targets:
            target_angle = (np.rad2deg(np.arctan2(target.pos[1], target.pos[0])) + 360) % 360
            beam_diff = abs((self.beam_angle - target_angle + 180) % 360 - 180)
            if not (beam_diff < 5.0): continue

            radar_params['rcs'] = target.rcs
            target_max_range = calculate_rre_max_range(radar_params)
            target_current_range = np.linalg.norm(target.pos)
            is_detected = target_current_range <= target_max_range

            if is_detected:
                radial_velocity, is_moving = moving_target_indicator(target)
                status = "NO MTI"
                if self.mti_enabled_var.get():
                    if is_moving:
                        color = 'red' if radial_velocity < 0 else 'deepskyblue'
                        self.ax_ppi.plot(target.pos[0], target.pos[1], 'o', color=color, markersize=7)
                        vel_line_end = target.pos + target.vel * 50
                        self.ax_ppi.plot([target.pos[0], vel_line_end[0]], [target.pos[1], vel_line_end[1]], color=color, linewidth=1.5)
                        status = "MOVING"
                    else:
                        self.ax_ppi.plot(target.pos[0], target.pos[1], '^', color='white', markersize=8)
                        status = "CLUTTER"
                else:
                    self.ax_ppi.plot(target.pos[0], target.pos[1], 'x', color='red', markersize=8)
                
                log_string += f"{target.id:<4} | {status:<8} | {target_current_range/1000:<9.1f} | {target_max_range/1000:<8.1f}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, log_string)
        self.log_text.config(state=tk.DISABLED)
        self.canvas_ppi.draw()

if __name__ == "__main__":
    app = RadarApp()
    app.mainloop()
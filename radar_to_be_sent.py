import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

C = 3e8  # Speed of light (m/s)
K_B = 1.38e-23  # Boltzmann constant

class Target:
    #Represents target
    def __init__(self, target_id, x, y, vx, vy, rcs, is_clutter=False):
        self.id = target_id
        self.pos = np.array([x, y], dtype=float) #store target's position
        self.vel = np.array([vx, vy], dtype=float) #store target's velocity
        self.rcs = rcs #radius of cross section of target - how detectable is the target
        self.is_clutter = is_clutter #storing stationary target
        self.accel = np.array([0.0, 0.0], dtype=float) #storing target's acceleration

    def update(self, dt):
        """Update target position based on its velocity and acceleration."""
        if not self.is_clutter: #if not clutter
            if np.random.rand() < 0.01: #target will now change course every 5 seconds 
                self.accel = (np.random.rand(2) - 0.5) * 20 #changing the acceleration (randomly)after being in a certain direction for a bit then changing its range from [0,1] to [-0.5,0.5], then to [-10,10]
            self.vel += self.accel * dt #(v=u+at)
            self.pos += self.vel * dt #(x=vdt+x)

#Radar range eqn and mulitple targets 

def calculate_rre_max_range(radar_params):
    #max range calculation
    power_tx = radar_params['tx_power']
    gain_db = radar_params['antenna_gain']
    wavelength = radar_params['wavelength']
    rcs = radar_params['rcs']
    snr_min_db = radar_params['snr_min_db']
    
    system_temp, bandwidth, noise_figure_db = 290.0, 1.0e6, 3.0
    #converting to linear scale as those parameters are used in RRE
    gain_linear = 10**(gain_db / 10.0)
    snr_min_linear = 10**(snr_min_db / 10.0)
    noise_figure_linear = 10**(noise_figure_db / 10.0)
    
    noise_power = K_B * system_temp * bandwidth * noise_figure_linear
    s_min = noise_power * snr_min_linear
    #rre=num/denom
    numerator = power_tx * gain_linear**2 * wavelength**2 * rcs
    denominator = (4 * np.pi)**3 * s_min
    #returning the rmax after using rre
    return (numerator / denominator)**0.25 if denominator else np.inf

def moving_target_indicator(target):
    range_mag = np.linalg.norm(target.pos) #calculates the cartesian distance from the radar
    if range_mag == 0: return 0, False #|r|=0, target is at radar ie. origin
    radial_velocity = np.dot(target.vel, target.pos)/range_mag #(v radial =v.r/|r|) projection formula 
    is_moving = abs(radial_velocity) > 1.0 #moving if absolute velocity >1m/s
    return radial_velocity, is_moving

#GUI
class RadarApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Radar Simulation") #window title 
        self.geometry("1400x1000") #window size

        self.is_running = False
        self.beam_angle = 0.0
        self.time_step = 0.1
        self.sim_time = 0.0

        self._setup_gui()
        self.reset_simulation()

    def _setup_gui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        #Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        sim_controls = ttk.LabelFrame(left_panel, text="Simulation Controls", padding=10)
        sim_controls.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(sim_controls, text="Start", command=self.toggle_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_controls, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=8)

        target_gen_frame = ttk.LabelFrame(left_panel, text="Target Generation", padding=10)
        target_gen_frame.pack(fill=tk.X, pady=10)
        
        self.num_targets_var = tk.IntVar(value=8)
        self.num_clutter_var = tk.IntVar(value=5)
        self._create_entry(target_gen_frame, "Aircraft Targets:", self.num_targets_var, 0)
        self._create_entry(target_gen_frame, "Stationary Targets:", self.num_clutter_var, 1)

        radar_params_frame = ttk.LabelFrame(left_panel, text="Radar Parameters", padding=10)
        radar_params_frame.pack(fill=tk.X, pady=10)

        self.tx_power_var = tk.DoubleVar(value=1.5e6)
        self.antenna_gain_var = tk.DoubleVar(value=40)
        self.frequency_var = tk.DoubleVar(value=1.3e9)
        self.snr_min_db_var = tk.DoubleVar(value=14.0)
        self.mti_enabled_var = tk.BooleanVar(value=True)

        self._create_entry(radar_params_frame, "Transmission Power(W):", self.tx_power_var, 0)
        self._create_entry(radar_params_frame, "Antenna Gain(dB):", self.antenna_gain_var, 1)
        self._create_entry(radar_params_frame, "Frequency(Hz):", self.frequency_var, 2)
        self._create_entry(radar_params_frame, "Min. Detection SNR(dB):", self.snr_min_db_var, 3)
        #ttk.Checkbutton(radar_params_frame, text="Enable MTI", variable=self.mti_enabled_var).grid(row=4, columnspan=2, pady=5)
        
        log_frame = ttk.LabelFrame(left_panel, text="Detection Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=45, width=50, font=("Calibri", 12), bg="black", fg="lime")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        #PPI display [Plane Position Indicator] - radar display
        self.fig_ppi, self.ax_ppi = plt.subplots(figsize=(8, 8))
        self.canvas_ppi = FigureCanvasTkAgg(self.fig_ppi, master=main_frame)
        self.canvas_ppi.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _create_entry(self, parent, text, var, row):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=15).grid(row=row, column=1, sticky="w")

    def toggle_simulation(self):
        self.is_running = not self.is_running
        self.start_button.config(text="Pause" if self.is_running else "Start") #start and stop buttons
        if self.is_running:
            self.run_simulation_step()

    def reset_simulation(self): #reset button
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
        for i in range(num_targets): #how many targets
            angle, r = np.random.rand() * 2 * np.pi, np.sqrt(np.random.rand()) * max_range * 0.95 #angles will be from 0 to 2pi 
            x, y = r * np.cos(angle), r * np.sin(angle) #polar to cartesian conversion
            vx, vy = (np.random.rand(2) - 0.5) * 600 #assigning velocity (0,1) to (-0.5,0.5) then *600 to bring in range of [-300,300]m/s realistic acceleration of a target
            rcs = 10**(np.random.uniform(-1, 1.5)) #some small/big targets between 0.1 - 31.6 m^2
            targets.append(Target(f"T{i}", x, y, vx, vy, rcs)) #assigning a label to the target ie. T0, T1
        for i in range(num_clutter): #mountain/sttionary
            angle, r = np.random.rand() * 2 * np.pi, np.random.uniform(0.1, 0.9) * max_range #angle= anywhere 360 degrees, range=[0.1-300]km
            x, y = r * np.cos(angle), r * np.sin(angle) #position using physics formula 
            rcs = np.random.uniform(50, 500) #equal probability os having rcs between 50-500, ie. small/big mountain
            targets.append(Target(f"C{i}", x, y, 0, 0, rcs, is_clutter=True)) #velocity of clutter=0 and C0, C1 
        return targets

    def run_simulation_step(self):
        if not self.is_running: return #starting the simulation 
        
        self.sim_time += self.time_step
        self.beam_angle = (self.beam_angle + 2.0) % 360 #2 degree movement in each frame 
        
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
        self.ax_ppi.text(0,-self.display_range *1.1,"PPI Display",color='black',fontsize=14,ha='center',va='top')
        self.ax_ppi.set_xlim(-self.display_range, self.display_range)
        self.ax_ppi.set_ylim(-self.display_range, self.display_range)

        for r in np.linspace(0.25, 1.0, 4) * self.display_range: #where all to put the concentric circles, at these intervals
            self.ax_ppi.add_patch(Circle((0, 0), r, fill=False, edgecolor='lime', linestyle='--', alpha=0.5)) #concentric circles
            self.ax_ppi.text(5000, r + 5000, f'{r/1000:.0f} km', color='lime', fontsize=8) #radius of circle written
        bx, by = self.display_range * np.cos(np.deg2rad(self.beam_angle)), self.display_range * np.sin(np.deg2rad(self.beam_angle)) #beam's x and y coordinates
        self.ax_ppi.plot([0, bx], [0, by], color='lime', linewidth=1) #beam of the radar
        radar_params = {'tx_power': self.tx_power_var.get(), 'antenna_gain': self.antenna_gain_var.get(),
                        'wavelength': C / self.frequency_var.get(), 'rcs': 1.0, 'snr_min_db': self.snr_min_db_var.get()} #collecting all the parameters to display
        
        log_string = f"Time: {self.sim_time:.1f}s | Beam: {self.beam_angle:.0f}°\n"
        log_string += "-------------------------------------------------------------------------------\n"
        log_string += "ID   | Status   | Range(km) | R_max(km) | Speed(m/s) | ETA\n"
        log_string += "-------------------------------------------------------------------------------\n"

        for target in self.targets:
            target_angle = (np.rad2deg(np.arctan2(target.pos[1], target.pos[0])) + 360) % 360 
            #first we get x and y position in radians, then it's changed to degrees, then finally to range [0,2pi]
            beam_diff = abs((self.beam_angle - target_angle + 180) % 360 - 180) #returns the shortest path ie. angular distance between radar beam and target
            if not (beam_diff < 5.0): continue #after 5 degree difference exceeds, the target will disappear from the radar's display

            radar_params['rcs'] = target.rcs
            target_max_range = calculate_rre_max_range(radar_params) #using rre we get max range
            target_current_range = np.linalg.norm(target.pos) #current range of target
            is_detected = target_current_range <= target_max_range #checks if target is in detectable range
            if is_detected: #if yes it shows as moving and detects range and all other requirements and shows in the detection log
                radial_velocity, is_moving = moving_target_indicator(target)
                speed_ms = np.linalg.norm(target.vel)
                eta_str = "--:--"
                #if no MTI is selected          
                status = "NO MTI"
                if self.mti_enabled_var.get():
                    if is_moving: #rad vel < 0 means moving
                        color = 'red'
                        self.ax_ppi.plot(target.pos[0], target.pos[1], 'o', color=color, markersize=7)
                        vel_line_end = target.pos + target.vel * 50 #tail of the target
                        self.ax_ppi.plot([target.pos[0], vel_line_end[0]], [target.pos[1], vel_line_end[1]], color=color, linewidth=1.5) #how are the coordinates of this tail moving?
                        status = "MOVING"                    
                        if radial_velocity < 0: #eta only for targets moving towards the radar
                            eta_seconds = target_current_range / abs(radial_velocity)
                            minutes = int(eta_seconds / 60)
                            seconds = int(eta_seconds % 60)
                            eta_str = f"{minutes:02d}:{seconds:02d}"
                    else:
                        self.ax_ppi.plot(target.pos[0], target.pos[1], '^', color='white', markersize=8)
                        status = "STATIONARY"
                else:
                    self.ax_ppi.plot(target.pos[0], target.pos[1], 'x', color='red', markersize=8) #detection log format and final output
                log_string += f"{target.id:<4} | {status:<8} | {target_current_range/1000:<9.1f} | {target_max_range/1000:<9.1f} | {speed_ms:<10.1f} | {eta_str}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, log_string)
        self.log_text.config(state=tk.DISABLED)
        self.canvas_ppi.draw()

if __name__ == "__main__":
    app = RadarApp()
    app.mainloop()

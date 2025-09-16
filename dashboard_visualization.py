import json
import tkinter as tk
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Main Dashboard Application Class ---
class SimulationDashboard:
    def __init__(self, master):
        self.master = master
        master.title("Rocket Intercept Dashboard")
        master.configure(bg='black')

        # --- Load and Process Data ---
        self.load_data()

        # If data loading failed, stop here
        if not self.data:
            tk.Label(master, text="Error: impact_trajectory.jsonl not found or empty.\nPlease run the simulation first.",
                     font=font.Font(family="Helvetica", size=14), fg="red", bg="black").pack(padx=50, pady=50)
            return

        # --- Create GUI Elements ---
        self.create_plot_frame()
        self.create_data_frame()

        # --- MODIFIED SECTION: Start Animation with a small delay ---
        self.frame_index = 0
        # Schedule the first update to run after 100ms, allowing the window to draw first
        self.master.after(100, self.update_animation)

    def load_data(self):
        try:
            with open('impact_trajectory.jsonl', 'r') as f:
                self.data = [json.loads(line) for line in f]
            # Ensure data is not empty
            if not self.data:
                raise FileNotFoundError
        except (FileNotFoundError, json.JSONDecodeError):
            print("Error: Could not find or read 'impact_trajectory.jsonl'. Run the simulation first.")
            self.data = []
            return

        self.rocket_path = np.array([d['rocket_pos'] for d in self.data])
        self.comet_path = np.array([d['comet_pos'] for d in self.data])
        self.time_stamps = [d['time_days'] for d in self.data]
        self.earth_pos = self.data[0]['earth_pos']
        self.sun_pos = np.array([0,0])

    def create_plot_frame(self):
        plot_frame = tk.Frame(self.master, bg='black')
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig, self.ax = plt.subplots(figsize=(8, 8))
        fig.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal')

        # Plot static elements
        self.ax.plot(self.sun_pos[0], self.sun_pos[1], 'o', markersize=15, color='yellow', label='Sun')
        self.ax.plot(self.earth_pos[0], self.earth_pos[1], 'o', markersize=8, color='blue', label='Earth')
        self.ax.plot(self.comet_path[:, 0], self.comet_path[:, 1], '--', color='cyan', alpha=0.5, label='Comet Path')
        self.ax.plot(self.rocket_path[:, 0], self.rocket_path[:, 1], '--', color='red', alpha=0.5, label='Rocket Path')

        # Initialize animated elements
        self.comet_head, = self.ax.plot([], [], 'o', markersize=10, color='cyan')
        self.rocket_head, = self.ax.plot([], [], '>', markersize=10, color='red')

        # Style plot
        self.ax.set_title('Live Trajectory', color='white')
        self.ax.set_xlabel('X Coordinate (AU)', color='white')
        self.ax.set_ylabel('Y Coordinate (AU)', color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
        self.ax.legend(facecolor='grey', labelcolor='white')
        self.ax.grid(True, linestyle=':', color='gray', alpha=0.5)

        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_data_frame(self):
        data_frame = tk.Frame(self.master, bg='black', padx=20, pady=20)
        data_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        data_font = font.Font(family="Courier", size=12)

        self.day_var = tk.StringVar(value="Day: -")
        self.dist_var = tk.StringVar(value="Distance: - km")
        self.rocket_pos_var = tk.StringVar(value="Rocket Pos: (-, -) AU")
        self.comet_pos_var = tk.StringVar(value="Comet Pos: (-, -) AU")

        tk.Label(data_frame, text="LIVE DATA", font=title_font, fg="cyan", bg="black").pack(pady=10)
        tk.Label(data_frame, textvariable=self.day_var, font=data_font, fg="white", bg="black", justify=tk.LEFT).pack(anchor='w', pady=5)
        tk.Label(data_frame, textvariable=self.dist_var, font=data_font, fg="white", bg="black", justify=tk.LEFT).pack(anchor='w', pady=5)
        tk.Label(data_frame, textvariable=self.rocket_pos_var, font=data_font, fg="white", bg="black", justify=tk.LEFT).pack(anchor='w', pady=5)
        tk.Label(data_frame, textvariable=self.comet_pos_var, font=data_font, fg="white", bg="black", justify=tk.LEFT).pack(anchor='w', pady=5)

    def update_animation(self):
        if self.frame_index >= len(self.data):
            # Optional: Display final status
            final_dist_km = np.linalg.norm(self.rocket_path[-1] - self.comet_path[-1]) * 149597870.7
            if final_dist_km < 15000:
                self.dist_var.set(f"Distance: IMPACT!")
            else:
                 self.dist_var.set(f"Distance: MISSION END")
            return

        current_rocket_pos = self.rocket_path[self.frame_index]
        current_comet_pos = self.comet_path[self.frame_index]
        self.rocket_head.set_data(current_rocket_pos[0], current_rocket_pos[1])
        self.comet_head.set_data(current_comet_pos[0], current_comet_pos[1])
        self.canvas.draw()
        
        day = self.time_stamps[self.frame_index]
        distance_km = np.linalg.norm(current_rocket_pos - current_comet_pos) * 149597870.7
        
        self.day_var.set(f"Day       : {day:.1f}")
        self.dist_var.set(f"Distance  : {distance_km:,.0f} km")
        self.rocket_pos_var.set(f"Rocket Pos: ({current_rocket_pos[0]:.2f}, {current_rocket_pos[1]:.2f}) AU")
        self.comet_pos_var.set(f"Comet Pos : ({current_comet_pos[0]:.2f}, {current_comet_pos[1]:.2f}) AU")

        self.frame_index += 1
        self.master.after(50, self.update_animation)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationDashboard(root)
    root.mainloop()
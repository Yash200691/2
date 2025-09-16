import numpy as np
import torch
import joblib
import json
import os

# --- Simulation Parameters ---
TIME_STEP_DAYS = 0.5
INTERCEPT_TIME_DAYS = 50
# NEW: Parameters for orbital insertion
ORBIT_INSERTION_THRESHOLD_AU = 0.08 # Approx 12 million km, the distance to trigger the burn
ORBITAL_RADIUS_AU = 0.00015        # The desired stable orbit radius (~22,000 km)

# --- Physical Constants ---
G = 0.0002959122082855911  # Gravitational constant in AU^3 / (solar_mass * day^2)
M_SUN = 1.0
# NEW: Estimated mass of the comet (based on a large asteroid like Vesta)
# This is tiny but crucial for calculating the orbit's physics.
M_COMET = 1.3e-10 # In solar masses
AU_KM = 149597870.7

# --- Helper Functions and Model Class (remain the same) ---
def convert_single_to_cartesian_pos(ra_deg, dec_deg, delta):
    ra_rad, dec_rad = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    x = delta * np.cos(dec_rad) * np.cos(ra_rad)
    y = delta * np.cos(dec_rad) * np.sin(ra_rad)
    return x, y

class CometLSTM(torch.nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# --- Main Simulation Logic ---
if __name__ == "__main__":
    print("Loading comet prediction model and data...")
    model = CometLSTM(); model.load_state_dict(torch.load('comet_model.pth')); model.eval()
    scaler = joblib.load('data_scaler.gz')
    last_sequence_scaled = np.load('last_sequence.npy')

    output_filename = 'impact_trajectory.jsonl'
    if os.path.exists(output_filename):
        os.remove(output_filename)

    earth_pos = np.array([1.0, 0.0])
    rocket_pos = np.copy(earth_pos)

    print(f"Predicting comet's position in {INTERCEPT_TIME_DAYS} days to calculate intercept...")
    temp_sequence = np.copy(last_sequence_scaled)
    num_predictions = int(INTERCEPT_TIME_DAYS / (4/24))
    for _ in range(num_predictions):
        with torch.no_grad():
            input_tensor = torch.from_numpy(temp_sequence).float()
            prediction_result = model(input_tensor).cpu().numpy()
            comet_future_pos_scaled = prediction_result.reshape(1, 4)
            temp_sequence = np.append(temp_sequence[1:], comet_future_pos_scaled, axis=0)
    comet_intercept_data = scaler.inverse_transform(comet_future_pos_scaled)[0]
    comet_intercept_pos = np.array(convert_single_to_cartesian_pos(comet_intercept_data[0], comet_intercept_data[1], comet_intercept_data[2]))
    print(f"Target intercept point (AU): {comet_intercept_pos}")

    direction_vector = comet_intercept_pos - rocket_pos
    rocket_vel = direction_vector / INTERCEPT_TIME_DAYS
    rocket_vel *= 1.15
    print(f"Calculated initial rocket velocity (AU/day): {rocket_vel}")

    current_time_days = 0
    comet_sequence = np.copy(last_sequence_scaled)
    
    # NEW: State variables for the new mission
    burn_executed = False
    is_orbiting = False
    comet_pos_previous = None # Needed to calculate comet's velocity

    print("\n--- Starting Simulation: Mission is Orbital Insertion ---")
    try:
        while True:
            # Predict Comet Position
            with torch.no_grad():
                input_tensor = torch.from_numpy(comet_sequence).float()
                prediction_result = model(input_tensor).cpu().numpy()
                next_comet_step_scaled = prediction_result.reshape(1, 4)
                comet_sequence = np.append(comet_sequence[1:], next_comet_step_scaled, axis=0)
            comet_data_unscaled = scaler.inverse_transform(next_comet_step_scaled)[0]
            comet_pos = np.array(convert_single_to_cartesian_pos(comet_data_unscaled[0], comet_data_unscaled[1], comet_data_unscaled[2]))

            if comet_pos_previous is None: comet_pos_previous = comet_pos

            distance_rocket_comet = np.linalg.norm(rocket_pos - comet_pos)
            
            # --- NEW: Check for Orbital Insertion Burn condition ---
            if not burn_executed and distance_rocket_comet < ORBIT_INSERTION_THRESHOLD_AU:
                print(f"\n*** PROXIMITY ALERT | Distance: {distance_rocket_comet*AU_KM:,.0f} km. EXECUTING ORBITAL INSERTION BURN! ***\n")
                
                # 1. Calculate comet's velocity
                comet_vel = (comet_pos - comet_pos_previous) / TIME_STEP_DAYS
                
                # 2. Calculate required velocity for a circular orbit
                to_comet_vec = comet_pos - rocket_pos
                orbit_dir_vec = np.array([-to_comet_vec[1], to_comet_vec[0]]) # Perpendicular vector
                orbit_dir_vec /= np.linalg.norm(orbit_dir_vec) # Normalize
                
                # 3. Calculate orbital speed (Vis-viva equation)
                orbit_speed = np.sqrt(G * M_COMET / ORBITAL_RADIUS_AU)
                
                # 4. The rocket's new velocity is the comet's velocity plus the orbital velocity
                rocket_vel = comet_vel + (orbit_dir_vec * orbit_speed)
                
                burn_executed = True
                is_orbiting = True

            # --- MODIFIED: Physics Calculation ---
            if is_orbiting:
                # After the burn, the rocket is primarily affected by the comet's gravity
                dist_vec = rocket_pos - comet_pos
                grav_force_mag = (G * M_COMET) / (np.linalg.norm(dist_vec)**2)
                acceleration = grav_force_mag * (-dist_vec / np.linalg.norm(dist_vec))
            else:
                # Before the burn, the rocket is primarily affected by the Sun's gravity
                dist_vec = rocket_pos
                grav_force_mag = (G * M_SUN) / (np.linalg.norm(dist_vec)**2)
                acceleration = grav_force_mag * (-dist_vec / np.linalg.norm(dist_vec))
            
            rocket_vel += acceleration * TIME_STEP_DAYS
            rocket_pos += rocket_vel * TIME_STEP_DAYS

            print(f"Day: {current_time_days:5.1f} | Rocket-Comet Distance: {distance_rocket_comet*AU_KM:,.0f} km")

            comet_pos_previous = comet_pos
            
            log_entry = {
                "time_days": float(current_time_days), "rocket_pos": [float(coord) for coord in rocket_pos],
                "comet_pos": [float(coord) for coord in comet_pos], "earth_pos": [float(coord) for coord in earth_pos]
            }
            with open(output_filename, 'a') as f: f.write(json.dumps(log_entry) + '\n')

            current_time_days += TIME_STEP_DAYS
            if current_time_days > 150: # Increased timeout for longer orbit visualization
                print("\n--- Simulation time limit reached. ---")
                break
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
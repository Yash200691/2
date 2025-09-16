import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
from datetime import datetime, timedelta # MODIFIED: Import timedelta
import os
import json

# --- Helper Functions (remain the same) ---
def ra_to_degrees(ra_str):
    try:
        h, m, s = map(float, str(ra_str).split())
        return (h + m/60 + s/3600) * 15
    except (ValueError, TypeError): return np.nan

def dec_to_degrees(dec_str):
    try:
        dec_str = str(dec_str).strip()
        sign = -1 if dec_str.startswith('-') else 1
        dec_str = dec_str.replace('+', '').replace('-', '')
        d, m, s = map(float, dec_str.split())
        return sign * (d + m/60 + s/3600)
    except (ValueError, TypeError): return np.nan

def convert_single_to_cartesian_pos(ra_deg, dec_deg, delta):
    ra_rad, dec_rad = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    x = delta * np.cos(dec_rad) * np.cos(ra_rad)
    y = delta * np.cos(dec_rad) * np.sin(ra_rad)
    return x, y

# --- PyTorch Model Definition (remains the same) ---
class CometLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=50, num_layers=2, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

# --- Main Live Prediction Loop ---
if __name__ == "__main__":
    MODEL_FILE, SCALER_FILE, DATA_FILE = 'comet_predictor.pth', 'scaler.pkl', 'extracted_ephemeris.csv'
    TIME_STEPS, TIME_STEP_HOURS = 50, 4

    print("Step 1: Initializing live predictor...")
    
    required_files = [MODEL_FILE, SCALER_FILE, DATA_FILE]
    if not all(os.path.exists(f) for f in required_files):
        print(f"Error: Missing required files.")
    else:
        # Load model, scaler, and initial data
        model = CometLSTM(); model.load_state_dict(torch.load(MODEL_FILE)); model.eval()
        scaler = joblib.load(SCALER_FILE)
        df = pd.read_csv(DATA_FILE)
        
        # MODIFIED: Get the last known date from the data file to start our simulation clock
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        current_comet_time = df['datetime'].iloc[-1]

        df['RA_deg'] = df['RA'].apply(ra_to_degrees); df['DEC_deg'] = df['DEC'].apply(dec_to_degrees)
        features = ['RA_deg', 'DEC_deg', 'delta', 'Sky_motion']
        df_clean = df[features].dropna()
        scaled_data = scaler.transform(df_clean.values)
        last_sequence = scaled_data[-TIME_STEPS:]
        previous_x, previous_y = None, None

        output_filename = 'live_comet_predictions.jsonl'
        print(f"Initialization complete. Appending 8-hour forecast data to '{output_filename}' (press Ctrl+C to stop).")
        print(f"Simulation starting from base date: {current_comet_time.strftime('%Y-%m-%d %H:%M')}")
        print("-" * 70)

        try:
            while True:
                # --- Lookahead prediction logic remains the same ---
                lookahead_sequence = last_sequence.copy()
                for _ in range(2): # Predict 2 steps (8 hours) into the future
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(lookahead_sequence).float().view(1, TIME_STEPS, 4)
                        next_step_scaled = model(input_tensor).cpu().numpy()
                    lookahead_sequence = np.append(lookahead_sequence[1:], next_step_scaled, axis=0)
                
                forecast_8hr_scaled = next_step_scaled
                forecast_8hr_real = scaler.inverse_transform(forecast_8hr_scaled)
                ra, dec, delta, motion = forecast_8hr_real[0]

                # MODIFIED: Calculate the date of the forecast
                forecast_comet_time = current_comet_time + timedelta(hours=8)

                # Calculation and saving logic now uses the 8-hour forecast data
                current_x, current_y = convert_single_to_cartesian_pos(ra, dec, delta)
                if previous_x is not None:
                    vx, vy = (current_x - previous_x) / TIME_STEP_HOURS, (current_y - previous_y) / TIME_STEP_HOURS
                else:
                    vx, vy = 0, 0
                
                prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                distance_in_km = delta * 149597870.7

                # MODIFIED: Update the print statement to include the comet's date
                print(f"[{prediction_timestamp}] FORECAST FOR [{forecast_comet_time.strftime('%Y-%m-%d %H:%M')}] -> RA: {ra:.4f}, Dec: {dec:.4f}")
                print(f"    Cartesian Coords -> x: {current_x:+.4f}, y: {current_y:+.4f}, vx: {vx:+.4f}, vy: {vy:+.4f}")
                print(f"    Distance from Earth -> {distance_in_km:,.0f} km\n")

                # MODIFIED: Add the comet's date to the JSON data
                data_point = {
                    "prediction_timestamp": prediction_timestamp,
                    "forecast_date": forecast_comet_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "ra_deg": float(ra), "dec_deg": float(dec), "delta_au": float(delta),
                    "x_au": float(current_x), "y_au": float(current_y), "vx_au_per_hr": float(vx), "vy_au_per_hr": float(vy),
                    "distance_from_earth_km": float(distance_in_km)
                }
                with open(output_filename, 'a') as file:
                    file.write(json.dumps(data_point) + '\n')

                # --- Advance the simulation's base time and state by ONE step ---
                current_comet_time += timedelta(hours=TIME_STEP_HOURS)
                with torch.no_grad():
                    input_tensor = torch.from_numpy(last_sequence).float().view(1, TIME_STEPS, 4)
                    next_actual_step_scaled = model(input_tensor).cpu().numpy()
                last_sequence = np.append(last_sequence[1:], next_actual_step_scaled, axis=0)
                
                previous_x, previous_y = current_x, current_y
                time.sleep(5)

        except KeyboardInterrupt:
            print(f"\nLive prediction stopped. Data is saved in '{output_filename}'.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
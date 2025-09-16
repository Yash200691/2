import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
INPUT_CSV_FILE = 'extracted_ephemeris.csv'
TIME_STEPS = 10 # How many past data points the model uses to predict the next one
EPOCHS = 200    # How many times the model will train over the entire dataset

# --- Helper Functions (copied from comet.py for consistency) ---
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

# --- Data Loading and Preprocessing ---
print(f"Loading and processing data from '{INPUT_CSV_FILE}'...")
df = pd.read_csv(INPUT_CSV_FILE)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%b-%d %H:%M')
df['RA_deg'] = df['RA'].apply(ra_to_degrees)
df['DEC_deg'] = df['DEC'].apply(dec_to_degrees)
df = df.dropna(subset=['RA_deg', 'DEC_deg', 'delta', 'deldot']).sort_values('datetime')

# Select features for the model
features = ['RA_deg', 'DEC_deg', 'delta', 'deldot']
data = df[features].values

# --- Scale the Data ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Data scaled successfully.")

# --- Create Sequences for LSTM Model ---
def create_sequences(input_data, time_steps):
    X, y = [], []
    for i in range(len(input_data) - time_steps):
        X.append(input_data[i:(i + time_steps)])
        y.append(input_data[i + time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(data_scaled, TIME_STEPS)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
print(f"Created {len(X_train)} training sequences.")

# --- Define the LSTM Model ---
class CometLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# --- Training the Model ---
model = CometLSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("\n--- Starting Model Training (This may take several minutes) ---")

for i in range(EPOCHS):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    # MODIFIED: Print status more frequently
    if (i+1) % 10 == 0:
        print(f'Epoch [{i+1}/{EPOCHS}] | Loss: {single_loss.item():.8f}')

print("--- Training Complete ---")

# --- Save the Essential Files ---
torch.save(model.state_dict(), 'comet_model.pth')
print("✅ Model saved as 'comet_model.pth'")
joblib.dump(scaler, 'data_scaler.gz')
print("✅ Data scaler saved as 'data_scaler.gz'")
last_sequence = data_scaled[-TIME_STEPS:]
np.save('last_sequence.npy', last_sequence)
print("✅ Last sequence for live prediction saved as 'last_sequence.npy'")
print("\nAll necessary files have been generated. You can now run the simulation.")
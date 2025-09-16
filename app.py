from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import joblib
import numpy as np
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# --- Model 1: deldot (10 steps) ---
class CometLSTM_Deldot(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model_deldot = CometLSTM_Deldot()
model_deldot.load_state_dict(torch.load('comet_model.pth', map_location='cpu'))
model_deldot.eval()
scaler_deldot = joblib.load('data_scaler.gz')
TIME_STEPS_DELDOT = 10

# --- Model 2: Sky_motion (50 steps) ---
class CometLSTM_SkyMotion(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=50, num_layers=2, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

model_skymotion = CometLSTM_SkyMotion()
model_skymotion.load_state_dict(torch.load('comet_predictor.pth', map_location='cpu'))
model_skymotion.eval()
scaler_skymotion = joblib.load('scaler.pkl')
TIME_STEPS_SKYMOTION = 50

@app.route('/predict_deldot', methods=['POST'])
def predict_deldot():
    data = request.json
    sequence = data.get('sequence')
    if not sequence or len(sequence) != TIME_STEPS_DELDOT or len(sequence[0]) != 4:
        return jsonify({"error": f"Input must be a list of {TIME_STEPS_DELDOT} items, each with 4 features (RA_deg, DEC_deg, delta, deldot)."}), 400

    sequence_np = np.array(sequence)
    scaled_input = scaler_deldot.transform(sequence_np)
    input_tensor = torch.from_numpy(scaled_input).float()

    with torch.no_grad():
        model_deldot.hidden_cell = (torch.zeros(1, 1, model_deldot.hidden_layer_size),
                                    torch.zeros(1, 1, model_deldot.hidden_layer_size))
        prediction_scaled = model_deldot(input_tensor).cpu().numpy().reshape(1, -1)

    predicted_values = scaler_deldot.inverse_transform(prediction_scaled)[0]
    response = {
        'RA_deg': predicted_values[0],
        'DEC_deg': predicted_values[1],
        'delta': predicted_values[2],
        'deldot': predicted_values[3]
    }
    return jsonify(response)

@app.route('/predict_skymotion', methods=['POST'])
def predict_skymotion():
    data = request.json
    sequence = data.get('sequence')
    if not sequence or len(sequence) != TIME_STEPS_SKYMOTION or len(sequence[0]) != 4:
        return jsonify({"error": f"Input must be a list of {TIME_STEPS_SKYMOTION} items, each with 4 features (RA_deg, DEC_deg, delta, Sky_motion)."}), 400

    sequence_np = np.array(sequence)
    scaled_input = scaler_skymotion.transform(sequence_np)
    input_tensor = torch.from_numpy(scaled_input).float().unsqueeze(0)  # batch_first

    with torch.no_grad():
        prediction_scaled = model_skymotion(input_tensor).cpu().numpy().reshape(1, -1)

    predicted_values = scaler_skymotion.inverse_transform(prediction_scaled)[0]
    response = {
        'RA_deg': predicted_values[0],
        'DEC_deg': predicted_values[1],
        'delta': predicted_values[2],
        'Sky_motion': predicted_values[3]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
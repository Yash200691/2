from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import joblib
import numpy as np
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# --- Model: deldot (10 steps) ---
class CometLSTM(nn.Module):
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

model = CometLSTM()
model.load_state_dict(torch.load('comet_model.pth', map_location='cpu'))
model.eval()
scaler = joblib.load('data_scaler.gz')
TIME_STEPS = 10

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = data.get('sequence')
    if not sequence or len(sequence) != TIME_STEPS or len(sequence[0]) != 4:
        return jsonify({"error": f"Input must be a list of {TIME_STEPS} items, each with 4 features (RA_deg, DEC_deg, delta, deldot)."}), 400

    sequence_np = np.array(sequence)
    scaled_input = scaler.transform(sequence_np)
    input_tensor = torch.from_numpy(scaled_input).float()

    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        prediction_scaled = model(input_tensor).cpu().numpy().reshape(1, -1)

    predicted_values = scaler.inverse_transform(prediction_scaled)[0]
    response = {
        'RA_deg': predicted_values[0],
        'DEC_deg': predicted_values[1],
        'delta': predicted_values[2],
        'deldot': predicted_values[3]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
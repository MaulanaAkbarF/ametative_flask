from flask import Flask, request
import pickle
import numpy as np
import os

app = Flask(__name__)

current_dir = os.getcwd()

with open(os.path.join('Machine_Learning','Models','Model_SKLearn_XGB.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join('Machine_Learning', 'Models','Encoder_sex.pkl'), 'rb') as f:
    le_sex = pickle.load(f)

with open(os.path.join('Machine_Learning', 'Models','Encoder_chest_pain_type.pkl'), 'rb') as f:
    le_chest_pain_type = pickle.load(f)
    
with open(os.path.join('Machine_Learning', 'Models','Encoder_resting_ecg.pkl'), 'rb') as f:
    le_resting_ecg = pickle.load(f)

with open(os.path.join('Machine_Learning', 'Models','Encoder_exercise_angina.pkl'), 'rb') as f:
    le_exercise_angina = pickle.load(f)

with open(os.path.join('Machine_Learning', 'Models','Encoder_st_slope.pkl'), 'rb') as f:
    le_st_slope = pickle.load(f)

with open(os.path.join('Machine_Learning', 'Models','Encoder_hasil.pkl'), 'rb') as f:
    le_hasil = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    sex = 0.0 if data['sex'] == 'M' else 1.0
    chest_pain_type = 0.0 if data['chest_pain_type'] == 'ATA' else 1.0 if data['chest_pain_type'] == 'NAP' else 2.0 if data['chest_pain_type'] == 'ASY' else 3.0
    resting_bp = data['resting_bp']
    cholesterol = data['cholesterol']
    fasting_bs = data['fasting_bs']
    resting_ecg = 0.0 if data['resting_ecg'] == 'Normal' else 1.0 if data['resting_ecg'] == 'ST' else 2.0
    max_hr = data['max_hr']
    exercise_angina = 0.0 if data['exercise_angina'] == 'N' else 1.0
    oldpeak = data['oldpeak']
    st_slope = 0.0 if data['st_slope'] == 'Up' else 1.0

    X = np.array([age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]).reshape(1, -1)
    pred = model.predict(X)
    
    hasil = le_hasil.inverse_transform(pred)[0]
    
    return {'hasil': hasil.tolist()}

if __name__ == '__main__':
    app.run(debug=True)

# Contoh perintah yang bisa kamu coba di Command Prompt/terminal:

# Prediksi Hasil: 1
# curl -X POST -H "Content-Type: application/json" -d "{\"age\": 48, \"sex\": \"F\", \"chest_pain_type\": \"NAP\", \"resting_bp\": 159, \"cholesterol\": 181, \"fasting_bs\": 0, \"resting_ecg\": \"Normal\", \"max_hr\": 157, \"exercise_angina\": \"N\", \"oldpeak\": 1, \"st_slope\": \"Flat\"}" http://localhost:5000/predict

# Prediksi Hasil: 0
# curl -X POST -H "Content-Type: application/json" -d "{\"age\": 39, \"sex\": \"M\", \"chest_pain_type\": \"NAP\", \"resting_bp\": 121, \"cholesterol\": 338, \"fasting_bs\": 0, \"resting_ecg\": \"Normal\", \"max_hr\": 169, \"exercise_angina\": \"N\", \"oldpeak\": 0, \"st_slope\": \"Up\"}" http://localhost:5000/predict
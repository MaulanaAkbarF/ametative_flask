from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
import xgboost as xgb

app = Flask(__name__)

current_dir = os.getcwd()

# Load the model and encoder
with open(os.path.join('Machine_Learning', 'Models', 'Model_SKLearn_XGB_VARK.pkl'), 'rb') as f:
    model = pickle.load(f)

# Load the model
# with open(os.path.join('Models', 'ID3_Model.pkl'), 'rb') as f_id3:
#     model_id3 = pickle.load(f_id3)

with open(os.path.join('Machine_Learning', 'Models', 'Encoder_Result.pkl'), 'rb') as f:
    le_result = pickle.load(f)

# Initialize Firebase
cred = credentials.Certificate(os.path.join(current_dir, "firebase_credential.json"))
firebase_admin.initialize_app(cred)
firestore_db = firestore.client()

feature_names = ['Visual', 'Auditory', 'Read/Write', 'Kinesthetic']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_keys = ['Visual', 'Auditory', 'Read/Write', 'Kinesthetic', 'Email']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing required fields'}), 400
        
        visual = data['Visual']
        auditory = data['Auditory']
        read_write = data['Read/Write']
        kinesthetic = data['Kinesthetic']
        email = data['Email']
        
        X = np.array([visual, auditory, read_write, kinesthetic]).reshape(1, -1)
        X_dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        
        pred = model.predict(X_dmatrix)
        result = le_result.inverse_transform([int(pred[0])])[0]
        
        doc_ref = firestore_db.collection('Predict_XGBoost').document(email)
        doc_ref.set({
            'visual': visual,
            'auditory': auditory,
            'readwrite': read_write,
            'kinesthetic': kinesthetic,
            'result': result
        }, merge=True)
        
        return jsonify({'Result': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/predict_id3', methods=['POST'])
# def predict_id3():
#     data = request.json
#     km_class = data['km_class']
#     rm_class = data['rm_class']
#     sample = {"km_class": km_class, "rm_class": rm_class}
#     prediction = predict(model_id3, sample)
#     return jsonify({'Prediction': prediction})

# def predict(tree, sample):
#     if not isinstance(tree, dict):
#         return str(tree)
    
#     root = next(iter(tree))
#     if root in sample:
#         value = sample[root]
#         if value in tree[root]:
#             return predict(tree[root][value], sample)
    
#     return "1"

if __name__ == '__main__':
    app.run(debug=True)

# conth curl untuk endpoint /predict: curl -X POST -H "Content-Type: application/json" -d "{\"Email\": \"test1@gmail.com\", \"Visual\": 5, \"Auditory\": 3, \"Read/Write\": 4, \"Kinesthetic\": 6}" http://localhost:5000/predict
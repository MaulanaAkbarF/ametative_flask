from flask import Flask, request
import pickle
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore, db
from google.cloud import firestore as gcf
import threading

cred = credentials.Certificate("C:\All Programming Projects\Dart\myonion\pythonscript\sendDataDB\credentials\google_credential.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://bawangmerahapp1-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

firestore_db = firestore.client()
db = gcf.Client()

app = Flask(__name__)

current_dir = os.getcwd()

with open(os.path.join('Machine_Learning','Models','Model_SKLearn_XGB.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join('Machine_Learning', 'Models','Encoders_Result.pkl'), 'rb') as f:
    le_hasil = pickle.load(f)

def on_snapshot(col_snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'ADDED':
            print(f'New document: {change.document.id}')
            threading.Thread(target=predict, args=(change.document.id,)).start()

def predict(doc_id):
    doc_snapshot = firestore_db.collection('DeteksiJantung').document(doc_id).get()
    data = doc_snapshot.to_dict()

    X = np.array([
        data['age'], 
        data['gender'], 
        data['chestpain'], 
        data['bloodpressure'], 
        data['cholesterol'], 
        data['bloodsugar'], 
        data['restingecg'], 
        data['maxhr'], 
        data['angina'], 
        data['oldpeak'], 
        data['slope']
    ]).reshape(1, -1)

    pred = model.predict(X)
    
    hasil = le_hasil.inverse_transform(pred)[0]

    kodeHasil = hasil.tolist()

    if (kodeHasil == 0):
        doc_ref = db.collection('DeteksiJantung').document(doc_id)
        doc_ref.set({
            'result': 'Jantung Sehat'
        }, merge=True)
    elif (kodeHasil == 1):
        doc_ref = db.collection('DeteksiJantung').document(doc_id)
        doc_ref.set({
            'result': 'Terkena Peyakit Jantung'
        }, merge=True)

col_ref = firestore_db.collection('DeteksiJantung')

col_watch = col_ref.on_snapshot(on_snapshot)

if __name__ == '__main__':
    app.run(debug=True)
    while True:
        pass

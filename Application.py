import tkinter as tk
from tkinter import ttk
from keras.models import load_model
import requests
import numpy as np
import os

current_dir = os.getcwd()
model = load_model(os.path.join('Machine_Learning','Models','Model_TensorFlow_XGB.keras'))

def predict_flask():
    url = "http://localhost:5000/predict"  # Ganti URL sesuai dengan URL Flask Anda
    data = {
        "age": age_entry.get(),
        "sex": "F" if sex_combobox.get() == "Perempuan" else "M",
        "chest_pain_type": chest_pain_type_combobox.get(),
        "resting_bp": resting_bp_entry.get(),
        "cholesterol": cholesterol_entry.get(),
        "fasting_bs": "Ya" if fasting_bs_combobox.get() == "Ya" else "Tidak",
        "resting_ecg": resting_ecg_combobox.get(),
        "max_hr": max_hr_entry.get(),
        "exercise_angina": "Ya" if exercise_angina_combobox.get() == "Ya" else "Tidak",
        "oldpeak": oldpeak_entry.get(),
        "st_slope": st_slope_combobox.get()
    }
    try:
        response = requests.post(url, json=data)
        result = response.json()
        result_label.config(text="Hasil Prediksi: " + str(result['hasil'][0]))
    except requests.exceptions.RequestException as e:
        result_label.config(text="Error: " + str(e))

root = tk.Tk()
root.title("Deteksi Penyakit Jantung")

root.geometry("420x450")

def resize(event):
    for widget in root.winfo_children():
        widget.grid_configure(sticky="ew")

root.bind("<Configure>", resize)

def create_label(text, row):
    label = tk.Label(root, text=text, anchor="w")
    label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
    return label

row = 0
create_label("Umur", row)
age_entry = tk.Entry(root)
age_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Jenis Kelamin", row)
sex_combobox = ttk.Combobox(root, values=["Laki-Laki", "Perempuan"])
sex_combobox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Jenis Nyeri Dada", row)
chest_pain_type_combobox = ttk.Combobox(root, values=["ATA", "NAP", "ASY", "TA"])
chest_pain_type_combobox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Tekanan Darah", row)
resting_bp_entry = tk.Entry(root)
resting_bp_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Kolesterol", row)
cholesterol_entry = tk.Entry(root)
cholesterol_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Gula Darah", row)
fasting_bs_combobox = ttk.Combobox(root, values=["Ya", "Tidak"])
fasting_bs_combobox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("EKG saat kondisi istirahat", row)
resting_ecg_combobox = ttk.Combobox(root, values=["Normal", "Infark Miokard", "LVH"])
resting_ecg_combobox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Denyut Jantung Maksimum", row)
max_hr_entry = tk.Entry(root)
max_hr_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Nyeri Dada saat Olahraga", row)
exercise_angina_combobox = ttk.Combobox(root, values=["Ya", "Tidak"])
exercise_angina_combobox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Perubahan Segmen ST pada EKG saat olahraga", row)
oldpeak_entry = tk.Entry(root)
oldpeak_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
create_label("Kemiringan Segmen ST", row)
st_slope_combobox = ttk.Combobox(root, values=["Naik", "Datar"])
st_slope_combobox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

row += 1
predict_button = tk.Button(root, text="Prediksi", command=predict_flask)
predict_button.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

for widget in root.winfo_children():
    widget.grid_configure(sticky="ew")

result_label = tk.Label(root, text="")
result_label.grid(row=13, column=0, columnspan=2)

root.mainloop()
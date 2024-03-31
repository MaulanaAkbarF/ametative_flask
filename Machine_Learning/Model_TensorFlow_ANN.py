import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import os

current_dir = os.getcwd()
df = pd.read_excel(os.path.join(current_dir,  "Machine_Learning", "DataTraining.xlsx"))

le_sex = LabelEncoder()
le_chest_pain_type = LabelEncoder()
le_resting_ecg = LabelEncoder()
le_exercise_angina = LabelEncoder()
le_st_slope = LabelEncoder()
le_heart_disease = LabelEncoder()

sex_mapping = {
    "M": 0,
    "F": 1,
    "Lainnya": 2
}

chest_pain_type_mapping = {
    "ATA": 0,
    "NAP": 1,
    "ASY": 2,
    "TA": 3,
    "Lainnya": 4
}

resting_ecg_mapping = {
    "Normal": 0,
    "ST": 1,
    "LVH": 2,
    "Lainnya": 3
}

exercise_angina_mapping = {
    "N": 0,
    "Y": 1,
    "Lainnya": 2
}

st_slope_mapping = {
    "Up": 0,
    "Flat": 1,
    "Lainnya": 2
}

le_sex.classes_ = list(sex_mapping.keys())
le_chest_pain_type.classes_ = list(chest_pain_type_mapping.keys())
le_resting_ecg.classes_ = list(resting_ecg_mapping.keys())
le_exercise_angina.classes_ = list(exercise_angina_mapping.keys())
le_st_slope.classes_ = list(st_slope_mapping.keys())

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['ChestPainType'] = le_sex.fit_transform(df['ChestPainType'])
df['RestingECG'] = le_sex.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le_sex.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le_sex.fit_transform(df['ST_Slope'])
df['HeartDisease'] = le_heart_disease.fit_transform(df['HeartDisease'])

if not os.path.exists('Models'):
    os.makedirs('Models')

with open(os.path.join('Machine_Learning','Models', 'Encoder_sex.pkl'), 'wb') as f:
    pickle.dump(le_sex, f)

with open(os.path.join('Machine_Learning','Models', 'Encoder_chest_pain_type.pkl'), 'wb') as f:
    pickle.dump(le_chest_pain_type, f)
    
with open(os.path.join('Machine_Learning','Models', 'Encoder_resting_ecg.pkl'), 'wb') as f:
    pickle.dump(le_resting_ecg, f)
    
with open(os.path.join('Machine_Learning','Models', 'Encoder_exercise_angina.pkl'), 'wb') as f:
    pickle.dump(le_exercise_angina, f)
    
with open(os.path.join('Machine_Learning','Models', 'Encoder_st_slope.pkl'), 'wb') as f:
    pickle.dump(le_st_slope, f)

with open(os.path.join('Machine_Learning','Models', 'Encoder_hasil.pkl'), 'wb') as f:
    pickle.dump(le_heart_disease, f)

X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, input_dim=11, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=10, callbacks=[es])

_, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Accuracy: {accuracy*100}%')

model.save(os.path.join('Machine_Learning','Models', 'Model_TensorFlow_OptimizedMLP.keras'))

with open(os.path.join('Machine_Learning','Models', 'Scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
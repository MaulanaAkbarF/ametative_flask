import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
from scipy.special import softmax
import xgboost as xgb
from xgboost import plot_importance
import os
import matplotlib.pyplot as plt

print('Membaca Data Excel...')
current_dir = os.getcwd()
df = pd.read_excel(os.path.join(current_dir, "Machine_Learning", "Classification Data", "VARK_Count.xlsx"))
df = df.drop(df.columns[0], axis=1) # Hapus kolom Inisial

print('Encoding kolom target...')
le_result = LabelEncoder()
df['Result'] = le_result.fit_transform(df['Result'])

print('Menyimpan hasil encoder...')
if not os.path.exists(os.path.join('Machine_Learning', 'Models')): os.makedirs(os.path.join('Machine_Learning', 'Models'))
with open(os.path.join('Machine_Learning', 'Models', 'Encoder_Result.pkl'), 'wb') as f: pickle.dump(le_result, f)

print('Memisahkan data menjadi training dan testing...')
X = df.drop(columns=['Result']) 
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(
    # Memisahkan data menjadi data training dan data testing
    X, y,

    # Menentukan ukuran data testing sebesar 25% dari total data
    test_size=0.25,

    # Mengatur seed agar hasil pelatihan model dapat direproduksi sehingga pembagian data 
    # atau proses acak lainnya dalam model akan konsisten setiap kali kode dijalankan
    random_state=42,

    # Menentukan data target yang akan dipisahkan
    stratify=y
)

print(f'\nModel Algortima: Extreme Gradient Boosting (XGB)')
print('Versi XGBoost:', xgb.__version__)

# Konversi data ke DMatrix (format yang digunakan oleh xgboost.train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Library XGBoost secara implisit menggunakan fungsi objektif berikut:
# Obj = ∑(i=1 to n) L(^yi, yi) + ∑(i=1 to k) R(fi)
params = {
    # XGBoost akan menggunakan semua thread yang tersedia di sistem untuk mempercepat pelatihan.
    # Dengan mengatur nthread menjadi -1, XGBoost akan secara otomatis mendeteksi jumlah thread CPU yang tersedia dan menggunakannya.
    'nthread': -1,

    # Mengatur seed agar hasil pelatihan model dapat direproduksi sehingga pembagian data
    # atau proses acak lainnya dalam model akan konsisten setiap kali kode dijalankan
    'random_state': 42,
    
    # Dataset menggunakan klasifikasi multiclass, sehingga fungsi objektif yang digunakan adalah 'multi:softmax'.
    'objective': 'multi:softmax',
    'num_class': len(le_result.classes_),

    # Nilai yang lebih kecil membuat model belajar lebih lambat tetapi lebih stabil, sehingga dapat menghasilkan hasil yang lebih baik dengan risiko overfitting yang lebih rendah. 
    # Nilai yang lebih besar mempercepat pelatihan tetapi bisa menyebabkan model melewatkan solusi optimal.
    # Dengan menetapkan learning_rate menjadi 0.1, model akan belajar lebih lambat dan lebih stabil, sehingga mengurangi risiko overfitting.
    # Namun, mungkin perlu meningkatkan n_estimators untuk mengimbangi langkah yang lebih kecil ini.
    'learning_rate': 0.1,

    # Menentukan kedalaman maksimum dari setiap pohon keputusan (decision tree) dalam model XGBoost. 
    # Nilai 3 berarti pohon tidak akan memiliki lebih dari 5 level percabangan,
    # Untuk membatasi kompleksitas model untuk mencegah overfitting
    # Dataset kecil dengan hanya 4 fitur tidak memerlukan pohon yang terlalu dalam. Mengurangi max_depth ke 3 akan membatasi kompleksitas pohon, sehingga mengurangi risiko overfitting. 
    # Pohon yang terlalu dalam cenderung menangkap noise dalam data kecil.
    'max_depth': 3,

    # Menentukan proporsi sampel yang akan digunakan dalam setiap iterasi untuk menghindari overfitting.
    # Dengan mengatur subsample menjadi 0.8, model akan menggunakan 80% sampel secara acak dalam setiap iterasi.
    'subsample': 0.8,

    # Menentukan jumlah minimum bobot (atau jumlah sampel) yang diperlukan di setiap daun (leaf) pohon keputusan.
    # Untuk mencegah model membuat daun dengan sedikit sampel, yang dapat mengurangi overfitting.
    'min_child_weight': 4,

    # Kekuatan regularisasi L2 (Ridge regularization) yang diterapkan pada bobot model.
    # Nilai 3 menunjukkan tingkat penalti yang moderat terhadap kompleksitas model.
    # Regularisasi L2 membantu mencegah overfitting dengan menambahkan penalti pada bobot yang besar, 
    # sehingga model lebih stabil dan tidak terlalu sensitif terhadap fluktuasi kecil dalam data.
   'reg_lambda': 3,

    # Kekuatan regularisasi L1 (Lasso regularization) yang diterapkan pada bobot model. 
    # Nilai 2 menunjukkan penalti yang moderat.
    # Regularisasi L1 mendorong sparsity (bobot nol) dalam model, yang berarti beberapa fitur yang kurang penting 
    # bisa diabaikan. Ini membantu dalam seleksi fitur dan juga mencegah overfitting.
    'reg_alpha': 2,
}

# Melatih model
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    evals=evals,
    maximize=False,
    verbose_eval=True,

    # Dataset kecil tidak memerlukan terlalu banyak pohon.
    # Dengan num_boost_round yang lebih kecil, dapat mengurangi kompleksitas model.
    num_boost_round=50,

    # Menggunakan early_stopping_rounds untuk menghentikan pelatihan jika tidak ada peningkatan dalam 10 iterasi berturut-turut.
    # Ini membantu mencegah overfitting dengan menghentikan pelatihan lebih awal jika model tidak menunjukkan peningkatan pada data validasi.
    early_stopping_rounds=10
)

# Prediksi pada data testing
y_pred = model.predict(dtest)

  # Konversi ke integer untuk klasifikasi
y_pred = [int(pred) for pred in y_pred]

print('Evaluating model...')
print("Distribusi kelas di y_test:", y_test.value_counts())
print("Distribusi kelas di y_pred:", pd.Series(y_pred).value_counts())
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Precision: {precision * 100:.2f}%')

recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall * 100:.2f}%')

precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision * 100:.2f}%')

precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
print("Precision per kelas:", precision_per_class)

f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1 * 100:.2f}%')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le_result.classes_),
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    min_child_weight=3,
    reg_lambda=3,
    reg_alpha=2,
    random_state=42,
    nthread=-1
)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted'
}

cv_results = cross_validate(
    xgb_model,
    X,y,
    cv=5,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

print('K-Fold Cross-validation scores:')
print(f"Accuracy per fold: {[f'{score*100:.2f}%' for score in cv_results['test_accuracy']]}")
print(f"Average accuracy: {cv_results['test_accuracy'].mean() * 100:.2f}%")
print(f"Precision per fold: {[f'{score*100:.2f}%' for score in cv_results['test_precision']]}")
print(f"Average precision: {cv_results['test_precision'].mean() * 100:.2f}%")
print(f"Recall per fold: {[f'{score*100:.2f}%' for score in cv_results['test_recall']]}")
print(f"Average recall: {cv_results['test_recall'].mean() * 100:.2f}%")

print('Plotting ROC Curve...')
y_pred_raw = model.predict(dtest, output_margin=True)  # Skor mentah (logits)
y_pred_proba = softmax(y_pred_raw, axis=1)  # Konversi ke probabilitas dengan softmax

n_classes = len(le_result.classes_)
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {le_result.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost (Multiclass)')
plt.legend(loc="lower right")
plt.show()

# Plot feature importance
print('Plotting Feature Importance...')
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight')  # Bisa juga gunakan 'gain' atau 'cover'
plt.title('Feature Importance - XGBoost')
plt.show()

# Save the model
with open(os.path.join('Machine_Learning', 'Models', 'Model_SKLearn_XGB_VARK.pkl'), "wb") as f: pickle.dump(model, f)
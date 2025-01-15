import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Memuat dataset
file_path = r'Classification.csv'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File {file_path} tidak ditemukan. Pastikan file tersedia di lokasi yang benar.")
    exit()

# Periksa apakah kolom yang diperlukan ada di dataset
required_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
if not all(col in data.columns for col in required_columns):
    print(f"Dataset harus memiliki kolom: {required_columns}")
    exit()

# Encoding kolom kategori
label_encoder_sex = LabelEncoder()
data['Sex_Encoded'] = label_encoder_sex.fit_transform(data['Sex'])

label_encoder_bp = LabelEncoder()
data['BP_Encoded'] = label_encoder_bp.fit_transform(data['BP'])

label_encoder_cholesterol = LabelEncoder()
data['Cholesterol_Encoded'] = label_encoder_cholesterol.fit_transform(data['Cholesterol'])

label_encoder_drug = LabelEncoder()
data['Drug_Encoded'] = label_encoder_drug.fit_transform(data['Drug'])

# Pilih fitur dan target untuk prediksi kolesterol
X_chol = data[['Age', 'Sex_Encoded', 'BP_Encoded', 'Na_to_K']]
y_chol = data['Cholesterol_Encoded']

# Normalisasi data
scaler = StandardScaler()
X_chol_scaled = scaler.fit_transform(X_chol)

# Split data menjadi train dan test untuk prediksi kolesterol
X_chol_train, X_chol_test, y_chol_train, y_chol_test = train_test_split(X_chol_scaled, y_chol, test_size=0.2, random_state=42, stratify=y_chol)

# Model Logistic Regression untuk prediksi kolesterol
chol_model = LogisticRegression(random_state=42)

# Cross-validation untuk kolesterol
chol_cv_scores = cross_val_score(chol_model, X_chol_train, y_chol_train, cv=5)  # 5 fold cross-validation
print("Akurasi Cross-Validation Kolesterol:", chol_cv_scores.mean())

# Hyperparameter tuning menggunakan GridSearchCV untuk kolesterol
chol_param_grid = {
    'penalty': ['l2'],
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 500, 1000]
}

chol_grid_search = GridSearchCV(estimator=chol_model, param_grid=chol_param_grid, cv=5, n_jobs=-1, verbose=2)
chol_grid_search.fit(X_chol_train, y_chol_train)

# Menampilkan hasil tuning terbaik untuk kolesterol
print("Best Hyperparameters Kolesterol:", chol_grid_search.best_params_)

# Evaluasi model terbaik pada data uji untuk kolesterol
best_chol_model = chol_grid_search.best_estimator_
y_chol_pred = best_chol_model.predict(X_chol_test)
chol_accuracy = accuracy_score(y_chol_test, y_chol_pred)
chol_report = classification_report(y_chol_test, y_chol_pred, target_names=label_encoder_cholesterol.classes_)

# Cetak hasil evaluasi untuk kolesterol
print("Akurasi Model Setelah Tuning (Kolesterol):", chol_accuracy)
print("\nLaporan Klasifikasi Kolesterol:\n", chol_report)

# Prediksi obat
X_drug = data[['Age', 'Sex_Encoded', 'BP_Encoded', 'Na_to_K']]
y_drug = data['Drug_Encoded']

# Normalisasi data untuk prediksi obat
X_drug_scaled = scaler.fit_transform(X_drug)

# Split data menjadi train dan test untuk prediksi obat
X_drug_train, X_drug_test, y_drug_train, y_drug_test = train_test_split(X_drug_scaled, y_drug, test_size=0.2, random_state=42, stratify=y_drug)

# Model Logistic Regression untuk prediksi obat
drug_model = LogisticRegression(random_state=42)

# Cross-validation untuk obat
drug_cv_scores = cross_val_score(drug_model, X_drug_train, y_drug_train, cv=5)
print("Akurasi Cross-Validation Obat:", drug_cv_scores.mean())

# Hyperparameter tuning menggunakan GridSearchCV untuk obat
drug_param_grid = {
    'penalty': ['l2'],
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 500, 1000]
}

drug_grid_search = GridSearchCV(estimator=drug_model, param_grid=drug_param_grid, cv=5, n_jobs=-1, verbose=2)
drug_grid_search.fit(X_drug_train, y_drug_train)

# Menampilkan hasil tuning terbaik untuk obat
print("Best Hyperparameters Obat:", drug_grid_search.best_params_)

# Evaluasi model terbaik pada data uji untuk obat
best_drug_model = drug_grid_search.best_estimator_
y_drug_pred = best_drug_model.predict(X_drug_test)
drug_accuracy = accuracy_score(y_drug_test, y_drug_pred)
drug_report = classification_report(y_drug_test, y_drug_pred, target_names=label_encoder_drug.classes_)

# Cetak hasil evaluasi untuk obat
print("Akurasi Model Setelah Tuning (Obat):", drug_accuracy)
print("\nLaporan Klasifikasi Obat:\n", drug_report)

# Simpan model, scaler, dan label encoder untuk kedua prediksi
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(best_chol_model, 'model/cholesterol_model_tuned.pkl')
joblib.dump(best_drug_model, 'model/drug_model_tuned.pkl')
joblib.dump(label_encoder_cholesterol, 'model/label_encoder_cholesterol.pkl')
joblib.dump(label_encoder_drug, 'model/label_encoder_drug.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\nModel, scaler, dan label encoder berhasil disimpan!")

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import numpy as np

# path ke data bersih
base_dir = Path(__file__).parent.parent
clean_path = base_dir / "clean_data" / "clean_play_tennis_dataset.csv"

if not clean_path.exists():
    raise FileNotFoundError("File clean_play_tennis_dataset.csv tidak ditemukan di folder clean_data")

# baca dataset
df = pd.read_csv(clean_path)

# pisahkan fitur (X) dan target (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# split data 70:30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ========== Training SVM ==========
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, zero_division=0)

print("\n=== Support Vector Machine (SVM) ===")
print("Akurasi:", svm_acc)
print(svm_report)

# Simpan model dan report SVM
svm_dir = base_dir / "model" / "SVM"
svm_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(svm_model, svm_dir / "svm_model.pkl")

with open(svm_dir / "svm_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi: {svm_acc}\n\n")
    f.write(svm_report)

# ========== Training Linear Regression ==========
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# prediksi hasil linear regression (kontinu)
y_pred_lr_continuous = lr_model.predict(X_test)

# konversi prediksi ke 0/1 terdekat agar bisa evaluasi klasifikasi
y_pred_lr = np.rint(y_pred_lr_continuous).astype(int)

lr_acc = accuracy_score(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr_continuous)
r2 = r2_score(y_test, y_pred_lr_continuous)

print("\n=== Linear Regression ===")
print("Akurasi (dibulatkan):", lr_acc)
print("MSE:", mse)
print("R² Score:", r2)

# Simpan model Linear Regression dan report
lr_dir = base_dir / "model" / "LR"
lr_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(lr_model, lr_dir / "lr_model.pkl")

with open(lr_dir / "lr_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi (dibulatkan): {lr_acc}\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"R2 Score: {r2}\n")

print("\nModel dan report berhasil disimpan.")

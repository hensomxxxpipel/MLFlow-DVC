import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = 'http://127.0.0.1:8080'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Tugas ML")

base_dir = Path(__file__).parent.parent
clean_path = base_dir / "clean_data" / "clean_play_tennis_dataset.csv"

if not clean_path.exists():
    raise FileNotFoundError("File clean_play_tennis_dataset.csv tidak ditemukan di folder clean_data")

df = pd.read_csv(clean_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==================== Training SVM + MLflow ====================
with mlflow.start_run(run_name="SVM"):
    svm_model = SVC(kernel="rbf", random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_report = classification_report(y_test, y_pred_svm, zero_division=0)

    print("\n=== Support Vector Machine (SVM) ===")
    print("Akurasi:", svm_acc)
    print(svm_report)

    mlflow.log_param("kernel", "rbf")
    mlflow.log_metric("accuracy", svm_acc)

    svm_dir = base_dir / "model" / "SVM"
    svm_dir.mkdir(parents=True, exist_ok=True)
    model_path = svm_dir / "svm_model.pkl"
    joblib.dump(svm_model, model_path)

    # log model sebagai MLflow model
    mlflow.sklearn.log_model(svm_model, "svm_model")

    # log file model.pkl sebagai artifact
    mlflow.log_artifact(model_path)

    report_path = svm_dir / "svm_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Akurasi: {svm_acc}\n\n")
        f.write(svm_report)

    mlflow.log_artifact(report_path)

# ==================== Training Linear Regression + MLflow ====================
with mlflow.start_run(run_name="LinearRegression"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr_continuous = lr_model.predict(X_test)
    y_pred_lr = np.rint(y_pred_lr_continuous).astype(int)

    lr_acc = accuracy_score(y_test, y_pred_lr)
    mse = mean_squared_error(y_test, y_pred_lr_continuous)
    r2 = r2_score(y_test, y_pred_lr_continuous)

    print("\n=== Linear Regression ===")
    print("Akurasi (dibulatkan):", lr_acc)
    print("MSE:", mse)
    print("R² Score:", r2)

    mlflow.log_metric("accuracy_rounded", lr_acc)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    lr_dir = base_dir / "model" / "LR"
    lr_dir.mkdir(parents=True, exist_ok=True)
    model_path = lr_dir / "lr_model.pkl"
    joblib.dump(lr_model, model_path)

    mlflow.sklearn.log_model(lr_model, "lr_model")

    # log file model.pkl sebagai artifact
    mlflow.log_artifact(model_path)

    report_path = lr_dir / "lr_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Akurasi (dibulatkan): {lr_acc}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n")

    mlflow.log_artifact(report_path)

print("\nModel, report, dan artifact berhasil disimpan & dilog ke MLflow.")

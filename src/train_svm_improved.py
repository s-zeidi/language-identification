print("===== IMPROVED SVM TRAINING STARTED =====")

import time
import pandas as pd
import joblib
from pathlib import Path

from datasets import load_from_disk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models" / "baselines"

RESULTS_FILE = RESULTS_DIR / "model_results.csv"


# -------------------------
# Load dataset
# -------------------------
def load_data():

    print("\n[1/5] Loading dataset...")

    train = load_from_disk(DATA_DIR / "train_arrow")
    test = load_from_disk(DATA_DIR / "test_arrow")

    x_train = train["sentence"]
    y_train = train["label"]

    x_test = test["sentence"]
    y_test = test["label"]

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples : {len(x_test)}")

    return x_train, y_train, x_test, y_test


# -------------------------
# TF-IDF features (Improved)
# -------------------------
def build_features(x_train, x_test):

    print("\n[2/5] Building improved TF-IDF features...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1,5),      # <-- changed
        min_df=2,
        sublinear_tf=True,      # <-- added
        smooth_idf=True
    )

    print("Fitting TF-IDF on training data...")
    x_train_vec = vectorizer.fit_transform(x_train)

    print("Transforming test data...")
    x_test_vec = vectorizer.transform(x_test)

    print("Feature space size:", x_train_vec.shape[1])

    return vectorizer, x_train_vec, x_test_vec


# -------------------------
# Evaluate model
# -------------------------
def evaluate(model, x_train, y_train, x_test, y_test):

    start = time.time()

    print("Training model...")
    model.fit(x_train, y_train)

    train_time = time.time() - start

    print("Predicting...")
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    return acc, macro_f1, train_time


# -------------------------
# Save model
# -------------------------
def save_model(name, model, vectorizer):

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    joblib.dump(vectorizer, MODELS_DIR / f"{name}_vectorizer.joblib")

    print("Model saved:", MODELS_DIR / f"{name}.joblib")


# -------------------------
# Save results
# -------------------------
def log_results(name, acc, f1, train_time):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{
        "model": name,
        "accuracy": acc,
        "macro_f1": f1,
        "train_time_sec": round(train_time, 2)
    }])

    if RESULTS_FILE.exists():
        row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(RESULTS_FILE, index=False)


# -------------------------
# Main
# -------------------------
def run():

    print("\n===== PIPELINE STARTED =====")

    x_train, y_train, x_test, y_test = load_data()

    vectorizer, x_train_vec, x_test_vec = build_features(x_train, x_test)

    print("\n[3/5] Training improved SVM...")

    model = LinearSVC(
        C=2.0,                 # stronger margin
        max_iter=5000
    )

    acc, f1, train_time = evaluate(
        model,
        x_train_vec,
        y_train,
        x_test_vec,
        y_test
    )

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")

    save_model("LinearSVM_improved", model, vectorizer)

    log_results("LinearSVM_improved", acc, f1, train_time)

    print("\n===== TRAINING FINISHED =====")


if __name__ == "__main__":
    run()
print("===== BASELINE SCRIPT STARTED =====")

import time
import pandas as pd
import joblib
from pathlib import Path

print("Libraries imported (basic)")

from datasets import load_from_disk

print("Datasets library imported")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

print("Sklearn imported successfully")


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models" / "baselines"

RESULTS_FILE = RESULTS_DIR / "model_results.csv"

print("Project paths configured")


# -------------------------
# Load dataset
# -------------------------
def load_data():

    print("\n[1/5] Loading dataset...")

    train = load_from_disk(DATA_DIR / "train_arrow")
    test = load_from_disk(DATA_DIR / "test_arrow")

    print("Datasets loaded from disk")

    x_train = train["sentence"]
    y_train = train["label"]

    x_test = test["sentence"]
    y_test = test["label"]

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples : {len(x_test)}")

    return x_train, y_train, x_test, y_test


# -------------------------
# TF-IDF features
# -------------------------
def build_features(x_train, x_test):

    print("\n[2/5] Building TF-IDF features (this may take several minutes)...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2
    )

    print("Fitting TF-IDF on training data...")
    x_train_vec = vectorizer.fit_transform(x_train)

    print("Transforming test data...")
    x_test_vec = vectorizer.transform(x_test)

    print("TF-IDF completed")
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

    print("Predicting on test set...")
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    return acc, macro_f1, train_time


# -------------------------
# Save model
# -------------------------
def save_model(name, model, vectorizer):

    print("[4/5] Saving model...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    joblib.dump(vectorizer, MODELS_DIR / f"{name}_vectorizer.joblib")

    print(f"Model saved → {MODELS_DIR / f'{name}.joblib'}")


# -------------------------
# Save results
# -------------------------
def log_results(name, acc, f1, train_time):

    print("[5/5] Logging results...")

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

    print("Results logged")


# -------------------------
# Main pipeline
# -------------------------
def run():

    print("\n===== BASELINE PIPELINE STARTED =====")

    x_train, y_train, x_test, y_test = load_data()

    vectorizer, x_train_vec, x_test_vec = build_features(x_train, x_test)

    models = {
        #"NaiveBayes": MultinomialNB(),
        #"LogisticRegression": LogisticRegression(max_iter=1000),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
           # n_jobs=1
        ),
        "LinearSVM": LinearSVC()
    }

    print("\n[3/5] Training models...")

    for name, model in models.items():

        print(f"\n----- Training {name} -----")

        acc, f1, train_time = evaluate(
            model,
            x_train_vec,
            y_train,
            x_test_vec,
            y_test
        )

        print(f"Accuracy : {acc:.4f}")
        print(f"Macro F1 : {f1:.4f}")
        print(f"Train time: {train_time:.2f}s")

        save_model(name, model, vectorizer)

        log_results(name, acc, f1, train_time)

    print("\n===== BASELINE FINISHED SUCCESSFULLY =====")


if __name__ == "__main__":
    run()
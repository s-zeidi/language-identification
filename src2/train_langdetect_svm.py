import pickle
from pathlib import Path

from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "langdetect_svm.pkl"
VECTORIZER_PATH = MODEL_DIR / "langdetect_vectorizer.pkl"


def main():

    print("Loading dataset")

    train = load_from_disk(DATA_DIR / "train_arrow")
    test = load_from_disk(DATA_DIR / "test_arrow")

    X_train = train["sentence"]
    y_train = train["label"]

    X_test = test["sentence"]
    y_test = test["label"]

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    print("\nBuilding n-gram tokenizer")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1,3),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Vocabulary size:", len(vectorizer.vocabulary_))

    print("\nTraining SVM")

    model = LinearSVC()

    model.fit(X_train_vec, y_train)

    print("Evaluating")

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("\nResults")
    print("Accuracy:", acc)
    print("Macro F1:", f1)

    print("\nSaving model")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model saved:", MODEL_PATH)
    print("Vectorizer saved:", VECTORIZER_PATH)


if __name__ == "__main__":
    main()
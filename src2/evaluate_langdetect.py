import pickle
from pathlib import Path
from tqdm import tqdm

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "langdetect_model.pkl"


def main():

    print("Loading model")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Loading test dataset")

    test = load_from_disk(DATA_DIR / "test_arrow")

    texts = test["sentence"]
    labels = test["label"]

    print("Test samples:", len(texts))

    preds = []

    print("\nEvaluating")

    for text in tqdm(texts):

        pred = model.predict(text)

        preds.append(pred)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print("\nResults")
    print("Accuracy:", acc)
    print("Macro F1:", f1)


if __name__ == "__main__":
    main()
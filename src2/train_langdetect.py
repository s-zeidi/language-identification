import pickle
from pathlib import Path

from datasets import load_from_disk

from src2.langdetect_model import LangDetectModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "langdetect_model.pkl"


def main():

    print("Loading dataset")

    train = load_from_disk(DATA_DIR / "train_arrow")

    texts = train["sentence"]
    labels = train["label"]

    print("Training samples:", len(texts))

    model = LangDetectModel()

    print("Training model")

    model.train(texts, labels)

    print("Saving model")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
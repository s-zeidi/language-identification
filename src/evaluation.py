import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import pickle
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk

from src.model import CharCNN


# =====================================================
# SELECT MODEL HERE
# =====================================================

"""
AVAILABLE MODELS

# sklearn
models/baselines/LinearSVM.joblib
models/baselines/NaiveBayes.joblib
models/baselines/LinearSVM_improved.joblib

# CNN
models/charcnn_mps.pkl
models/charcnn_ngram.pkl
"""

MODEL_PATH = "models/baselines/LinearSVM_improved.joblib"


"""
AVAILABLE TOKENIZERS

models/tokenizer.pkl
models/tokenizer_ngram.pkl
"""

TOKENIZER_PATH = None
# Example:
# TOKENIZER_PATH = "models/tokenizer.pkl"
# TOKENIZER_PATH = "models/tokenizer_ngram.pkl"


BATCH_SIZE = 128


class LanguageDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoded = self.tokenizer.encode(self.texts[idx])

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def save_results(model_name, acc, f1):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_FILE = PROJECT_ROOT / "results" / "model_results.csv"

    row = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "macro_f1": f1
    }])

    if RESULTS_FILE.exists():
        row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(RESULTS_FILE, index=False)


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    model_path = PROJECT_ROOT / MODEL_PATH

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Device:", device)
    print("Model path:", model_path)

    # ---------------------------
    # Load dataset
    # ---------------------------

    print("\nLoading test dataset")

    test = load_from_disk(DATA_DIR / "test_arrow")

    texts = test["sentence"]
    labels = test["label"]

    num_classes = len(set(labels))

    print("Test samples:", len(texts))

    # =====================================================
    # SKLEARN MODELS
    # =====================================================

    if model_path.suffix == ".joblib":

        print("\nDetected sklearn model")

        model = joblib.load(model_path)

        vectorizer_path = model_path.parent / f"{model_path.stem}_vectorizer.joblib"

        print("Loading vectorizer:", vectorizer_path)

        vectorizer = joblib.load(vectorizer_path)

        print("Vectorizing test data")

        X_test = vectorizer.transform(texts)

        preds = model.predict(X_test)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        model_name = model_path.stem

    # =====================================================
    # CNN MODELS
    # =====================================================

    else:

        print("\nDetected CNN model")

        tokenizer_path = PROJECT_ROOT / TOKENIZER_PATH

        print("Loading tokenizer:", tokenizer_path)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        if hasattr(tokenizer, "char2id"):
            vocab_size = len(tokenizer.char2id)

        elif hasattr(tokenizer, "ngram2id"):
            vocab_size = len(tokenizer.ngram2id)

        else:
            raise ValueError("Unknown tokenizer format")

        dataset = LanguageDataset(texts, labels, tokenizer)

        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        print("Vocabulary size:", vocab_size)

        print("Loading CNN model")

        model = CharCNN(vocab_size, num_classes).to(device)

        with open(model_path, "rb") as f:
            state_dict = pickle.load(f)

        model.load_state_dict(state_dict)

        model.eval()

        all_preds = []
        all_labels = []

        print("\nEvaluating\n")

        with torch.no_grad():

            for x, y in tqdm(loader):

                x = x.to(device)

                outputs = model(x)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        model_name = model_path.stem

    print("\nAccuracy:", acc)
    print("Macro F1:", f1)

    save_results(model_name, acc, f1)

    print("\nSaved to results/model_results.csv")


if __name__ == "__main__":
    main()
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk

from src.model import CharCNN


# ==============================
# CHANGE ONLY THIS
# ==============================

MODEL_TYPE = "ngram"   # "char"  or  "ngram"


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


def load_model_and_tokenizer(model_dir):

    if MODEL_TYPE == "char":

        model_path = model_dir / "charcnn_mps.pkl"
        tokenizer_path = model_dir / "tokenizer.pkl"
        model_name = "CharCNN_char"

    elif MODEL_TYPE == "ngram":

        model_path = model_dir / "charcnn_ngram.pkl"
        tokenizer_path = model_dir / "tokenizer_ngram.pkl"
        model_name = "CharCNN_ngram"

    else:
        raise ValueError("MODEL_TYPE must be 'char' or 'ngram'")

    print("Loading tokenizer:", tokenizer_path.name)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # detect vocab automatically
    if hasattr(tokenizer, "char2id"):
        vocab_size = len(tokenizer.char2id)

    elif hasattr(tokenizer, "ngram2id"):
        vocab_size = len(tokenizer.ngram2id)

    else:
        raise ValueError("Tokenizer format not recognized")

    print("Vocabulary size:", vocab_size)

    return model_path, tokenizer, vocab_size, model_name


def save_results(model_name, acc, f1):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_FILE = PROJECT_ROOT / "results" / "model_results.csv"

    RESULTS_FILE.parent.mkdir(exist_ok=True)

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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Device:", device)
    print("Model type:", MODEL_TYPE)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"

    # --------------------------
    # Load model + tokenizer
    # --------------------------

    model_path, tokenizer, vocab_size, model_name = load_model_and_tokenizer(MODEL_DIR)

    # --------------------------
    # Load dataset
    # --------------------------

    print("\nLoading test dataset")

    test = load_from_disk(DATA_DIR / "test_arrow")

    texts = test["sentence"]
    labels = test["label"]

    print("Test samples:", len(texts))

    dataset = LanguageDataset(texts, labels, tokenizer)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    num_classes = len(set(labels))

    # --------------------------
    # Load model
    # --------------------------

    print("\nLoading model:", model_path.name)

    model = CharCNN(vocab_size, num_classes).to(device)

    with open(model_path, "rb") as f:
        state_dict = pickle.load(f)

    model.load_state_dict(state_dict)

    model.eval()

    # --------------------------
    # Evaluation
    # --------------------------

    print("\nEvaluating\n")

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in tqdm(loader):

            x = x.to(device)

            outputs = model(x)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print("\nAccuracy:", acc)
    print("Macro F1:", f1)

    save_results(model_name, acc, f1)

    print("\nSaved to results/model_results.csv")


if __name__ == "__main__":
    main()
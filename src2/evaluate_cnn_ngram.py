import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from datasets import load_from_disk
from src2.model_cnn_ngram import NgramCNN


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


def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"

    MODEL_PATH = MODEL_DIR / "cnn_ngram.pt"
    TOKENIZER_PATH = MODEL_DIR / "tokenizer_ngram.pkl"

    print("Loading tokenizer")

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading dataset")

    test = load_from_disk(DATA_DIR / "test_arrow")

    texts = test["sentence"]
    labels = test["label"]

    dataset = LanguageDataset(texts, labels, tokenizer)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    vocab_size = len(tokenizer.ngram2id)
    num_classes = len(set(labels))

    print("Vocabulary size:", vocab_size)
    print("Languages:", num_classes)

    print("Loading model")

    model = NgramCNN(vocab_size, num_classes).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

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

    print("\nResults")
    print("Accuracy:", acc)
    print("Macro F1:", f1)


if __name__ == "__main__":
    main()
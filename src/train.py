import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import pickle

from datasets import load_from_disk

from src.ngram_tokenizer import NgramTokenizer
from src.model import CharCNN


MAX_LEN = 512
BATCH_SIZE = 256
EPOCHS = 12


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

    # DEVICE
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # PATHS
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"

    MODEL_DIR.mkdir(exist_ok=True)

    # LOAD DATA
    print("\nLoading dataset")

    train = load_from_disk(DATA_DIR / "train_arrow")
    val = load_from_disk(DATA_DIR / "validation_arrow")

    train_texts = train["sentence"]
    train_labels = train["label"]

    val_texts = val["sentence"]
    val_labels = val["label"]

    print("Train samples:", len(train_texts))
    print("Validation samples:", len(val_texts))

    # TOKENIZER
    print("\nBuilding n-gram tokenizer")

    tokenizer = NgramTokenizer(n=3, max_length=MAX_LEN)
    tokenizer.build_vocab(train_texts)

    vocab_size = len(tokenizer.ngram2id)
    num_classes = len(set(train_labels))

    print("Vocabulary size:", vocab_size)
    print("Languages:", num_classes)

    # SAVE TOKENIZER
    tokenizer_path = MODEL_DIR / "tokenizer_ngram.pkl"

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print("Tokenizer saved →", tokenizer_path)

    # DATASET
    train_dataset = LanguageDataset(train_texts, train_labels, tokenizer)
    val_dataset = LanguageDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    # MODEL
    print("\nBuilding CNN model")

    model = CharCNN(vocab_size, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    # TRAINING
    print("\nTraining started\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for x, y in progress:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")

    # SAVE MODEL
    model_path = MODEL_DIR / "charcnn_ngram.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model.state_dict(), f)

    print("\nModel saved →", model_path)


if __name__ == "__main__":
    main()
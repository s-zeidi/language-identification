import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import load_from_disk

from src2.ngram_tokenizer import NgramTokenizer
from src2.model_cnn_ngram import NgramCNN


BATCH_SIZE = 128
EPOCHS = 5
MAX_LEN = 512


class LanguageDataset(Dataset):

    def __init__(self,texts,labels,tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):

        encoded = self.tokenizer.encode(self.texts[idx])

        return (
            torch.tensor(encoded),
            torch.tensor(self.labels[idx])
        )


def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"

    print("Loading dataset")

    train = load_from_disk(DATA_DIR/"train_arrow")
    val = load_from_disk(DATA_DIR/"validation_arrow")

    train_texts = train["sentence"]
    train_labels = train["label"]

    val_texts = val["sentence"]
    val_labels = val["label"]

    print("Building tokenizer")

    tokenizer = NgramTokenizer(max_length=MAX_LEN)
    tokenizer.build_vocab(train_texts)

    vocab_size = len(tokenizer.ngram2id)
    num_classes = len(set(train_labels))

    print("Vocabulary:",vocab_size)

    train_dataset = LanguageDataset(train_texts,train_labels,tokenizer)
    val_dataset = LanguageDataset(val_texts,val_labels,tokenizer)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)

    model = NgramCNN(vocab_size,num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        model.train()

        loop = tqdm(train_loader)

        for x,y in loop:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs,y)

            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(),MODEL_DIR/"cnn_ngram.pt")

    with open(MODEL_DIR/"tokenizer_ngram.pkl","wb") as f:
        pickle.dump(tokenizer,f)

    print("Model saved")


if __name__ == "__main__":
    main()

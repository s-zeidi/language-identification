import pickle
import json
import pandas as pd
import torch
import joblib
from pathlib import Path

from src.model import CharCNN
from src.tokenizer import CharTokenizer


# ==============================
# SELECT MODEL HERE
# ==============================

MODEL_TYPE = "nb"
# options: "svm", "nb", "cnn"


# ==============================
# PATHS
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

LABEL_PATH = MODEL_DIR / "label_mapping.json"
LANGUAGE_CSV = DATA_DIR / "language_codes.csv"


# ==============================
# LOAD LANGUAGE MAPPING
# ==============================

with open(LABEL_PATH) as f:
    label_map = json.load(f)

df = pd.read_csv(LANGUAGE_CSV)
code_to_name = dict(zip(df.code, df.language))


# ==============================
# LOAD MODEL
# ==============================

if MODEL_TYPE == "svm":

    print("Loading SVM model")

    model = joblib.load(MODEL_DIR / "baselines/LinearSVM.joblib")
    vectorizer = joblib.load(MODEL_DIR / "baselines/LinearSVM_vectorizer.joblib")


elif MODEL_TYPE == "nb":

    print("Loading Naive Bayes model")

    model = joblib.load(MODEL_DIR / "baselines/NaiveBayes.joblib")
    vectorizer = joblib.load(MODEL_DIR / "baselines/NaiveBayes_vectorizer.joblib")


elif MODEL_TYPE == "cnn":

    print("Loading CNN model")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = pickle.load(open(MODEL_DIR / "tokenizer.pkl", "rb"))

    vocab_size = len(tokenizer.char2id)
    num_classes = len(label_map)

    model = CharCNN(vocab_size, num_classes)

    model.load_state_dict(
        torch.load(MODEL_DIR / "charcnn_mps.pt", map_location=device)
    )

    model.to(device)
    model.eval()


# ==============================
# PREDICT FUNCTION
# ==============================

def predict(text):

    if MODEL_TYPE in ["svm", "nb"]:

        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

    elif MODEL_TYPE == "cnn":

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        encoded = tokenizer.encode(text)

        x = torch.tensor([encoded]).to(device)

        with torch.no_grad():

            outputs = model(x)
            pred = torch.argmax(outputs, dim=1).item()

    lang_code = label_map[str(pred)]

    return code_to_name.get(lang_code, lang_code)


# ==============================
# INTERACTIVE LOOP
# ==============================

def main():

    print("Model:", MODEL_TYPE)

    while True:

        text = input("\nEnter text: ")

        if text == "quit":
            break

        if text.strip() == "":
            print("Empty text")
            continue

        language = predict(text)

        print("Detected language:", language)


if __name__ == "__main__":
    main()
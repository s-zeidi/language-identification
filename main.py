import joblib
import json
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = MODEL_DIR / "baselines"

# load model + vectorizer
model = joblib.load(BASELINE_DIR / "NaiveBayes.joblib")
vectorizer = joblib.load(BASELINE_DIR / "NaiveBayes_vectorizer.joblib")

# load label mapping
with open(MODEL_DIR / "label_mapping.json") as f:
    label_map = json.load(f)

# language names (simplified)
LANG_NAMES = {
    "eng":"English",
    "deu":"German",
    "fas":"Farsi",
    "spa":"Spanish",
    "fra":"French",
    "ita":"Italian",
    "zho":"Chinese",
    "kor":"Korean",
    "jpn":"Japanese",
    "rus":"Russian",
    "nld":"Dutch",
    "sco":"Scots"
}

print("Naive Bayes loaded. Type text (or 'quit').")

while True:

    text = input("\nEnter text: ")

    if text.lower() == "quit":
        break

    X = vectorizer.transform([text])

    pred = model.predict(X)[0]

    iso_code = label_map[str(pred)]

    language = LANG_NAMES.get(iso_code, iso_code)

    print("Detected language:", language)
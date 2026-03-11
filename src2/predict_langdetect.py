import pickle
import pandas as pd
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "langdetect_model.pkl"
LABEL_PATH = PROJECT_ROOT / "models" / "label_mapping.json"
LANGUAGE_CSV = PROJECT_ROOT / "data" / "language_codes.csv"


def main():

    print("Loading model")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Loading label mapping")

    with open(LABEL_PATH) as f:
        label_map = json.load(f)

    print("Loading language names")

    df = pd.read_csv(LANGUAGE_CSV)
    code_to_name = dict(zip(df.code, df.language))

    while True:

        text = input("\nEnter text: ")

        if text == "quit":
            break

        pred = model.predict(text)

        # numeric label → ISO code
        lang_code = label_map[str(pred)]

        # ISO code → full language name
        language_name = code_to_name.get(lang_code, lang_code)

        print("Detected language:", language_name)


if __name__ == "__main__":
    main()
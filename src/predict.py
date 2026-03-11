import joblib
import pickle
import json
import torch
from pathlib import Path

from src.model import CharCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = MODELS_DIR / "baselines"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------- Language Name Mapping --------
LANGUAGE_NAMES = {
    "afr":"Afrikaans","als":"Alemannic","amh":"Amharic","ara":"Arabic","asm":"Assamese",
    "ava":"Avar","aym":"Aymara","aze":"Azerbaijani","bak":"Bashkir","bar":"Bavarian",
    "bel":"Belarusian","ben":"Bengali","bho":"Bhojpuri","bos":"Bosnian","bre":"Breton",
    "bul":"Bulgarian","cat":"Catalan","ceb":"Cebuano","ces":"Czech","che":"Chechen",
    "chv":"Chuvash","cmn":"Chinese Mandarin","cos":"Corsican","cym":"Welsh","dan":"Danish",
    "deu":"German","dsb":"Lower Sorbian","ell":"Greek","eng":"English","epo":"Esperanto",
    "est":"Estonian","eus":"Basque","fas":"Farsi","fao":"Faroese","fin":"Finnish",
    "fra":"French","fry":"West Frisian","gle":"Irish","glg":"Galician","grn":"Guarani",
    "guj":"Gujarati","hat":"Haitian Creole","hau":"Hausa","heb":"Hebrew","hin":"Hindi",
    "hrv":"Croatian","hsb":"Upper Sorbian","hun":"Hungarian","hye":"Armenian","ibo":"Igbo",
    "ind":"Indonesian","isl":"Icelandic","ita":"Italian","jav":"Javanese","jpn":"Japanese",
    "kal":"Greenlandic","kan":"Kannada","kat":"Georgian","kaz":"Kazakh","khm":"Khmer",
    "kir":"Kyrgyz","kor":"Korean","kur":"Kurdish","lao":"Lao","lat":"Latin",
    "lav":"Latvian","lit":"Lithuanian","ltz":"Luxembourgish","mal":"Malayalam","mar":"Marathi",
    "mkd":"Macedonian","mlg":"Malagasy","mlt":"Maltese","mon":"Mongolian","mri":"Maori",
    "msa":"Malay","mya":"Burmese","nep":"Nepali","nld":"Dutch","nno":"Norwegian Nynorsk",
    "nob":"Norwegian Bokmål","oci":"Occitan","ori":"Odia","pan":"Punjabi","pol":"Polish",
    "por":"Portuguese","pus":"Pashto","que":"Quechua","ron":"Romanian","rus":"Russian",
    "sco":"Scots","sin":"Sinhala","slk":"Slovak","slv":"Slovenian","som":"Somali",
    "spa":"Spanish","sqi":"Albanian","srp":"Serbian","sun":"Sundanese","swa":"Swahili",
    "swe":"Swedish","tam":"Tamil","tat":"Tatar","tel":"Telugu","tgk":"Tajik",
    "tgl":"Tagalog","tha":"Thai","tur":"Turkish","uig":"Uyghur","ukr":"Ukrainian",
    "urd":"Urdu","uzb":"Uzbek","vie":"Vietnamese","vol":"Volapük","war":"Waray",
    "yid":"Yiddish","yor":"Yoruba","zho":"Chinese","zul":"Zulu"
}


class LanguageDetector:

    def __init__(self, model_type="svm"):

        self.model_type = model_type

        if model_type == "svm":
            self._load_svm()

        elif model_type == "nb":
            self._load_nb()

        elif model_type == "cnn":
            self._load_cnn()

        else:
            raise ValueError("model_type must be svm, nb, or cnn")

        # Load label mapping (WiLI label -> ISO code)
        with open(MODELS_DIR / "label_mapping.json") as f:
            self.label_map = json.load(f)


    # ---------- SVM ----------
    def _load_svm(self):

        print("Loading Linear SVM")

        self.model = joblib.load(BASELINE_DIR / "LinearSVM_improved.joblib")
        self.vectorizer = joblib.load(BASELINE_DIR / "LinearSVM_improved_vectorizer.joblib")


    # ---------- NAIVE BAYES ----------
    def _load_nb(self):

        print("Loading Naive Bayes")

        self.model = joblib.load(BASELINE_DIR / "NaiveBayes.joblib")
        self.vectorizer = joblib.load(BASELINE_DIR / "NaiveBayes_vectorizer.joblib")


    # ---------- CNN ----------
    def _load_cnn(self):

        print("Loading CNN")

        with open(MODELS_DIR / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        vocab_size = len(self.tokenizer.char2id)

        with open(MODELS_DIR / "num_classes.pkl", "rb") as f:
            num_classes = pickle.load(f)

        self.model = CharCNN(vocab_size, num_classes).to(DEVICE)

        with open(MODELS_DIR / "charcnn_mps.pkl", "rb") as f:
            state_dict = pickle.load(f)

        self.model.load_state_dict(state_dict)
        self.model.eval()


    # ---------- PREDICT ----------
    def predict(self, text):

        if self.model_type in ["svm", "nb"]:

            vec = self.vectorizer.transform([text])
            pred = self.model.predict(vec)[0]

        elif self.model_type == "cnn":

            encoded = self.tokenizer.encode(text)
            x = torch.tensor([encoded]).to(DEVICE)

            with torch.no_grad():
                logits = self.model(x)
                pred = torch.argmax(logits, dim=1).item()

        # Convert numeric label -> ISO code
        iso_code = self.label_map[str(pred)]

        # Convert ISO -> Full name
        language_name = LANGUAGE_NAMES.get(iso_code, iso_code)

        return language_name
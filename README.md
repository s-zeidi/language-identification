# Language Identification with Naive Bayes

This project implements a **language identification system** using
**character n-gram features** and a **Multinomial Naive Bayes
classifier**.

The model predicts the language of a given sentence.\
The trained model and vectorizer are stored on **Hugging Face** and
downloaded automatically when running the notebook.

------------------------------------------------------------------------

# Project Structure

language-identification/ │ ├── notebook.ipynb \# Main Jupyter notebook
with all steps\
├── report.pdf \# Project report\
├── requirements.txt \# Python dependencies\
├── README.md \# Project documentation

------------------------------------------------------------------------

# Installation

Clone the repository:

git clone `<repository-url>`{=html}\
cd language-identification

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

# Running the Project

Open the notebook:

jupyter notebook notebook.ipynb

or

jupyter lab notebook.ipynb

Run all notebook cells sequentially.

The notebook will:

1.  Install required libraries
2.  Download the trained model from Hugging Face
3.  Load the model and vectorizer
4.  Run predictions on sample text

------------------------------------------------------------------------

# Model

The language detection system uses:

Feature extraction: Character n‑grams (1--3 grams) with TF‑IDF.

Classifier: Multinomial Naive Bayes.

This approach captures language‑specific character patterns that help
distinguish between languages.

------------------------------------------------------------------------

# Model Files (Downloaded from Hugging Face)

nb_model.joblib\
vectorizer.joblib\
label_mapping.json\
language_codes.csv

These contain:

-   trained Naive Bayes classifier
-   TF‑IDF vectorizer
-   label mappings
-   language code mappings

------------------------------------------------------------------------

# Example

Input:

Hello how are you today

Output:

Language Code: en\
Language: English

Another example:

Bonjour comment allez vous

Output:

Language Code: fr\
Language: French

------------------------------------------------------------------------

# Requirements

scikit-learn\
datasets\
pandas\
joblib\
huggingface_hub\
tqdm

Install with:

pip install -r requirements.txt

------------------------------------------------------------------------

# Author

Language Identification Project

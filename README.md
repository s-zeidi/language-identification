# Language Identification with Naive Bayes

This project implements a **language identification system** using
**character n-gram features** and a ** Naive Bayes
classifier**.

The model identifies the language of a given sentence.\
The trained model and vectorizer are stored on **Hugging Face** and
downloaded automatically when running the notebook.

------------------------------------------------------------------------

# Project Structure

language-identification/ 
│ 

├── language_detection_demo.ipynb \# Main Jupyter notebook
with all steps\
├── report.pdf \# Project report\
├── requirements.txt \# Python dependencies\
├── README.md \# Project documentation

------------------------------------------------------------------------

# Installation


Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

# Running the Project

Open the notebook:

language_detection_demo.ipynb


Run all notebook cells sequentially.

The notebook will:

1.  Install required libraries
2.  Download the trained model from Hugging Face
3.  Load the model and vectorizer
4.  Run predictions on sample text

------------------------------------------------------------------------

# Model

The language detection system uses:

Feature extraction: Character n‑grams (1,5 grams) with TF‑IDF.

Classifier: Naive Bayes.

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

Language: English

Another example:

Bonjour comment allez vous

Output:

Language: French

------------------------------------------------------------------------


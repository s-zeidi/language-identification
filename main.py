from src.predict import LanguageDetector

detector = LanguageDetector("svm")

while True:

    text = input("\nEnter text: ")

    if text == "quit":
        break

    lang = detector.predict(text)

    print("Detected language:", lang)
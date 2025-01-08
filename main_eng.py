import os
import random
from collections import Counter
from langdetect import detect
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from rake_nltk import Rake
import nltk

# Funcție pentru a citi textul de la tastatură sau din fișier
def read_text(file_path='input.txt'):
    choice = input("Read text from (1) Command Line or (2) File? Enter 1 or 2: ")
    if choice == '1':
        return input("Enter the text: ")
    elif choice == '2':
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            print("File not found!")
            exit()
    else:
        print("Invalid choice!")
        exit()

# Funcție pentru detectarea limbii textului
def detect_language(text):
    # Verificare în ce limbă este textul
    language = detect(text) # detectează limba textului
    print(f"Textul este scris în limba: {language}")
    return language

# Function to analyze stylometry of the text
def analyze_stylometry(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalnum()]

    num_words = len(words)
    num_characters = len(text)
    word_freq = Counter(words)

    print("\nStylometric analysis:")
    print(f"Word count: {num_words}")
    print(f"Character count: {num_characters}")
    print(f"Word frequency: {word_freq}")

    return words

# Function to get synonyms, hypernyms, and antonyms using WordNet
def get_synonyms_hypernyms_antonyms(word):
    synonyms, hypernyms, antonyms = set(), set(), set()

    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.lemmas()[0].name())

    return list(synonyms), list(hypernyms), list(antonyms)

# Function to normalize underscores in words
def normalize_underscores(word):
    return word.replace("_", "-")  # Replace underscores with hyphens

# Function to replace words in the text with synonyms, hypernyms, or antonyms
def replace_words(text, words):
    replacements = {}
    for word in words:
        synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms(word)
        if random.random() < 0.2:  # At least 20% chance
            if synonyms:
                replacements[word] = random.choice(synonyms)
            elif hypernyms:
                replacements[word] = random.choice(hypernyms)
            elif antonyms:
                replacements[word] = f"not {random.choice(antonyms)}"

    # Replace words and normalize underscores to hyphens
    replaced_text = ' '.join([normalize_underscores(replacements.get(word, word)) for word in text.split()])

    return replaced_text

# Function to extract keywords using RAKE
def extract_keywords(text, top_n=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:top_n]
    return keywords

# Function to generate sentences from keywords
def generate_sentences_from_keywords(keywords, text):
    sentences = sent_tokenize(text)
    keyword_sentences = []

    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence:
                keyword_sentences.append(sentence)
                break
        else:
            keyword_sentences.append(f"The concept of '{keyword}' is significant in this context.")

    return keyword_sentences

# Main function to execute all tasks
def main():
    text = read_text()

    language = detect_language(text)
    words = analyze_stylometry(text)

    if language == 'ro':
        print("Processing Romanian text...")
    elif language == 'en':
        print("Processing English text...")
    else:
        print("Unsupported language.")
        exit()

    print("\nGenerating alternative version of the text...")
    modified_text = replace_words(text, words)
    print(f"Modified text:\n{modified_text}")

    print("\nExtracting keywords and generating sentences...")
    keywords = extract_keywords(text)
    print(f"Keywords: {keywords}")

    keyword_sentences = generate_sentences_from_keywords(keywords, text)
    for sentence in keyword_sentences:
        print(sentence)

if __name__ == "__main__":
    main()
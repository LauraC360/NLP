import os
import random
from langdetect import detect
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter
import requests
from rake_nltk import Rake


# Descărcare pachete necesare
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')


# Funcție pentru a citi textul de la tastatură sau din fișier
def read_text(file_path='input.txt'):
    choice = input("Citire text de la (1) Linia de Comandă sau (2) Fișier? Introduceți 1 sau 2: ")
    if choice == '1':
        return input("Introduceți textul: ")
    elif choice == '2':
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            print("Fișierul nu a fost găsit!")
            exit()
    else:
        print("Alegere invalidă!")
        exit()


# Funcție pentru detectarea limbii textului
def detect_language(text):
    # Verificare în ce limbă este textul
    language = detect(text) # detectează limba textului
    if language == 'ro':
        print(f"Textul este scris în limba: Română")
    elif language == 'en':
        print(f"Textul este scris în limba: Engleză")
    else:
        print(f"Textul este scris în limba: {language}")
    return language

# Funcție pentru analiza stilometriei textului
def analyze_stylometry(text):
    tokens = word_tokenize(text)
    # Eliminare semne de punctuație
    words = [word for word in tokens if word.isalnum()]

    # Calcularea datelor stilometrice
    num_words = len(words)
    num_characters = len(text)
    word_freq = Counter(words)
    print(f"Date stilometrice despre text:")
    print(f"Lungimea în cuvinte: {num_words}")
    print(f"Lungimea în caractere: {num_characters}")
    print(f"Frecvența cuvintelor: {word_freq}")
    return words


# Funcție pentru a obține sinonimele din text
# Cu ajutorul wordnet
def get_synonyms_hypernyms_antonyms_ro(word):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # Obținem toate synset-urile pentru cuvânt
    synsets = wn.synsets(word, lang='ron')  # 'ron' este codul pentru limba română
    if not synsets:
        print(f"Nu am găsit rezultate pentru cuvântul: {word}")
        return [], [], []  # Returnează liste goale dacă nu se găsesc synsets

    for synset in synsets:
        # Obținem sinonimele pentru cuvânt
        for lemma in synset.lemmas(lang='ron'):
            synonyms.add(lemma.name())

        # Obținem hiperonimele pentru cuvânt
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.lemmas()[0].name())

    return list(synonyms), list(hypernyms), list(antonyms)


# Funcție pentru a obține sinonimele, hiperonimele și antonimele în limba engleză
def get_synonyms_hypernyms_antonyms_en(word):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # Obținem toate synset-urile pentru cuvânt
    synsets = wn.synsets(word)  # 'word' este cuvântul în limba engleză
    if synonyms:
        print(f"Am găsit sinonime pentru cuvântul: {word}")
    if not synsets:
        print(f"Nu am găsit rezultate pentru cuvântul: {word}")
        return [], [], []  # Returnează liste goale dacă nu se găsesc synsets

    for synset in synsets:
        # Obținem sinonimele pentru cuvânt
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
            # Obținem antonimele pentru cuvânt
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
        # Obținem hiperonimele pentru cuvânt
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.lemmas()[0].name())

    return list(synonyms), list(hypernyms), list(antonyms)

# Testare
# cuvant = "big"
# synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(cuvant)
# print("Synonyms:", synonyms)
# print("Hypernyms:", hypernyms)
# print("Antonyms", antonyms)

# Testare
# # Test pentru cuvântul "casă"
# synonyms, antonyms = get_synonyms_hypernyms_antonyms_ro('locuință')
# print("Sinonime:", synonyms)
# #print("Hipernime:", hypernyms)
# print("Antonime:", antonyms)


# Funcție pentru a înlocui cuvintele cu sinonime
import random

# Function to normalize underscores in words
def normalize_underscores(word):
    return word.replace("_", "-")  # Replace underscores with hyphens

# Function to replace words in the text with synonyms, hypernyms, or antonyms
def replace_words(text, words):
    replacements = {}
    for word in words:
        synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)
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

def apply_replacements(text, replacements):
    words = text.split()
    new_text = ' '.join([replacements.get(word, word) for word in words])
    return new_text

# Funcție pentru procesarea întregului task
def nlp_task():
    text = read_text() # citește textul

    if detect_language(text) == 'ro':
        # Procesare text în limba română
        print("Procesare text în limba română...")
        words = analyze_stylometry(text)
        new_text = replace_words(text, words)
        print("Textul modificat cu sinonime:")
        print(new_text)
        #new_text = apply_replacements(text, new_words)
        #print(new_text)
    elif detect_language(text) == 'en':
        # Procesare text în limba engleză
        print("Procesare text în limba engleză...")

        words = analyze_stylometry(text)
        new_text = replace_words(text, words)

        keywords = extract_keywords(text)
        print("Textul modificat cu sinonime:")
        print(new_text)

        print("Propozitiile generate din cuvinte cheie:")
        keyword_sentences = generate_sentences_from_keywords(keywords, text)
        print(keyword_sentences)
    else:
        # Procesare text în altă limbă
        print("Textul este scris într-o limbă nesuportată.")
        return

# Apelarea funcției pentru procesarea întregului task
nlp_task()
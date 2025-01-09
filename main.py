import os
import re
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
    num_sentences = len(sent_tokenize(text))
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sentence_length = len(words) / num_sentences

    print(f"Date stilometrice despre text:")
    print(f"Lungimea în cuvinte: {num_words}")
    print(f"Lungimea în caractere: {num_characters}")
    print(f"Frecvența cuvintelor: {word_freq}")

    print(f"Numărul de propoziții: {num_sentences}")
    print(f"Lungimea medie a cuvintelor: {avg_word_length:.2f}")
    print(f"Lungimea medie a propozițiilor: {avg_sentence_length:.2f}")
    return words


# Funcție pentru a obține sinonimele din text
# Cu ajutorul wordnet
def get_synonyms_hypernyms_antonyms_ro(word):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # Obținem toate synset-urile pentru cuvânt
    synsets = wn.synsets(word, lang='ron')  # 'ron' este codul pentru limba română
    if synsets:
        print(f"Am găsit sinonime pentru cuvântul: {word}")
    else:
        print(f"Nu am găsit rezultate pentru cuvântul: {word}")
        return [], [], []  # Returnează liste goale dacă nu se găsesc synsets

    for synset in synsets:
        # Obținem sinonimele pentru cuvânt
        for lemma in synset.lemmas(lang='ron'):
            synonyms.add(lemma.name())

        # Obținem hiperonimele pentru cuvânt
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.lemmas()[0].name())

    # Test
    print("Cuvântul:", word)
    print("Sinonime:", synonyms)
    print("Hipernime:", hypernyms)
    print("Antonime:", antonyms)
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

    # Test
    print("Cuvântul:", word)
    print("Sinonime:", synonyms)
    print("Hipernime:", hypernyms)
    print("Antonime:", antonyms)
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

#
# Function to normalize underscores in words
def normalize_underscores(word):
    return word.replace("_", "-")  # Replace underscores with hyphens

def normalize_text(text):
    words = text.split()
    normalized_words = [normalize_underscores(word) for word in words]
    return ' '.join(normalized_words)

# Function to replace words in the text with synonyms, hypernyms, or antonyms
def replace_words(text, words):
    replacements = {}

    # Înlocuirea cuvintelor cu sinonime, hiperonime sau antonime
    for word in words:
        synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)

        # Dacă avem sinonime sau antonime, alegem unul dintre ele aleator
        if random.random() < 0.2:  # 20% șansă
            if antonyms:  # Dacă avem antonime, le folosim
                replacements[word] = f"not {random.choice(antonyms)}"
            elif synonyms:  # Dacă avem sinonime, le folosim
                replacements[word] = random.choice(synonyms)
            elif hypernyms:  # Dacă avem hiperonime, le folosim
                replacements[word] = random.choice(hypernyms)

    # Înlocuirea cuvintelor în text cu sinonime/antonime/hiperonime
    replaced_text = ' '.join([replacements.get(word, word) for word in text.split()])

    return replaced_text


# def replace_words_with_all_variants(text, words):
#     replacements = {}
#
#     for word in words:
#         synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)
#
#         replacement_options = []
#
#         # Adăugăm antonimele cu "not" dacă sunt disponibile
#         if antonyms:
#             replacement_options.append(f"not {random.choice(antonyms)}")
#
#         # Adăugăm sinonimele dacă sunt disponibile
#         if synonyms:
#             replacement_options.extend(synonyms)
#
#         # Adăugăm hiperonimele dacă sunt disponibile
#         if hypernyms:
#             replacement_options.extend(hypernyms)
#
#         # Dacă avem variante de înlocuire, alegem toate
#         if replacement_options:
#             replacements[word] = replacement_options
#
#     # Creăm propoziții folosind toate variantele de înlocuire
#     new_texts = []
#     for word, replacement_list in replacements.items():
#         for replacement in replacement_list:
#             new_text = ' '.join([replacement if w == word else w for w in text.split()])
#             new_texts.append(new_text)
#
#     return new_texts


# def replace_words_with_all_variants(text, words, keywords, max_sentences=150):
#     replacements = {}
#
#     # Creăm un dicționar cu toate variantele posibile pentru fiecare cuvânt
#     for word in words:
#         if word in keywords:  # Înlocuim doar cuvintele cheie
#             synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)
#
#             # Adăugăm toate variantele (sinonime, antonime, hiperonime)
#             replacement_options = []
#
#             if antonyms:
#                 replacement_options.append(f"not {random.choice(antonyms)}")
#
#             if synonyms:
#                 replacement_options.extend(synonyms)
#
#             if hypernyms:
#                 replacement_options.extend(hypernyms)
#
#             if replacement_options:
#                 replacements[word] = replacement_options
#         else:
#             replacements[word] = [word]  # Lăsăm cuvintele non-cheie nemodificate
#
#     # Creăm combinațiile posibile
#     replacement_combinations = [[]]
#
#     # Creăm combinațiile prin bucle imbricate
#     for word in words:
#         word_replacements = replacements.get(word, [word])
#         new_combinations = []
#         for existing_combination in replacement_combinations:
#             for replacement in word_replacements:
#                 new_combinations.append(existing_combination + [replacement])
#         replacement_combinations = new_combinations
#
#     # Generăm propozițiile din combinațiile de înlocuiri
#     new_texts = []
#     for combination in replacement_combinations:
#         # Creăm propoziția folosind combinația curentă
#         new_text = ' '.join(combination)
#         new_texts.append(new_text)
#
#     # Filtrăm propozițiile pentru a păstra doar cele cu cele mai multe cuvinte înlocuite
#     def count_replacements(original_text, new_text):
#         # Comparăm cuvintele din textul original și textul înlocuit
#         original_words = original_text.split()
#         new_words = new_text.split()
#         replacements_count = sum(1 for o, n in zip(original_words, new_words) if o != n)
#         return replacements_count
#
#     # Sortăm propozițiile după numărul de cuvinte înlocuite
#     new_texts.sort(key=lambda t: count_replacements(text, t), reverse=True)
#
#     # Limita numărul de propoziții (dacă este necesar)
#     # Păstrează doar primele N propoziții cu cele mai multe înlocuiri
#     N = max_sentences  # Păstrezi doar primele max_sentences propoziții
#     new_texts = new_texts[:N]
#
#     # Dacă ai generat prea puține propoziții, adaugă aleatorii pentru a păstra diversitatea
#     if len(new_texts) < max_sentences:
#         additional_sentences_needed = max_sentences - len(new_texts)
#         random_sentences = random.sample(new_texts, len(new_texts))  # Aleatoriu din propozițiile deja generate
#         new_texts.extend(random_sentences[:additional_sentences_needed])
#
#     return new_texts

# def replace_words_with_all_variants(text, words, max_sentences=150):
#     replacements = {}
#
#     # Creăm un dicționar cu toate variantele posibile pentru fiecare cuvânt
#     for word in words:
#         synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)
#
#         # Adăugăm toate variantele (sinonime, antonime, hiperonime)
#         replacement_options = []
#
#         if antonyms:
#             replacement_options.append(f"not {random.choice(antonyms)}")
#
#         if synonyms:
#             replacement_options.extend(synonyms)
#
#         if hypernyms:
#             replacement_options.extend(hypernyms)
#
#         if replacement_options:
#             replacements[word] = replacement_options
#         else:
#             replacements[word] = [word]  # Lăsăm cuvintele non-cheie nemodificate
#
#     # Creăm combinațiile posibile
#     replacement_combinations = [[]]
#
#     # Creăm combinațiile prin bucle imbricate
#     for word in words:
#         word_replacements = replacements.get(word, [word])
#         new_combinations = []
#         for existing_combination in replacement_combinations:
#             for replacement in word_replacements:
#                 new_combinations.append(existing_combination + [replacement])
#         replacement_combinations = new_combinations
#
#     # Creăm o listă de dicționare cu înlocuirile pentru fiecare propoziție
#     sentence_replacements = []
#     for combination in replacement_combinations:
#         # Creăm propoziția folosind combinația curentă
#         new_text = ' '.join(combination)
#
#         # Creăm un dicționar pentru această propoziție
#         replacement_dict = {}
#
#         # Comparăm propoziția generată cu propoziția originală pentru a înlocui cuvintele
#         for original_word, new_word in zip(words, combination):
#             if original_word != new_word:  # Dacă cuvântul a fost înlocuit
#                 replacement_dict[original_word] = new_word
#
#         if replacement_dict:  # Adăugăm în dicționar doar dacă sunt înlocuiri
#             sentence_replacements.append(replacement_dict)
#
#     # Limita numărul de propoziții (dacă este necesar)
#     N = max_sentences  # Păstrezi doar primele max_sentences propoziții
#     sentence_replacements = sentence_replacements[:N]
#
#     # Dacă ai generat prea puține propoziții, adaugă aleatoriu pentru a păstra diversitatea
#     if len(sentence_replacements) < max_sentences:
#         additional_sentences_needed = max_sentences - len(sentence_replacements)
#         random_sentences = random.sample(sentence_replacements, len(sentence_replacements))  # Aleatoriu din propozițiile deja generate
#         sentence_replacements.extend(random_sentences[:additional_sentences_needed])
#
#     return sentence_replacements

def replace_words_20_percent(text, words, max_sentences=150):
    replacements = {}
    word_count = len(words)
    min_replaced = int(word_count * 0.20)  # 20% din cuvinte trebuie înlocuite
    replaced_count = 0

    # Creăm un dicționar cu toate variantele posibile pentru fiecare cuvânt
    for word in words:
        if replaced_count >= min_replaced:  # Dacă am înlocuit deja 20% din cuvinte, ieșim
            break

        synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)

        # Adăugăm toate variantele (sinonime, antonime, hiperonime) dacă există
        replacement_options = []
        if antonyms:
            replacement_options.append(f"not {random.choice(antonyms)}")
        if synonyms:
            replacement_options.extend(synonyms)
        if hypernyms:
            replacement_options.extend(hypernyms)

        if replacement_options:
            replacements[word] = replacement_options
            replaced_count += 1  # Incrementăm numărul de cuvinte înlocuite
        else:
            replacements[word] = [word]  # Lăsăm cuvintele fără înlocuiri nemodificate

    # Dacă am înlocuit deja 20% din cuvinte, începem să generăm propozițiile
    replacement_combinations = [[]]
    for word in words:
        word_replacements = replacements.get(word, [word])
        new_combinations = []
        for existing_combination in replacement_combinations:
            for replacement in word_replacements:
                new_combinations.append(existing_combination + [replacement])
        replacement_combinations = new_combinations

    # Creăm o listă de propoziții generate și verificăm procentul de cuvinte înlocuite
    sentence_replacements = []
    for combination in replacement_combinations:
        new_text = ' '.join(combination)
        # Numărăm cuvintele înlocuite
        replacement_dict = {}
        for original_word, new_word in zip(words, combination):
            if original_word != new_word:
                replacement_dict[original_word] = new_word
                replaced_count += 1

        # Adăugăm propoziția doar dacă am înlocuit cel puțin 20% din cuvintele originale
        if replacement_dict and replaced_count >= min_replaced:
            sentence_replacements.append(replacement_dict)

    # Limita numărul de propoziții
    N = max_sentences
    sentence_replacements = sentence_replacements[:N]

    # Dacă nu am ajuns la numărul de propoziții dorit, adăugăm propoziții suplimentare aleatorii
    if len(sentence_replacements) < max_sentences:
        additional_sentences_needed = max_sentences - len(sentence_replacements)
        random_sentences = random.sample(sentence_replacements, len(sentence_replacements))  # Aleatoriu
        sentence_replacements.extend(random_sentences[:additional_sentences_needed])

    return sentence_replacements


def replace_words_keywords_only(text, words, keywords, max_sentences=150):
    replacements = {}

    # Creăm un dicționar cu toate variantele posibile pentru fiecare cuvânt
    for word in words:
        if word in keywords:
            synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(word)

            # Adăugăm toate variantele (sinonime, antonime, hiperonime)
            replacement_options = []

            if antonyms:
                replacement_options.append(f"not {random.choice(antonyms)}")

            if synonyms:
                replacement_options.extend(synonyms)

            if hypernyms:
                replacement_options.extend(hypernyms)

            if replacement_options:
                replacements[word] = replacement_options
            else:
                replacements[word] = [word]  # Lăsăm cuvintele non-cheie nemodificate

    # Creăm combinațiile posibile
    replacement_combinations = [[]]

    # Creăm combinațiile prin bucle imbricate
    for word in words:
        word_replacements = replacements.get(word, [word])
        new_combinations = []
        for existing_combination in replacement_combinations:
            for replacement in word_replacements:
                new_combinations.append(existing_combination + [replacement])
        replacement_combinations = new_combinations

    # Creăm o listă de dicționare cu înlocuirile pentru fiecare propoziție
    sentence_replacements = []
    for combination in replacement_combinations:
        # Creăm propoziția folosind combinația curentă
        new_text = ' '.join(combination)

        # Creăm un dicționar pentru această propoziție
        replacement_dict = {}

        # Comparăm propoziția generată cu propoziția originală pentru a înlocui cuvintele
        for original_word, new_word in zip(words, combination):
            if original_word != new_word:  # Dacă cuvântul a fost înlocuit
                replacement_dict[original_word] = new_word

        if replacement_dict:  # Adăugăm în dicționar doar dacă sunt înlocuiri
            sentence_replacements.append(replacement_dict)

    # Limita numărul de propoziții (dacă este necesar)
    N = max_sentences  # Păstrezi doar primele max_sentences propoziții
    sentence_replacements = sentence_replacements[:N]

    # Dacă ai generat prea puține propoziții, adaugă aleatoriu pentru a păstra diversitatea
    if len(sentence_replacements) < max_sentences:
        additional_sentences_needed = max_sentences - len(sentence_replacements)
        random_sentences = random.sample(sentence_replacements, len(sentence_replacements))  # Aleatoriu din propozițiile deja generate
        sentence_replacements.extend(random_sentences[:additional_sentences_needed])

    return sentence_replacements

def apply_replacements_to_text(text, replacements):
    # Împărțim textul astfel încât să păstrăm cuvintele și semnele de punctuație
    words_with_punctuation = re.findall(r'\b\w+\b|[^\w\s]', text)

    new_words = []
    for word in words_with_punctuation:
        # Verificăm dacă cuvântul este în dicționarul de înlocuiri
        word_lower = word.lower()
        if word_lower in replacements:
            new_word = replacements[word_lower]
            # Dacă cuvântul original începe cu literă mare, aplicăm și capitalizarea
            if word[0].isupper():
                new_word = new_word.capitalize()
            new_words.append(new_word)
        else:
            new_words.append(word)

    # Reconstruim textul fără a adăuga spații suplimentare între semnele de punctuație și cuvinte
    new_text = ''.join([f' {word}' if word not in ',.?!;:' else word for word in new_words]).strip()

    # Corectăm capitalizarea propozițiilor
    new_text = re.sub(r'(^|\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), new_text)

    return new_text


def generate_texts_from_replacements(original_text, replacements_list):
    # Generăm toate variantele de text pe baza fiecărui dicționar de înlocuiri
    generated_texts = []
    for replacements in replacements_list:
        new_text = apply_replacements_to_text(original_text, replacements)
        generated_texts.append(new_text)

    return generated_texts

def extract_keywords(text, top_n=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:top_n]
    return keywords

# # Function to generate sentences from keywords
# def generate_sentences_from_keywords(keywords, text):
#     sentences = sent_tokenize(text)
#     keyword_sentences = []
#
#     for keyword in keywords:
#         for sentence in sentences:
#             if keyword in sentence:
#                 keyword_sentences.append(sentence)
#                 break
#         else:
#             keyword_sentences.append(f"The concept of '{keyword}' is significant in this context.")
#
#     return keyword_sentences

def generate_sentences_from_keywords(keywords, text):
    sentences = sent_tokenize(text)
    keyword_sentences = []

    for keyword in keywords:
        found = False
        # Căutăm propoziția care conține cuvântul cheie
        for sentence in sentences:
            if keyword in sentence:
                keyword_sentences.append(sentence)
                found = True
                break

        # Dacă nu găsim propoziția care conține cuvântul cheie, o generăm
        if not found:
            # Extragem sinonime pentru cuvântul cheie
            synonyms, hypernyms, antonyms = get_synonyms_hypernyms_antonyms_en(keyword)
            new_sentence = f"The concept of '{keyword}' is important in this context."

            if synonyms:
                new_sentence = f"The synonym for '{keyword}' is '{random.choice(synonyms)}'."
            elif hypernyms:
                new_sentence = f"A more general term for '{keyword}' could be '{random.choice(hypernyms)}'."
            elif antonyms:
                new_sentence = f"The opposite of '{keyword}' could be '{random.choice(antonyms)}'."

            keyword_sentences.append(new_sentence)

    return keyword_sentences

def apply_replacements(text, replacements):
    words = text.split()
    new_text = ' '.join([replacements.get(word, word) for word in words])
    return new_text

def write_to_file(file_path, original_text, keywords, new_texts):
    with open(file_path, 'w', encoding='utf-8') as file:
        for new_text in new_texts:
            file.write(f"Input: {original_text}\n")  # Scrie propoziția originală
            file.write(f"Keywords: {', '.join(keywords)}\n")  # Scrie cuvintele cheie
            file.write(f"Output: {new_text}\n\n")  # Scrie propoziția generată

# Funcție pentru procesarea întregului task
def nlp_task():
    text = read_text() # citește textul

    if detect_language(text) == 'ro':
        # Procesare text în limba română
        print("Procesare text în limba română...")
        words = analyze_stylometry(text)


        # Generare propoziții noi cu toate variantele de înlocuire
        keywords = extract_keywords(text)
        new_texts = replace_words_20_percent(text, words)
        print("Propozițiile generate cu sinonime:")
        for new_text in new_texts:
            print(normalize_text(new_text))

        #new_text = replace_words(text, words)
        #print("Textul modificat cu sinonime:")
        #print(new_text)
        #new_text = apply_replacements(text, new_words)
        #print(new_text)
    elif detect_language(text) == 'en':
        # Procesare text în limba engleză
        print("Procesare text în limba engleză...")

        words = analyze_stylometry(text)
        #new_text = replace_words(text, words)

        keywords = extract_keywords(text)
        print("Keywords:", keywords)

        print("Propozițiile generate cu sinonime:")
        replacements_list = replace_words_20_percent(text, words)
        new_texts = generate_texts_from_replacements(text, replacements_list)

        for new_text in new_texts:
            print(new_text)

        write_to_file("training_data.txt", text, keywords, new_texts)


        #print(new_text)

        # print("Propozitiile generate din cuvinte cheie:")
        # keyword_sentences = generate_sentences_from_keywords(keywords, text)
        # print(keyword_sentences)
    else:
        # Procesare text în altă limbă
        print("Textul este scris într-o limbă nesuportată.")
        return

# Apelarea funcției pentru procesarea întregului task
nlp_task()
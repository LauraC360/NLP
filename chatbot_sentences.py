from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from nltk import word_tokenize

# Încarcă modelul și tokenizer-ul
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")


# Funcție pentru a verifica structura gramaticală simplă
def is_grammatical(sentence):
    tokens = word_tokenize(sentence)
    return len(tokens) > 3 and any(word.isalpha() for word in tokens)


# Funcție pentru a genera propoziții valide (prima metodă)
def generate_valid_sentence_with_keyword_method_1(keyword, max_attempts=20):
    for attempt in range(max_attempts):
        # Generează text
        input_ids = tokenizer.encode(keyword, return_tensors="pt")
        generated_output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.92,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decodează textul generat
        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        # Extrage propozițiile din text
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_text)

        # Găsește prima propoziție validă
        for sentence in sentences:
            if (
                    sentence.strip().endswith(".")
                    and keyword.lower() in sentence.lower()
                    and sentence[0].isupper()
            ):
                return sentence.strip()

    return f"Nu s-a putut genera o propoziție validă pentru cuvântul cheie '{keyword}' după {max_attempts} încercări."


# Funcție pentru a genera propoziții valide (a doua metodă)
def generate_sentence_with_keyword_method_2(keyword, max_attempts=5):
    prompt = f"Generate a grammatically correct sentence containing the word '{keyword}':"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for attempt in range(max_attempts):
        # Generează textul
        generated_output = model.generate(input_ids,
                                          max_length=50,
                                          num_return_sequences=1,
                                          no_repeat_ngram_size=2,
                                          top_p=0.8,
                                          top_k=30,
                                          temperature=0.5)
        # Decodează textul generat
        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        # Extrage textul relevant
        output_section = generated_text.split("Output:")[-1].strip()

        # Împarte textul în propoziții și verifică dacă sunt gramaticale
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', output_section)
        valid_sentences = [s for s in sentences if keyword.lower() in s.lower() and is_grammatical(s)]

        # Returnează prima propoziție validă
        if valid_sentences:
            return valid_sentences[0]

    # Dacă nu se găsește o propoziție validă
    return f"Failed to generate a valid sentence with the keyword '{keyword}' after {max_attempts} attempts."


# Listele de cuvinte cheie
keywords = ["party", "nice", "house", "together"]

# Deschide fișierul pentru a scrie propozițiile
with open("generated_sentences_combined.txt", "w") as file:
    for keyword in keywords:
        # Generăm propoziții folosind ambele metode
        sentence_method_1 = generate_valid_sentence_with_keyword_method_1(keyword)
        sentence_method_2 = generate_sentence_with_keyword_method_2(keyword)

        # Scrie rezultatele în fișier
        file.write(f"Generated sentence for keyword '{keyword}' (Method 1):\n{sentence_method_1}\n")
        file.write(f"Generated sentence for keyword '{keyword}' (Method 2):\n{sentence_method_2}\n\n")

print("Propozițiile au fost salvate în fișierul 'generated_sentences_combined.txt'.")

from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from nltk import word_tokenize

# Încarcă modelul și tokenizer-ul antrenat
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")

# Funcție pentru a verifica structura gramaticală simplă
def is_grammatical(sentence):
    tokens = word_tokenize(sentence)
    return len(tokens) > 3 and any(word.isalpha() for word in tokens)

# Funcție pentru a genera propoziții valide
def generate_sentence_with_keyword(keyword, max_attempts=5):
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

# Generează propoziții pentru fiecare cuvânt cheie
generated_sentences = []
for keyword in keywords:
    sentence = generate_sentence_with_keyword(keyword)
    generated_sentences.append(sentence)

# Afișează propozițiile generate
for i, sentence in enumerate(generated_sentences):
    print(f"Generated sentence for keyword '{keywords[i]}':\n{sentence}\n")
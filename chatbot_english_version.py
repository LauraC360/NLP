from transformers import AutoTokenizer, AutoModelForCausalLM

# Încarcă tokenizer-ul și modelul fine-tunat
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")


# Funcție pentru a genera propoziții pe baza unui cuvânt cheie
def generate_sentence_with_keyword(input_text, keyword):
    prompt = f"Input: {input_text} Keywords: {keyword} Output:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generează textul
    generated_output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decodează și returnează propoziția generată
    return tokenizer.decode(generated_output[0], skip_special_tokens=True)


# Exemplu de text
input_text = ""

# Listele de cuvinte cheie
keywords = ["grandma", "small", "ready", "old", "house"]

# Generează propoziții pentru fiecare cuvânt cheie
generated_sentences = []
for keyword in keywords:
    sentence = generate_sentence_with_keyword(input_text, keyword)
    generated_sentences.append(sentence)

# Afișează propozițiile generate
for i, sentence in enumerate(generated_sentences):
    print(f"Generated sentence for keyword '{keywords[i]}':\n{sentence}\n")

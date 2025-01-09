# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# # Încarcă modelul și tokenizer-ul
# tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
# model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")
#
# # Încarcă propozițiile din fișier
# def load_sentences(file_path):
#     with open(file_path, "r", encoding="utf-8") as file:
#         sentences = [line.strip() for line in file if line.strip()]
#     return sentences
#
# # Funcție pentru a genera propoziții cu cuvinte cheie
# def generate_sentence_with_keyword(keyword, max_attempts=5):
#     for attempt in range(max_attempts):
#         # Generează text
#         input_ids = tokenizer.encode(keyword, return_tensors="pt")
#         generated_output = model.generate(
#             input_ids,
#             max_length=50,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2,
#             top_p=0.92,
#             top_k=50,
#             temperature=0.7,
#         )
#         # Decodează textul generat
#         generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
#         # Filtrează propozițiile coerente care conțin cuvântul cheie
#         if keyword.lower() in generated_text.lower():
#             return generated_text
#
#     return f"Nu s-a putut genera o propoziție validă pentru cuvântul cheie '{keyword}'."
#
# # Exemplu de utilizare
# file_path = "training_data.txt"
# sentences = load_sentences(file_path)
# keywords = ["party", "nice", "house", "together"]
#
# for keyword in keywords:
#     generated_sentence = generate_sentence_with_keyword(keyword)
#     print(f"Generated sentence for keyword '{keyword}': {generated_sentence}")


from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Încarcă modelul și tokenizer-ul
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")

# Funcție pentru a genera propoziții și a opri la prima propoziție validă
def generate_valid_sentence_with_keyword(keyword, max_attempts=20):
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

keywords = ["party", "nice", "house", "together"]

for keyword in keywords:
    generated_sentence = generate_valid_sentence_with_keyword(keyword)
    print(f"Generated sentence for keyword '{keyword}': {generated_sentence}")



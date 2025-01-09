from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Citirea datelor de antrenament dintr-un fișier txt
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split("\n")  # Împarte datele în propoziții (enunțuri)

    training_data = []
    for item in data:
        # Salvează fiecare propoziție în lista de date
        training_data.append({
            "text": item.strip()
        })

    return training_data

# Pregătește datele pentru antrenament
training_data = load_data("training_data.txt")

# Creează un dataset
train_dataset = Dataset.from_list(training_data)


def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # Ajustează lungimea maximă
        return_tensors="pt"
    )
    # Copiază input_ids în labels
    inputs["labels"] = inputs["input_ids"].clone()
    # Ignoră pierderea pentru tokenii de padding
    inputs["labels"][inputs["input_ids"] == tokenizer.pad_token_id] = -100
    return inputs


train_dataset = train_dataset.map(tokenize_function, batched=True)

# Configurarea antrenamentului
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
    report_to="none"
)

# Folosirea trainerului pentru a antrena modelul
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# Antrenarea modelului
trainer.train()

# Salvarea modelului antrenat
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import jsonlines

# Load dataset
data = []
with jsonlines.open("../data/qa_dataset.jsonl") as reader:
    for obj in reader:
        instruction = str(obj.get("instruction", "")).strip()
        output = str(obj.get("output", "")).strip()
        combined_text = instruction + " " + output
        data.append({
            "text": combined_text,
            "label": 1  # Modify this if you have classification labels
        })

dataset = Dataset.from_list(data)

# Tokenizer & Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess function
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# Split into train and test
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/legal_qa_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save model
model.save_pretrained("../models/legal_qa_model")
tokenizer.save_pretrained("../models/legal_qa_model")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import jsonlines
import os

DATA_PATH = "../data/qa_dataset.jsonl"
MODEL_DIR = "../models/legal_qa_model"
MODEL_NAME = "distilbert-base-uncased"

def load_dataset(data_path):
    data = []
    try:
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                instruction = str(obj.get("instruction", "")).strip()
                output = str(obj.get("output", "")).strip()
                combined_text = instruction + " " + output
                data.append({
                    "text": combined_text,
                    "label": 1  # Default label, can be extended later
                })
        return Dataset.from_list(data)
    except FileNotFoundError:
        print(f"❌ Dataset not found at {data_path}")
        return None

def train_model():
    # ✅ Skip training if model already exists
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"✅ Model already exists at '{MODEL_DIR}'. Skipping training.")
        return

    dataset = load_dataset(DATA_PATH)
    if dataset is None:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(preprocess, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("🚀 Starting training...")
    trainer.train()

    # Save model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Trained model saved to: {MODEL_DIR}")

def main():
    train_model()

if __name__ == "__main__":
    main()

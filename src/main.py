import json
import random
import jsonlines
import os

DATA_PATH = "../data/extracted_texts.json"
OUTPUT_PATH = "../data/qa_dataset.jsonl"

def build_qa_dataset(input_path=DATA_PATH, output_path=OUTPUT_PATH):
    # Check if QA dataset already exists
    if os.path.exists(output_path):
        print(f"✅ QA dataset already exists at '{output_path}'. Skipping generation.")
        return

    # Load your extracted legal document text
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_path}")
        return

    # Build QA-style dataset
    qa_dataset = []
    for item in data:
        text = item.get("text", "")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            continue
        question = random.choice(lines)
        answer = random.choice(lines)
        qa_dataset.append({
            "instruction": question,
            "input": "",
            "output": answer
        })

    # Save dataset
    with jsonlines.open(output_path, mode='w') as writer:
        for entry in qa_dataset:
            writer.write(entry)

    print(f"✅ Saved QA dataset for fine-tuning at: {output_path}")

def main():
    build_qa_dataset()

if __name__ == "__main__":
    main()

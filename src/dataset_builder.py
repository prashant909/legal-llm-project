import json
import random
import jsonlines

# Load your extracted legal document text
with open("../data/extracted_texts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Build QA-style dataset
qa_dataset = []
for item in data:
    text = item["text"]
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
output_path = "../data/qa_dataset.jsonl"
with jsonlines.open(output_path, mode='w') as writer:
    for entry in qa_dataset:
        writer.write(entry)

print(f"✅ Saved QA dataset for fine-tuning: {output_path}")
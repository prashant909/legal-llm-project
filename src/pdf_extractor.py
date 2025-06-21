import os
import fitz
import json

folder_path = "../data/legislative_documents"
all_texts = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        print(f"🔍 Reading: {filename}")
        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    try:
                        text += page.get_text()
                    except Exception as e:
                        print(f"⚠️ Error extracting page from {filename}: {e}")
                all_texts.append({"file": filename, "text": text})
        except Exception as e:
            print(f"❌ Failed to open PDF {filename}: {e}")

# Save output
output_path = "../data/extracted_texts.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_texts, f, ensure_ascii=False, indent=2)

print(f"\n✅ All extracted text saved to: {output_path}")
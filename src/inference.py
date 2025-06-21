from transformers import pipeline

# Load trained model
qa_pipeline = pipeline("question-answering", model="../models/legal_qa_model")

def ask_query(question):
    result = qa_pipeline(question=question, context="India's Constitution outlines fundamental rights.")
    print(f"Q: {question}")
    print(f"A: {result['answer']}")

if __name__ == "__main__":
    ask_query("What are the fundamental rights?")
    ask_query("Explain Article 14 of the Indian Constitution.")
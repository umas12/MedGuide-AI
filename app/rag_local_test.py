# app/rag_local_test.py

from rag_pipeline import rag_qa_guarded

def main():
    print("🩺 Welcome to MedGuide-AI (Local QA Mode)")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        question = input("🧠 Your question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting. Take care!")
            break

        result = rag_qa_guarded(question)
        print("\n🤖 Answer:", result["answer"])
        print("📊 Confidence:", result.get("score", "N/A"))
        print("📚 Sources:", result.get("sources", []))
        print("-" * 50)

if __name__ == "__main__":
    main()

import sys
import time

from rag_engine import RAGEngine
from bot import build_prompt


def cmd_index(force: bool = False):
    engine = RAGEngine()
    print("Indexing knowledge base...")
    engine.index(force=force)
    print(f"Done. Collection has {engine.collection.count()} chunks.")


def cmd_query(text: str):
    engine = RAGEngine()
    context, docs = engine.query(text)
    print(f"--- Query ---\n{text}\n")
    print(f"--- Context ({len(docs)} chunks) ---")
    for i, d in enumerate(docs):
        print(f"\n[{i+1}] {d[:200]}...")
    print(f"\n--- Answer ---")
    prompt = build_prompt(text, context)
    answer = engine.query_ollama(prompt)
    print(answer)


def cmd_interactive():
    engine = RAGEngine()
    if engine.collection is None or engine.collection.count() == 0:
        print("Indexing knowledge base first...")
        engine.index()
        print(f"Indexed {engine.collection.count()} chunks.")
    print("\nCustomerBot RAG — type 'quit' to exit, 'reindex' to rebuild index\n")
    history = []
    while True:
        try:
            query = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "reindex":
            engine.index(force=True)
            print("Reindexed.")
            continue
        context, _ = engine.query(query)
        prompt = build_prompt(query, context, history=history)
        answer = engine.query_ollama(prompt)
        print(f"\nAgent: {answer}\n")
        history.append(f"Customer: {query}")
        history.append(f"Agent: {answer}")


def main():
    if len(sys.argv) < 2:
        cmd_interactive()
        return
    command = sys.argv[1]
    if command == "index":
        force = "--force" in sys.argv or "-f" in sys.argv
        cmd_index(force=force)
    elif command == "query" and len(sys.argv) >= 3:
        cmd_query(" ".join(sys.argv[2:]))
    elif command == "interactive":
        cmd_interactive()
    else:
        print("Usage:")
        print("  python main.py                     Interactive mode")
        print("  python main.py interactive         Interactive mode")
        print("  python main.py index [--force]     Index/reindex knowledge base")
        print('  python main.py query "your question"   Single query')
        sys.exit(1)


if __name__ == "__main__":
    main()

import json
import requests
import numpy as np
import faiss
from typing import List, Dict
from datetime import datetime

# Models and API config
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434"

# Load messages from JSON
def load_messages(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Get embedding for text using Ollama API
def get_embedding(text: str, model: str = EMBED_MODEL) -> List[float]:
    url = f"{OLLAMA_URL}/api/embeddings"
    response = requests.post(url, json={"model": model, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

# Embed messages (cache to avoid repeated calls)
def embed_messages(messages: List[Dict], model: str = EMBED_MODEL) -> np.ndarray:
    print("[INFO] Embedding messages...")
    return np.array([get_embedding(msg["content"], model) for msg in messages], dtype='float32')

# Create FAISS index
def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Semantic filter by embedding similarity to filter_text
def semantic_filter(messages: List[Dict], index: faiss.IndexFlatL2, filter_text: str, top_k: int = 20) -> List[Dict]:
    filter_vec = np.array([get_embedding(filter_text)], dtype='float32')
    distances, indices = index.search(filter_vec, top_k)
    filtered_msgs = [messages[i] for i in indices[0] if i < len(messages)]
    return filtered_msgs

# Semantic search within a subset of messages
def semantic_search(messages: List[Dict], query: str, model: str = EMBED_MODEL, top_k: int = 5) -> List[Dict]:
    if not messages:
        print("[INFO] No messages to search.")
        return []
    embeddings = embed_messages(messages, model)
    index = create_faiss_index(embeddings)
    query_vec = np.array([get_embedding(query, model)], dtype='float32')
    distances, indices = index.search(query_vec, top_k)
    return [messages[i] for i in indices[0]]

# Exact filter by sender
def filter_by_sender(messages: List[Dict], sender: str) -> List[Dict]:
    return [m for m in messages if m["sender"].lower() == sender.lower()]

# Exact filter by category keyword in subject
def filter_by_category(messages: List[Dict], category_keyword: str) -> List[Dict]:
    return [m for m in messages if category_keyword.lower() in m.get("subject", "").lower()]

# Exact filter by date range (format YYYY-MM-DD)
def filter_by_date_range(messages: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        print("[ERROR] Invalid date format. Use YYYY-MM-DD.")
        return []
    filtered = []
    for m in messages:
        date_str = m.get("date")
        if not date_str:
            continue
        try:
            msg_dt = datetime.strptime(date_str, "%Y-%m-%d")
            if start_dt <= msg_dt <= end_dt:
                filtered.append(m)
        except:
            continue
    return filtered

# Summarize messages with granularity options
def summarize_messages(messages: List[Dict], granularity: str = "detailed", model: str = CHAT_MODEL) -> str:
    if not messages:
        return "No messages to summarize."
    combined_text = "\n".join([f"From {m['sender']} on {m.get('date', 'unknown date')}: {m['content']}" for m in messages])
    
    if granularity == "short":
        prompt = f"Provide a brief summary of the following messages:\n{combined_text}"
    elif granularity == "bullet":
        prompt = f"Summarize the following messages as concise bullet points:\n{combined_text}"
    else:  # detailed
        prompt = f"Provide a detailed summary of the following messages:\n{combined_text}"
    
    url = f"{OLLAMA_URL}/api/chat"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    response = requests.post(url, json=data, stream=True)
    response.raise_for_status()
    
    full_text = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode('utf-8'))
            if "message" in chunk and "content" in chunk["message"]:
                full_text += chunk["message"]["content"]
    return full_text

# Print message nicely
def print_message(msg: Dict):
    status = "READ" if msg.get("read") else "UNREAD"
    print(f"[{status}] {msg['sender']} - {msg.get('subject', '')} ({msg.get('date', 'no date')}): {msg['content']}")

def main():
    path = "messages.json"
    messages = load_messages(path)
    print("[INFO] Loaded messages")

    # Embed all messages ONCE here for semantic filtering/searching
    all_embeddings = embed_messages(messages)
    all_index = create_faiss_index(all_embeddings)
    print("[INFO] FAISS index created")

    while True:
        print("\nOptions:")
        print("1. Semantic filter by text (sender/category/date or other)")
        print("2. Exact filter by sender")
        print("3. Exact filter by category (subject keyword)")
        print("4. Exact filter by date range")
        print("5. Summarize messages")
        print("6. Exit")

        choice = input("Enter your choice: ").strip()

        filtered_msgs = []
        if choice == "1":
            filter_text = input("Enter semantic filter text (sender/category/date or any): ").strip()
            filtered_msgs = semantic_filter(messages, all_index, filter_text, top_k=20)
            print(f"\n[INFO] {len(filtered_msgs)} messages matched semantically with '{filter_text}'.")
            if not filtered_msgs:
                print("No messages matched.")
                continue
            
            # Semantic search within filtered subset
            query = input("Enter semantic search query within filtered messages (leave blank to skip): ").strip()
            if query:
                search_results = semantic_search(filtered_msgs, query)
                print("\nSemantic Search Results:")
                for msg in search_results:
                    print_message(msg)
                filtered_msgs = search_results
            else:
                print("\nFiltered Messages (top 10):")
                for msg in filtered_msgs[:10]:
                    print_message(msg)

        elif choice == "2":
            sender = input("Enter sender name: ").strip()
            filtered_msgs = filter_by_sender(messages, sender)
            if not filtered_msgs:
                print("No messages found from that sender.")
                continue
            print(f"\n[INFO] {len(filtered_msgs)} messages from sender '{sender}'.")
            for msg in filtered_msgs[:10]:
                print_message(msg)

        elif choice == "3":
            category = input("Enter category keyword (subject): ").strip()
            filtered_msgs = filter_by_category(messages, category)
            if not filtered_msgs:
                print("No messages found in that category.")
                continue
            print(f"\n[INFO] {len(filtered_msgs)} messages in category '{category}'.")
            for msg in filtered_msgs[:10]:
                print_message(msg)

        elif choice == "4":
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            filtered_msgs = filter_by_date_range(messages, start_date, end_date)
            if not filtered_msgs:
                print("No messages found in that date range.")
                continue
            print(f"\n[INFO] {len(filtered_msgs)} messages from {start_date} to {end_date}.")
            for msg in filtered_msgs[:10]:
                print_message(msg)

        elif choice == "5":
            print("Summarize messages options:")
            print("a. Summarize ALL messages")
            print("b. Summarize by sender")
            print("c. Summarize by category (subject keyword)")
            print("d. Summarize by date range")
            sub_choice = input("Choose option (a/b/c/d): ").strip().lower()

            if sub_choice == "a":
                filtered_msgs = messages
            elif sub_choice == "b":
                sender = input("Enter sender name: ").strip()
                filtered_msgs = filter_by_sender(messages, sender)
                if not filtered_msgs:
                    print("No messages found from that sender.")
                    continue
            elif sub_choice == "c":
                category = input("Enter category keyword (subject): ").strip()
                filtered_msgs = filter_by_category(messages, category)
                if not filtered_msgs:
                    print("No messages found in that category.")
                    continue
            elif sub_choice == "d":
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                filtered_msgs = filter_by_date_range(messages, start_date, end_date)
                if not filtered_msgs:
                    print("No messages found in that date range.")
                    continue
            else:
                print("Invalid choice.")
                continue

            print("\nChoose summary granularity:")
            print("1. Detailed")
            print("2. Short")
            print("3. Bullet points")
            gran_choice = input("Enter your choice: ").strip()
            gran_map = {"1": "detailed", "2": "short", "3": "bullet"}
            granularity = gran_map.get(gran_choice, "detailed")

            summary = summarize_messages(filtered_msgs, granularity)
            print("\nSummary:\n", summary)

            continue

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")
            continue

        # After filtering (except summarization), optionally ask for summarization
        if filtered_msgs:
            ask_sum = input("\nWould you like to summarize these filtered messages? (y/n): ").strip().lower()
            if ask_sum == 'y':
                print("\nChoose summary granularity:")
                print("1. Detailed")
                print("2. Short")
                print("3. Bullet points")
                gran_choice = input("Enter your choice: ").strip()
                gran_map = {"1": "detailed", "2": "short", "3": "bullet"}
                granularity = gran_map.get(gran_choice, "detailed")

                summary = summarize_messages(filtered_msgs, granularity)
                print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()

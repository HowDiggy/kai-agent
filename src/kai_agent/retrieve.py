import argparse
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
COLLECTION_NAME = "syslogs"
MODEL_NAME = "all-MiniLM-L6-v2"

def search_logs(query: str, top_k: int = 5):
    # 1. Connect
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(COLLECTION_NAME)
    collection.load() # Load vectors into memory for speed

    # 2. Embed the Query
    # We must use the EXACT same model we used for ingestion
    print(f"🧠 Embedding query: '{query}'...")
    embedder = SentenceTransformer(MODEL_NAME)
    query_vector = embedder.encode([query])

    # 3. Search (Vector Similarity)
    search_params = {
        "metric_type": "L2", 
        "params": {"nprobe": 10}, 
    }
    
    print(f"🔍 Searching {collection.num_entities} logs...")
    results = collection.search(
        data=query_vector, 
        anns_field="embedding", 
        param=search_params, 
        limit=top_k, 
        output_fields=["timestamp", "user", "src_ip", "event_type", "raw_text"]
    )

    # 4. Display Results
    print(f"\n--- Found {len(results[0])} matches ---")
    for hit in results[0]:
        # The 'distance' is how different they are (Lower L2 distance = Better match)
        score = hit.distance
        data = hit.entity
        
        print(f"[{score:.4f}] {data.get('timestamp')} | {data.get('event_type')}")
        print(f"    User: {data.get('user')} | IP: {data.get('src_ip')}")
        print(f"    Raw:  {data.get('raw_text')[:100]}...")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search for Logs")
    parser.add_argument("query", type=str, help="The natural language query (e.g., 'failed logins')")
    args = parser.parse_args()

    search_logs(args.query)
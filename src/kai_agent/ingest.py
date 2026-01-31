import asyncio
import httpx
import json
from pathlib import Path
from pydantic import BaseModel, Field
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
VLLM_URL = "http://localhost:8005/v1/completions"
MODEL_NAME = "kai-log-parser"
LOG_FILE = Path("data/system.log")
COLLECTION_NAME = "syslogs"

# --- INIT RESOURCES ---
print("Loading Embedding Model...")
# This runs locally on CPU (or GPU if available) to turn text into numbers
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")
collection = Collection(COLLECTION_NAME)

# --- SCHEMA ---
class SyslogEntry(BaseModel):
    timestamp: str = Field(description="e.g. 'Jan 28 12:00:00'")
    hostname: str
    process: str
    pid: int
    event_type: str
    user: str
    src_ip: str
    port: int

alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Parse the following syslog entry into structured JSON.
Example Input: Jan 28 12:00:00 firewall-1 sshd[1234]: Failed password for invalid user admin from 1.2.3.4 port 5555 ssh2
Example Response: {{"timestamp": "Jan 28 12:00:00", "hostname": "firewall-1", "process": "sshd", "pid": 1234, "event_type": "authentication_failure", "user": "admin", "src_ip": "1.2.3.4", "port": 5555}}

### Input:
{}

### Response:
"""

async def parse_and_store(client: httpx.AsyncClient, line: str):
    if not line.strip(): return

    # 1. PARSE (Ask vLLM for Structure)
    json_schema = SyslogEntry.model_json_schema()
    try:
        response = await client.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "prompt": alpaca_template.format(line),
                "max_tokens": 200,
                "temperature": 0.0,
                "guided_json": json_schema
            },
            timeout=30.0
        )
        response.raise_for_status()
        parsed_data = json.loads(response.json()['choices'][0]['text'])
    except Exception as e:
        print(f"❌ Parse Error: {e}")
        return

    # 2. EMBED (Ask SentenceTransformer for Meaning)
    # We embed the raw line to capture the full context
    vector = embedder.encode(line).tolist()

    # 3. STORE (Save to Milvus)
    # Milvus inserts are column-based: [[vec], [time], [host]...]
    insert_data = [
        [vector],           
        [parsed_data.get("timestamp", "unknown")],
        [parsed_data.get("hostname", "unknown")],
        [parsed_data.get("event_type", "unknown")],
        [parsed_data.get("user", "unknown")],
        [parsed_data.get("src_ip", "unknown")],
        [line]              
    ]
    
    collection.insert(insert_data)
    print(f"✅ Stored: {parsed_data['user']} @ {parsed_data['src_ip']}")

async def main():
    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found!")
        return

    print(f"Reading logs from {LOG_FILE}...")
    
    async with httpx.AsyncClient() as client:
        with open(LOG_FILE, "r") as f:
            for line in f:
                await parse_and_store(client, line.strip())
    
    # Commit to disk
    collection.flush()
    print("🚀 Ingestion Complete. Data flushed to Milvus.")

if __name__ == "__main__":
    asyncio.run(main())
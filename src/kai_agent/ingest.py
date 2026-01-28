import re
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from src.kai_agent.legacy_log_db import LogDatabase  # Import your Milvus adapter

class LogIngestor:
    """
    Ingests raw syslog files and converts them into semantic embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize with a lightweight embedding model.
        
        Args:
            model_name: HuggingFace model ID. 'all-MiniLM-L6-v2' is efficient for CPU.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.logs: List[Dict[str, Any]] = []
        self.embeddings: torch.Tensor = None

    def parse_file(self, file_path: str) -> int:
        """
        Parses an OPNsense/BSD syslog file (Tab-separated).
        
        Format: Timestamp <tab> Level <tab> Process <tab> Message
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")

        print(f"Parsing {file_path}...")
        parsed_count = 0
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue # Skip malformed lines
                
                # Extract structured fields
                timestamp = parts[0]
                level = parts[1]
                process = parts[2]
                message = parts[3]

                # We clean the message to remove variable noise (like PIDs [123])
                # This helps the embedding model focus on the *type* of event.
                clean_msg = re.sub(r'\[\d+\]', '', message).strip()

                self.logs.append({
                    "timestamp": timestamp,
                    "level": level,
                    "process": process,
                    "original_message": message,
                    "clean_message": clean_msg
                })
                parsed_count += 1
        
        print(f"Successfully parsed {parsed_count} logs.")
        return parsed_count

    def embed_logs(self) -> None:
        """
        Converts the 'clean_message' of all parsed logs into vectors.
        """
        if not self.logs:
            print("No logs to embed.")
            return

        print("Generating embeddings (this may take a moment)...")
        messages = [log["clean_message"] for log in self.logs]
        
        # This returns a Tensor of shape (num_logs, 384)
        self.embeddings = self.model.encode(messages, convert_to_tensor=True)
        print(f"Created embeddings with shape: {self.embeddings.shape}")

if __name__ == "__main__":
    # 1. Initialize Ingestor & Database Connection
    ingestor = LogIngestor()
    
    # Connect to Milvus (assuming localhost since you are running this from the same machine or forwarding ports)
    # If running inside a container, you might need "milvus-standalone" as host.
    db = LogDatabase(host="192.168.1.42", port="19530")

    # 2. Parse & Embed
    ingestor.parse_file("./data/system.log")
    ingestor.embed_logs()
    
    # 3. Store in Vector DB (Persistence!)
    # We pass the vectors and the raw log dictionaries to be indexed
    if ingestor.embeddings is not None:
        db.upsert_logs(ingestor.embeddings, ingestor.logs)
    
        # 4. Search using the DB (Proof of Life)
        query = "suspicious login attempt"
        print(f"\nQuerying Milvus Database for: '{query}'")
        
        # We need to embed the query first using the same model
        query_vec = ingestor.model.encode(query)
        
        # Search via the DB class
        results = db.search(query_vec, top_k=3)
        
        print("-" * 50)
        for res in results:
            score = res['score']
            payload = res['payload']
            print(f"Score: {score:.4f} | Process: {payload.get('process', 'N/A')}")
            print(f"Log: {payload.get('clean_message', 'N/A')}\n")
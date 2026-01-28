from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List, Dict, Any
import uuid
import hashlib
class LogDatabase:
    """
    Manages the storage and retrieval of log embeddings using Milvus (Enterprise Vector DB).
    """

    def __init__(self, host: str = "localhost", port: str = "19530"):
        """
        Connects to the Milvus Standalone instance.
        """
        print(f"Connecting to Milvus at {host}:{port}...")
        connections.connect("default", host=host, port=port)
        
        self.collection_name = "opnsense_logs"
        self.collection = self._ensure_collection()

    def _ensure_collection(self) -> Collection:
        """
        Defines the schema and creates the collection if it doesn't exist.
        Schema:
            - id: VARCHAR (Primary Key)
            - vector: FLOAT_VECTOR (Dim=384)
            - process: VARCHAR (Filterable metadata)
            - message: VARCHAR (The raw log)
            - timestamp: VARCHAR
        """
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' exists. Loading...")
            collection = Collection(self.collection_name)
            collection.load()
            return collection

        print(f"Creating collection '{self.collection_name}'...")
        
        # 1. Define Fields
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="process", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="message", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        ]

        # 2. Define Schema
        schema = CollectionSchema(fields, "OPNsense Security Logs")

        # 3. Create Collection
        collection = Collection(self.collection_name, schema)

        # 4. Create Index (HNSW - The Graph Algorithm)
        # M: Max edges per node (Graph density)
        # efConstruction: Search depth during build time
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        print("Building HNSW Index...")
        collection.create_index(field_name="vector", index_params=index_params)
        
        collection.load()
        return collection

    def upsert_logs(self, embeddings: Any, metadata: List[Dict[str, Any]]):
        """
        Inserts vectors and their metadata into Milvus.
        Uses Content Hashing + Batch Deduplication to ensure Idempotency.
        """
        print(f"Processing {len(metadata)} logs for upsert...")
        
        # 1. Prepare container lists
        ids = []
        unique_vectors = []
        unique_processes = []
        unique_messages = []
        unique_timestamps = []
        
        # Set to track IDs *within this batch* to prevent duplicates
        seen_ids = set()
        
        # Convert embeddings to list if necessary
        if hasattr(embeddings, "tolist"):
            vectors_list = embeddings.tolist()
        else:
            vectors_list = embeddings

        for i, meta in enumerate(metadata):
            # 2. Deterministic ID Generation
            unique_str = f"{meta['timestamp']}-{meta['process']}-{meta['clean_message']}"
            hash_object = hashlib.md5(unique_str.encode())
            log_id = str(uuid.UUID(hex=hash_object.hexdigest()))
            
            # 3. DEDUPLICATION CHECK
            if log_id in seen_ids:
                # We have already seen this exact log content in this file. Skip it.
                continue
            
            seen_ids.add(log_id)
            
            # Add to lists
            ids.append(log_id)
            unique_vectors.append(vectors_list[i])
            unique_processes.append(meta["process"])
            unique_messages.append(meta["clean_message"])
            unique_timestamps.append(meta["timestamp"])

        print(f"Batch deduplication: Reduced {len(metadata)} raw logs to {len(ids)} unique entries.")

        if not ids:
            print("No new unique logs to insert.")
            return

        data = [
            ids,
            unique_vectors,
            unique_processes,
            unique_messages,
            unique_timestamps
        ]

        # 4. Upsert the clean unique batch
        self.collection.upsert(data)
        self.collection.flush()
        print("Upsert complete.")

    def search(self, query_vector: Any, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the database for the nearest vectors.
        """
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        search_params = {
            "metric_type": "COSINE", 
            "params": {"ef": 64} # ef: search scope
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["message", "process"] # What to return
        )

        # Parse results
        parsed_results = []
        for hits in results:
            for hit in hits:
                parsed_results.append({
                    "score": hit.score,
                    "payload": {
                        "clean_message": hit.entity.get("message"),
                        "process": hit.entity.get("process")
                    }
                })
        return parsed_results
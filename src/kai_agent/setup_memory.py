from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 1. Connect to Milvus
print("Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")

COLLECTION_NAME = "syslogs"
DIMENSION = 384  # Matches 'all-MiniLM-L6-v2' embedding size

def recreate_collection():
    # 2. Drop if exists (Clean Slate for the new Schema)
    if utility.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    # 3. Define Fields (Matching your vLLM extraction)
    fields = [
        # Primary Key
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # The Vector (Semantic Meaning)
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        # Metadata Fields (Structured Data from AI)
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="hostname", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="event_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="user", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="src_ip", dtype=DataType.VARCHAR, max_length=64),
        # Store original text for context
        FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=2048) 
    ]

    schema = CollectionSchema(fields, description="System Logs with Embeddings")

    # 4. Create Collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✅ Collection '{COLLECTION_NAME}' created.")

    # 5. Create Index (IVF_FLAT for speed/accuracy balance)
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Index created. Memory is ready.")

if __name__ == "__main__":
    recreate_collection()
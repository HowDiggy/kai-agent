# Kai Agent: Local AI Security Analyst

**Kai Agent** is an idempotent, retrieval-augmented generation (RAG) system designed to act as a Tier-3 Security Operations Center (SOC) analyst. It ingests raw network logs (OPNsense/Syslog), maps them into a semantic vector space, and utilizes a local Large Language Model (LLM) to perform evidence-based incident investigation.

This project was built to demonstrate applied AI skills, moving beyond simple API wrappers to tackle infrastructure, data engineering, and model optimization constraints common in cybersecurity environments.

---

## Architecture

The system follows a modular **Retrieval-Augmented Generation (CAG/RAG)** architecture:

1.  **Ingestion Layer:**
    * Parses semi-structured OPNsense logs.
    * Generates deterministic content hashes (MD5) for idempotency (deduplication).
    * Computes semantic embeddings using `all-MiniLM-L6-v2`.
2.  **Semantic Memory (Vector DB):**
    * **Milvus (v2.6)** running on Docker.
    * Uses **HNSW** (Hierarchical Navigable Small World) indexing for $O(\log N)$ retrieval latency.
3.  **Retrieval Engine:**
    * **Two-Stage Pipeline:** Broad retrieval (Top-25) via Bi-Encoder, followed by precision refinement (Top-5) using a **Cross-Encoder** (`ms-marco-MiniLM`).
4.  **Reasoning Engine (The Brain):**
    * **vLLM** serving `gpt-oss-120b` (or `Llama-3`) on high-performance hardware (DGX Spark).
    * **Context-Augmented Generation (CAG):** Maintains a sliding window of conversation history to resolve pronouns and context.

---

## Tech Stack

* **Language:** Python 3.12 (Type-hinted, `src` layout)
* **Dependency Management:** Poetry
* **Vector Database:** Milvus (Docker Compose)
* **Inference:** vLLM (OpenAI-compatible API)
* **Models:**
    * Embedding: `sentence-transformers/all-MiniLM-L6-v2`
    * Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
    * Generative: `gpt-oss-120b` (via local inference server)
* **Fine-Tuning:** Unsloth (QLoRA)

---

## Getting Started

### Prerequisites
* Python 3.11+
* Docker & Docker Compose
* Access to an LLM endpoint (Local vLLM or OpenAI API)

### Installation

1.  **Clone and Install Dependencies:**
    ```bash
    git clone [https://github.com/HowDiggy/kai-agent.git](https://github.com/HowDiggy/kai-agent.git)
    cd kai-agent
    poetry install
    ```

2.  **Start Infrastructure (Milvus & Attu):**
    ```bash
    docker compose up -d
    ```
    * Milvus will be available at `localhost:19530`.
    * Attu GUI will be available at `localhost:8081`.

3.  **Prepare Data:**
    Place your `system.log` file in the `./data/` directory.

### Usage

**1. Ingest Logs (ETL):**
Parse logs, generate embeddings, and upsert to Milvus.
```bash
poetry run python -m src.kai_agent.ingest
2. Chat with the Analyst: Start the interactive session.

Bash
poetry run python -m src.kai_agent.rag
Example Interaction:

User: What happened with the DHCP service recently? Kai: [Citing logs 1-5] The dhclient process on interface igb0 executed multiple RENEW requests...

Testing
Run the unit test suite:

Bash
poetry run pytest
```
--- 
### License
MIT
---

### `ROADMAP.md`

# Project Roadmap & Study Curriculum

This roadmap outlines the 12-day "Applied AI Scientist" implementation plan. It moves from theoretical foundations to production-grade engineering, focusing on the specific constraints of cybersecurity (latency, accuracy, data privacy).

---

## Phase 1: Foundations & Data Engineering

### Day 1: The Transformer Architecture
* **Goal:** Master the mathematical "Why" behind Attention.
* **Concept:** $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
* **Deliverable:** `src/kai_agent/attention_inspector.py`
    * [x] Implemented a tool to visualize attention weights in `bert-base`.
    * [x] Verified how tokens "attend" to semantically relevant neighbors.

### Day 2: Embeddings & Vector Space
* **Goal:** Map raw logs into a semantic vector space.
* **Concept:** Embeddings as geometry. Cosine Similarity as a measure of meaning.
* **Deliverable:** `src/kai_agent/ingest.py` (Version 1)
    * [x] Parsed OPNsense `system.log`.
    * [x] Generated embeddings using `all-MiniLM-L6-v2`.
    * [x] Performed in-memory semantic search ("Network Down" $\approx$ "Connection failed").

### Day 3: Vector Databases & Scalability
* **Goal:** Move from in-memory scripts to production infrastructure.
* **Concept:** HNSW (Hierarchical Navigable Small World) Indexing for $O(\log N)$ search.
* **Deliverable:** Docker Infrastructure & `src/kai_agent/legacy_log_db.py`
    * [x] Deployed Milvus Standalone & Attu via Docker Compose.
    * [x] Implemented **Idempotency** using Content Hashing (MD5) to prevent duplicate logs.
    * [x] Migrated ingestion script to upsert data into Milvus.

---

## Phase 2: The Reasoning Engine (RAG)

### Day 4: Naive RAG (Retrieval Augmented Generation)
* **Goal:** Connect Memory (Milvus) to Brain (LLM).
* **Concept:** Prompt Grounding & Evidence Citation.
* **Deliverable:** `src/kai_agent/rag.py` (Version 1)
    * [x] Integrated Local vLLM (running on DGX Spark).
    * [x] Built the `Retrieve -> Prompt -> Generate` loop.
    * [x] Engineered system prompts to force "Evidence-Based" answers.

### Day 5: Advanced RAG (Reranking)
* **Goal:** Solve the "Loss of Precision" in vector search.
* **Concept:** Bi-Encoders (Fast/Broad) vs. Cross-Encoders (Slow/Accurate).
* **Deliverable:** `src/kai_agent/reranker.py`
    * [x] Implemented a Two-Stage Retrieval pipeline.
    * [x] Stage 1: Fetch Top-25 via Milvus.
    * [x] Stage 2: Re-score using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Day 6: Conversational Memory (CAG)
* **Goal:** Enable stateful follow-up questions.
* **Concept:** Context Augmented Generation (CAG) & Query Rewriting.
* **Deliverable:** `src/kai_agent/memory.py`
    * [x] Implemented sliding window conversation memory.
    * [x] Added **Query Transformation**: Using the LLM to rewrite user queries (resolving pronouns like "it" or "that") before retrieval.

---

## Phase 3: Fine-Tuning (The "Applied Scientist" Skill)

### Day 7: Fine-Tuning with Unsloth
* **Goal:** Teach a generic model to be an expert Log Parser.
* **Concept:** LoRA (Low-Rank Adaptation) & QLoRA (Quantization).
* **Tasks:**
    * [ ] Generate synthetic "Gold Standard" dataset (Log $\to$ JSON).
    * [ ] Set up Unsloth environment on DGX (GPU).
    * [ ] Train `Llama-3-8B` to parse unstructured logs into strict JSON schemas.

### Day 8: Synthetic Data & Evaluation
* **Goal:** Evaluate system reliability without human labeling.
* **Concept:** LLM-as-a-Judge.
* **Tasks:**
    * [ ] Generate synthetic "Red Team" questions based on logs.
    * [ ] Use DeepEval to measure Hallucination Rate and Context Recall.

### Day 9: Inference Optimization
* **Goal:** Decrease latency for real-time analysis.
* **Concept:** KV Caching, Continuous Batching, & PagedAttention.
* **Tasks:**
    * [ ] Benchmark vLLM throughput (Tokens/Sec).
    * [ ] Optimize quantization settings.

---

## Phase 4: Agents & Production

### Day 10: Agentic Workflows
* **Goal:** Move from "Chatbot" to "Autonomous Agent".
* **Concept:** ReAct Pattern (Reason + Act) & Tool Use.
* **Tasks:**
    * [ ] Implement **LangGraph**.
    * [ ] Create tools: `check_ip_reputation`, `block_firewall_ip`.

### Day 11: End-to-End System Design
* **Goal:** Architecture review.
* **Tasks:**
    * [ ] Diagram full system data flow.
    * [ ] Prepare "War Stories" (e.g., how we solved the Idempotency crash on Day 3).

### Day 12: Final Polish & Mock Interview
* **Goal:** Interview Readiness.
* **Tasks:**
    * [ ] Full mock interview simulation.
    * [ ] Code review and final refactor.
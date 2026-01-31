import os
from typing import List, Dict, Any
from openai import OpenAI
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
from src.kai_agent.memory import ConversationMemory

# --- CONFIGURATION (UPDATED) ---
VLLM_HOST = "http://localhost:8005/v1" 
MODEL_NAME = "kai-log-parser"  # Your fine-tuned model
COLLECTION_NAME = "syslogs"    # Your new Day 11 schema

class ConversationalAnalyst:
    """
    The 'Brain' of the agent. 
    Integrates Memory (CAG), Retrieval (Milvus), and Reasoning (Fine-Tuned Llama-3).
    """
    
    def __init__(self):
        
        # 1. Setup Retrieval (New Schema)
        print(f"Connecting to Memory (Milvus)...")
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()
        
        # 2. Setup Embeddings
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # 3. Setup Chat Memory (Preserved from your old version)
        self.memory = ConversationMemory(max_turns=3)
        
        # 4. Setup Reasoning (New Fine-Tuned Brain)
        print(f"Connecting to Brain (vLLM at {VLLM_HOST})...")
        self.client = OpenAI(base_url=VLLM_HOST, api_key="EMPTY")
        self.model_name = MODEL_NAME

    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        """
        Embeds query -> Vector Search (Top 10) -> Format for Llama-3
        """
        query_vec = self.embedder.encode([query])
        
        # Search parameters for IVF_FLAT index
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=query_vec, 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k, 
            # Note: We fetch the NEW fields we defined in Day 11
            output_fields=["timestamp", "event_type", "user", "raw_text"]
        )

        if not results:
            return ""

        # Format for the LLM: "Row [ID]: <Content>"
        context_block = ""
        for i, hit in enumerate(results[0]):
            data = hit.entity
            # Using the structured fields helps the model cite evidence
            context_block += f"[{i+1}] Time: {data.get('timestamp')} | Event: {data.get('event_type')} | User: {data.get('user')} | Log: {data.get('raw_text')}\n"
        
        return context_block

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def chat(self, user_query: str) -> str:
        """
        Performs the full RAG loop: Retrieval -> Prompt -> Generation
        """
        print(f"\nUser: '{user_query}'")
        
        # 1. Retrieve Context
        context_logs = self.retrieve_context(user_query)
        
        if not context_logs:
            return "No relevant logs found in the database."

        print(f"Retrieved relevant logs...")

        # 2. Prompt Engineering (Fine-Tuned Style)
        system_prompt = (
            "You are a Cyber Security Analyst. "
            "Analyze the provided logs to answer the user's question. "
            "Cite specific timestamps and events as evidence. "
            "If the logs don't support the answer, state that clearly."
        )

        user_prompt = f"""
        PREVIOUS CONVERSATION:
        {self.memory.get_history_block()}

        CURRENT QUESTION:
        {user_query}

        SYSTEM LOGS EVIDENCE:
        {context_logs}
        
        RESPONSE:
        """

        # 3. Generation
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                max_tokens=512,
                stop=["###"] # Stop token to prevent hallucinated conversations
            )
            answer = response.choices[0].message.content

            # 4. Update Memory
            self.memory.add_turn(user_query, answer)
            return answer

        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    agent = ConversationalAnalyst()
    
    # Test Interaction
    print("=" * 60)
    print(agent.chat("What happened with the ipv6 configuration?"))
    print("-" * 50)
    print(agent.chat("Did the DHCP service report any errors?")) 
    print("=" * 60)
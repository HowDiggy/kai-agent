import os
from typing import List, Dict, Any
from openai import OpenAI
from src.kai_agent.legacy_log_db import LogDatabase
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
from src.kai_agent.reranker import LogReranker
from src.kai_agent.memory import ConversationMemory


class ConversationalAnalyst:
    """
    The 'Brain' of the agent. 
    Integrates Memory (CAG), Retrieval (Milvus), Reranking (Cross-Encoder), and Reasoning (vLLM).
    """
    
    def __init__(self, 
                 milvus_host: str = "192.168.1.42", 
                 vllm_host: str = "http://192.168.1.42:8000/v1", 
                 model_name: str = "gpt-oss-120b"):
        
        # 1. Setup Retrieval (Memory)
        print(f"Connecting to Memory (Milvus at {milvus_host})...")
        self.db = LogDatabase(host=milvus_host)
        
        # 2. Setup Embeddings (Must match ingestion model)
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # 3. Setup Chat Memory (Day 6 feature)
        self.memory = ConversationMemory(max_turns=3)
        
        # 4. Setup Reasoning (Local Brain on DGX)
        print(f"Connecting to Brain (vLLM at {vllm_host})...")
        # vLLM requires an API key string, even if it's fake.
        self.client = OpenAI(base_url=vllm_host, api_key="EMPTY")
        self.model_name = model_name

        # 5. Setup Reranker (Day 5 feature)
        self.reranker = LogReranker()

    def rewrite_query(self, user_query: str) -> str:
        """
        Uses the LLM to de-reference pronouns based on history.
        Example: "Was it valid?" -> "Was the DHCP renewal valid?"
        """
        history = self.memory.get_history_block()
        if not history:
            return user_query

        print("Rewriting query based on history...")
        prompt = f"""
        {history}
        CURRENT QUERY: {user_query}
        
        TASK: Rewrite the CURRENT QUERY to be standalone and explicit, resolving any pronouns (it, that, he) using the PREVIOUS CONVERSATION. Do not answer the question, just rewrite it.
        
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            rewritten = response.choices[0].message.content.strip()
            # Clean up if the model adds quotes
            rewritten = rewritten.replace('"', '').replace("'", "")
            print(f"Original: '{user_query}' -> Rewritten: '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"Rewrite failed: {e}")
            return user_query

    def retrieve_context(self, query: str, top_k: int = 25) -> str:
        """
        Embeds query -> Vector Search (Top 25) -> Rerank (Top 5)
        """
        query_vec = self.embedder.encode(query)
        initial_results = self.db.search(query_vec, top_k=top_k)
        
        if not initial_results:
            return ""
        
        refined_results = self.reranker.rank_logs(query, initial_results, top_k=5)

        # Format for the LLM: "Row [ID]: <Content>"
        context_block = ""
        for i, res in enumerate(refined_results):
            payload = res['payload']
            # We explicitly format this so the LLM can cite specific log lines
            context_block += f"[{i+1}] (Score: {res['rerank_score']:.2f}) Time: {payload.get('timestamp')} | Process: {payload.get('process')} | Msg: {payload.get('clean_message')}\n"
        
        return context_block

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def chat(self, user_query: str) -> str:
        """
        Performs the full RAG loop: Retrieval -> Prompt -> Generation
        """
        print(f"\nUser: '{user_query}'")
        
        # 1. Query Rewrite (The Magic Step)
        standalone_query = self.rewrite_query(user_query)

        # Step 2: Retrieval using rewritten query
        context_logs = self.retrieve_context(standalone_query)
        
        if not context_logs:
            return "No relevant logs found in the database."

        print(f"Retrieved context logs...")

        # Step 3: Prompt Engineering (Applied Scientist Level)
        # We enforce a 'Persona' and 'Evidence' requirement.
        system_prompt = (
            "You are Kai, a Senior Security Operations Center (SOC) Analyst. "
            "Your goal is to investigate network events based ONLY on the provided logs and the Conversation History.\n"
            "RULES:\n"
            "1. Evidence First: Cite specific timestamps and process names.\n"
            "2. No Hallucinations: If the logs don't explain the cause, admit it.\n"
            "3. Brevity: Be concise and technical."
        )

        user_prompt = f"""
        USER INQUIRY: {self.memory.get_history_block}

        CURRENT QUESTION:
        {user_query}

        NEW RETRIEVED LOGS (for context):
        {context_logs}
        
        RESPONSE:
        """

        # Step 3: Generation (Local Inference)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Low temp = High Factuality
                max_tokens=512
            )
            answer = response.choices[0].message.content

            # 4. Update Memory
            self.memory.add_turn(user_query, answer)
            return answer

        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    # CONFIGURATION
    # Update this to your DGX Spark's LAN IP
    DGX_IP = "192.168.1.42" 
    
    # Initialize Agent
    agent = ConversationalAnalyst(
        milvus_host=DGX_IP, 
        vllm_host=f"http://{DGX_IP}:8000/v1",
        model_name="gpt-oss-120b" # Ensure this matches the model name served by vLLM
    )
    
    # # Test Scenarios
    # questions = [
    #     "What happened with the DHCP service recently?",
    #     "Are there any errors related to 'unbound' or DNS?",
    # ]
    
    # for q in questions:
    #     print("=" * 60)
    #     answer = analyst.analyze(q)
    #     print("\nFINAL ANSWER:")
    #     print(answer)
    #     print("=" * 60)

    # Simulation of a conversation
    print("=" * 60)
    print(agent.chat("What happened with DHCP recently?"))
    print("-" * 50)
    # This requires memory to understand "that" refers to the DHCP event
    print(agent.chat("Was that on the igb0 interface?")) 
    print("=" * 60)
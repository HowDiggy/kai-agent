from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class LogReranker:
    """
    Refines retrieval results using a Cross-Encoder.
    This acts as a 'second opinion' to filter out irrelevant logs that 
    passed the initial vector search.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Cross-Encoder: {model_name}...")
        # CrossEncoders process (Query, Doc) pairs together
        self.model = CrossEncoder(model_name)

    def rank_logs(self, query: str, initial_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-scores the initial search results based on relevance to the query.
        """
        if not initial_results:
            return []

        # Prepare pairs for the model: [('Query', 'Log1'), ('Query', 'Log2')...]
        # We use the 'clean_message' for scoring
        pairs = [[query, res['payload']['clean_message']] for res in initial_results]
        
        # Predict scores (higher is better)
        scores = self.model.predict(pairs)

        # Attach scores to results
        for i, res in enumerate(initial_results):
            res['rerank_score'] = scores[i]

        # Sort by new score descending
        sorted_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Debug print to show the reranking effect
        print(f"\n--- Reranking Effect (Query: '{query}') ---")
        for i, res in enumerate(sorted_results[:top_k]):
            old_score = res['score']
            new_score = res['rerank_score']
            msg = res['payload']['clean_message'][:50]
            print(f"#{i+1}: Vector Score: {old_score:.4f} -> Rerank Score: {new_score:.4f} | {msg}...")
        print("-" * 50)

        return sorted_results[:top_k]
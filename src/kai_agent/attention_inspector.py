import torch
from transformers import BertTokenizer, BertModel, PreTrainedTokenizer, PreTrainedModel
from typing import List, Tuple, Dict, Any
import numpy as np

class AttentionInspector:
    """
    A tool to inspect and visualize the self-attention mechanism of a Transformer model.
    It allows us to peek into the 'black box' and see which tokens the model prioritizes.
    
    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer for processing text input.
        model (PreTrainedModel): The transformer model to inspect.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        """
        Initializes the inspector with a pre-trained BERT model.

        Args:
            model_name (str): The Hugging Face model identifier. Defaults to 'bert-base-uncased'.
        """
        # We use 'bert-base-uncased' because its attention heads are highly interpretable for
        # security logs (bi-directional context).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name} on {self.device}...")
        
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(model_name)
        # output_attentions=True is critical; it forces the model to return the 
        # raw attention weights (the result of the Softmax) along with the logits.
        self.model: PreTrainedModel = BertModel.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (disable dropout, etc.)

    def get_attention_map(self, text: str) -> Dict[str, Any]:
        """
        Computes the attention weights for a given input text.

        The returned 'attention' matrix represents the Softmax(QK^T / sqrt(d_k)) 
        portion of the equation.

        Pre-conditions:
            - text must not be empty.

        Post-conditions:
            - Returns a dictionary containing tokens and the attention matrix.

        Args:
            text (str): The log line or text to analyze.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - 'tokens' (List[str]): The list of input tokens.
                - 'attention' (np.ndarray): The attention weights from the last layer.
                  Shape: (num_heads, seq_len, seq_len)
        """
        if not text:
            raise ValueError("Input text cannot be empty.")

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # outputs.attentions is a tuple of tensors (one for each layer)
        # We take the last layer: shape (batch_size, num_heads, seq_len, seq_len)
        last_layer_attention = outputs.attentions[-1]
        
        # Remove batch dimension -> (num_heads, seq_len, seq_len)
        attention_matrix = last_layer_attention.squeeze(0).cpu().numpy()
        
        # Convert input IDs back to tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return {
            "tokens": tokens,
            "attention": attention_matrix
        }

    def print_top_attentions(self, text: str, target_word: str, top_k: int = 3) -> None:
        """
        Prints the tokens that the 'target_word' pays the most attention to,
        filtering out common 'Attention Sink' tokens (punctuation, separators)
        to reveal semantic relationships.

        Args:
            text (str): The full log line.
            target_word (str): The specific word to analyze (must exist in text).
            top_k (int): Number of top attention scores to display.
        """
        result = self.get_attention_map(text)
        tokens = result["tokens"]
        attention_matrix = result["attention"]

        # Average attention across all heads (simple view)
        # Shape: (seq_len, seq_len)
        avg_attention = np.mean(attention_matrix, axis=0)

        try:
            # Find the index of the target word (first occurrence)
            target_idx = tokens.index(target_word)
        except ValueError:
            print(f"Error: Word '{target_word}' not found in tokens: {tokens}")
            return

        # Get attention scores for this token against all others
        scores = avg_attention[target_idx]
        
        # Create a list of (token, score) tuples
        token_scores = list(zip(tokens, scores))
        
        # Filter out [CLS], [SEP], and punctuation like ']', ':'
        # This removes the "Attention Sink" noise
        filtered_scores = [
            (t, s) for t, s in token_scores 
            if t not in ["[CLS]", "[SEP]", "]", ":", ".", "[", "s"]
        ]
        
        # Sort by score descending
        sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_k]

        print(f"\nAnalysis for token: '{target_word}' (Filtered)")
        print(f"Context: {text}")
        print("-" * 40)
        print(f"{'Token':<15} | {'Attention Score':<10}")
        print("-" * 40)
        
        for token, score in sorted_scores:
            print(f"{token:<15} | {score:.4f}")

if __name__ == "__main__":
    # Example: SSH Failure Log
    log_line = "Jan 20 10:00:01 firewall-1 sshd[123]: Failed password for root from 10.0.0.5"
    
    inspector = AttentionInspector()
    
    # Analyze what 'failed' pays attention to
    inspector.print_top_attentions(log_line, target_word="failed")
    
    # Analyze what 'root' pays attention to
    inspector.print_top_attentions(log_line, target_word="root")
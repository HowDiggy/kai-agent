import pytest
from src.kai_agent.attention_inspector import AttentionInspector

def test_attention_shape():
    """
    Verifies that the attention matrix has the correct shape corresponding
    to the number of tokens in the input.
    """
    text = "hello world"
    inspector = AttentionInspector()
    result = inspector.get_attention_map(text)
    
    tokens = result["tokens"]
    attention = result["attention"]
    
    # BERT adds [CLS] and [SEP], so "hello world" becomes 4 tokens: [CLS] hello world [SEP]
    seq_len = len(tokens)
    num_heads = 12 # BERT base has 12 heads
    
    assert len(tokens) == 4
    assert attention.shape == (num_heads, seq_len, seq_len)

def test_empty_input_raises_error():
    """Verifies that empty input raises a ValueError as defined in preconditions."""
    inspector = AttentionInspector()
    with pytest.raises(ValueError):
        inspector.get_attention_map("")
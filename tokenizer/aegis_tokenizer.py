import tiktoken
from typing import List

class BPETokenizer:
    """
    Python implementation mimicking the Aegis-Tokenizer interface 
    for compatibility and robust execution on Windows.
    Internally uses tiktoken for high-performance BPE.
    """
    
    def __init__(self, model_path: str = None):
        self._tokenizer = tiktoken.get_encoding("gpt2")
        
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self._tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        return self._tokenizer.decode(tokens)
        
    def batch_encode(self, texts: List[str], num_threads: int = None, use_multiprocessing: bool = False) -> List[List[int]]:
        return self._tokenizer.encode_batch(texts, num_threads=num_threads)

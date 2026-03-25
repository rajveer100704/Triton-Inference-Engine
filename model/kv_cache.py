import torch

class KVCache:
    def __init__(self, batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: torch.device):
        """
        Static KV Cache pre-allocated on the GPU.
        """
        self.max_seq_len = max_seq_len
        # Ensure memory is contiguous for optimal kernel performance
        self.keys = torch.zeros((batch_size, n_heads, max_seq_len, head_dim), dtype=torch.float16, device=device).contiguous()
        self.values = torch.zeros((batch_size, n_heads, max_seq_len, head_dim), dtype=torch.float16, device=device).contiguous()
        self.cur_pos = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Updates the cache with new keys and values.
        k, v shape: (batch_size, n_heads, seq_len_to_add, head_dim)
        """
        seq_len_to_add = k.size(2)
        if self.cur_pos + seq_len_to_add > self.max_seq_len:
            raise ValueError(f"KV Cache exceeded maximum sequence length {self.max_seq_len}")
            
        self.keys[:, :, self.cur_pos : self.cur_pos + seq_len_to_add, :] = k
        self.values[:, :, self.cur_pos : self.cur_pos + seq_len_to_add, :] = v
        self.cur_pos += seq_len_to_add

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the keys and values up to the current position.
        """
        return self.keys[:, :, :self.cur_pos, :], self.values[:, :, :self.cur_pos, :]

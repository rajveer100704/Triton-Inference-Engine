import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config
from .kv_cache import KVCache

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert config.model.d_model % config.model.n_heads == 0
        self.n_heads = config.model.n_heads
        self.d_model = config.model.d_model
        self.head_dim = self.d_model // self.n_heads
        
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
    def forward(self, x, kv_cache: KVCache = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        
        if kv_cache is not None:
            # We are using KV cache.
            # Only add the current sequence's keys and values to the cache
            kv_cache.update(k, v)
            # Retrieve the full history for attention computation
            k, v = kv_cache.get()
            
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # We don't need a formal mask explicitly if we use scaled_dot_product_attention with is_causal=True
        # However, is_causal=True only works properly if q and k lengths are the same.
        # When using KV cache and decoding 1 token at a time, q len is 1, k len is > 1, so no mask is needed.
        # If Q len > 1 (e.g., prefill phase), we DO need is_causal=True.
        is_prefill = (T > 1) and (kv_cache is None or kv_cache.cur_pos == T)
        
        y = None
        if self.config.inference.use_triton:
            try:
                from kernels.triton_attention import triton_attention
                # Triton kernel expects unscaled Q
                # scale is passed directly to the kernel
                sm_scale = 1.0 / math.sqrt(self.head_dim)
                
                # Currently simple triton kernel expects full inputs without mask complexities
                # or we just pass it to the kernel which handles causal masking
                y = triton_attention(q, k, v, sm_scale)
            except Exception as e:
                print(f"Triton kernel failed: {e}. Falling back to PyTorch.")
                
        if y is None:
            # Fallback to PyTorch
            with torch.cuda.amp.autocast(dtype=torch.float16):
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_prefill)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc    = nn.Linear(config.model.d_model, 4 * config.model.d_model, bias=False)
        self.act     = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.model.d_model, config.model.d_model, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.model.d_model, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.model.d_model, bias=False)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache: KVCache = None):
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.model.vocab_size, config.model.d_model),
            wpe = nn.Embedding(config.inference.max_seq_len, config.model.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.model.n_layers)]),
            ln_f = nn.LayerNorm(config.model.d_model, bias=False),
        ))
        
        self.lm_head = nn.Linear(config.model.d_model, config.model.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, kv_caches=None):
        device = idx.device
        b, t = idx.size()
        
        if kv_caches is not None:
            # During decoding, idx is shape (b, 1), and we fetch the current position from cache length
            pos_offset = kv_caches[0].cur_pos
            pos = torch.arange(pos_offset, pos_offset + t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, d_model)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        
        x = tok_emb + pos_emb
        
        for i, block in enumerate(self.transformer.h):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (b, t, vocab_size)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Baseline generate loop. Uses generic kv_caches logic if config enables it.
        """
        import time
        device = idx.device
        b = idx.size(0)
        
        use_kv_cache = self.config.inference.use_kv_cache
        kv_caches = None
        ttft_ms = 0.0
        
        if use_kv_cache:
            kv_caches = [
                KVCache(
                    batch_size=b, 
                    max_seq_len=self.config.inference.max_seq_len, 
                    n_heads=self.config.model.n_heads, 
                    head_dim=self.config.model.d_model // self.config.model.n_heads, 
                    device=device
                ) 
                for _ in range(self.config.model.n_layers)
            ]
            
            # Prefill phase - Track TTFT precisely
            torch.cuda.synchronize() if idx.is_cuda else None
            t0 = time.perf_counter()
            
            _ = self(idx, kv_caches=kv_caches)
            
            torch.cuda.synchronize() if idx.is_cuda else None
            t1 = time.perf_counter()
            ttft_ms = (t1 - t0) * 1000.0
            
            # Next token to process in decoding loop
            idx_next = idx[:, -1:]
        else:
            # No cache, we must pass the full sequence every time
            idx_cond = idx
            
        for step in range(max_new_tokens):
            if step == 0 and not use_kv_cache:
                torch.cuda.synchronize() if idx.is_cuda else None
                t0 = time.perf_counter()
                
            if use_kv_cache:
                # Decoding phase: only pass the last token
                logits = self(idx_next, kv_caches=kv_caches)
                logits = logits[:, -1, :] / temperature
            else:
                # No cache: pass the growing sequence
                logits = self(idx_cond, kv_caches=None)
                logits = logits[:, -1, :] / temperature
                
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            if step == 0 and not use_kv_cache:
                torch.cuda.synchronize() if idx.is_cuda else None
                t1 = time.perf_counter()
                ttft_ms = (t1 - t0) * 1000.0
                
            if not use_kv_cache:
                idx_cond = idx
                
        return idx, ttft_ms

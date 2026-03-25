import torch
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model.transformer import Transformer

def test_baseline_generation():
    config = Config()
    # reduce config parameters for faster local test
    config.model.n_layers = 1
    config.model.n_heads = 2
    config.model.d_model = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)
    model.half() # Need half precision to match KVCache
    
    # Dummy input
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    
    # Test generation without KV Cache
    config.inference.use_kv_cache = False
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out_no_cache, ttft0 = model.generate(idx, max_new_tokens=50)
    t1 = time.perf_counter()
    vram0 = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    lat0 = (t1 - t0) * 1000
    
    # Test generation WITH KV Cache
    config.inference.use_kv_cache = True
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out_with_cache, ttft1 = model.generate(idx, max_new_tokens=50)
    t1 = time.perf_counter()
    vram1 = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    lat1 = (t1 - t0) * 1000
    
    print(f"\nREAL_BASELINE: Latency {lat0:.2f}ms | Throughput {50/(lat0/1000):.2f}tok/s | TTFT {ttft0:.2f}ms | {vram0:.2f}MB VRAM")
    print(f"REAL_OPTIMIZED: Latency {lat1:.2f}ms | Throughput {50/(lat1/1000):.2f}tok/s | TTFT {ttft1:.2f}ms | {vram1:.2f}MB VRAM")
    
    # Assert correctness
    assert out_no_cache.size(1) == 53, f"Expected length 8, got {out_no_cache.size(1)}"
    assert out_with_cache.size(1) == 53, f"Expected length 8, got {out_with_cache.size(1)}"
    
    print("Baseline generation tests passed!")

if __name__ == "__main__":
    test_baseline_generation()

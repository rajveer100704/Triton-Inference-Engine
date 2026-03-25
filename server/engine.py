import asyncio
import time
import torch
import sys
import os
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model.transformer import Transformer
from tokenizer import BPETokenizer

class InferenceRequest:
    def __init__(self, prompt: str, max_tokens: int, temperature: float, top_k: Optional[int], stream: bool = False):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.stream = stream
        self.future = asyncio.Future() if not stream else asyncio.Queue()
        self.start_time = time.time()
        self.tokens_generated = 0
        self.cache_hits = 0

class InferenceEngine:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Baseline PyTorch Model
        self.model = Transformer(self.config).to(self.device)
        self.model.half() # Ensure FP16 precision
        self.model.eval()
        
        # Load Tokenizer
        self.tokenizer = BPETokenizer()
        
        # Scheduling properties
        self.queue = asyncio.Queue()
        self.max_batch_size = self.config.inference.batch_size
        self.batch_timeout = 0.010 # 10ms batching window
        
        self.worker_task = None
        
    async def start(self):
        self.worker_task = asyncio.create_task(self._worker_loop())
        
    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
                
    async def generate(self, prompt: str, max_tokens: int, temperature: float, top_k: Optional[int]) -> Dict[str, Any]:
        req = InferenceRequest(prompt, max_tokens, temperature, top_k, stream=False)
        await self.queue.put(req)
        return await req.future
        
    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, top_k: Optional[int]):
        req = InferenceRequest(prompt, max_tokens, temperature, top_k, stream=True)
        await self.queue.put(req)
        
        while True:
            chunk = await req.future.get()
            if chunk is None: # EOF signal
                break
            yield chunk

    async def _worker_loop(self):
        while True:
            batch: List[InferenceRequest] = []
            
            # Wait for the first request
            req = await self.queue.get()
            batch.append(req)
            
            # Try to grab more requests within a 10ms window
            end_time = asyncio.get_event_loop().time() + self.batch_timeout
            while len(batch) < self.max_batch_size:
                timeout = end_time - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
                    
            if len(batch) > 0:
                # To prevent blocking the event loop entirely while the GPU runs,
                # we offload the blocking generation to a thread.
                await asyncio.to_thread(self._process_batch, batch)

    @torch.no_grad()
    def _process_batch(self, batch: List[InferenceRequest]):
        # Encode
        texts = [r.prompt for r in batch]
        input_ids = self.tokenizer.batch_encode(texts)
        
        # Left-pad inputs to match longest in batch
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        for ids in input_ids:
            padded_ids.append([self.tokenizer._tokenizer.eot_token] * (max_len - len(ids)) + ids)
            
        start_t = time.perf_counter()
        
        idx = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        
        # Take the maximum tokens from batch to bound the generation.
        # A true continuous batching engine handles varying completion dynamically.
        # For our v1 static KV cache, we generate uniformly until the max bound.
        max_new_tokens = max(r.max_tokens for r in batch)
        
        # Since varying temperatures/top_k in a batched tensor operation is complex, 
        # we simplify by using the parameters from the first request in the batch for this MVP.
        temperature = batch[0].temperature
        top_k = batch[0].top_k
        
        # Generate entirely
        torch.cuda.reset_peak_memory_stats(self.device) if self.device.type == 'cuda' else None
        out, ttft_ms = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if self.device.type == 'cuda' else 0.0
        
        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000.0
        
        # Decode and respond
        for i, req in enumerate(batch):
            # Strip prompt and padding
            generated_ids = out[i][max_len:].tolist()
            text = self.tokenizer.decode(generated_ids)
            
            tokens_generated = len(generated_ids)
            
            result = {
                "text": text,
                "latency_ms": latency_ms,
                "tokens_generated": tokens_generated,
                "tokens_per_sec": tokens_generated / (latency_ms / 1000.0) if latency_ms > 0 else 0,
                "cache_hits": tokens_generated - 1 if self.config.inference.use_kv_cache else 0,
                "ttft_ms": ttft_ms,
                "peak_vram_mb": peak_vram_mb
            }
            
            # Simple streaming fallback: if streaming requested, just send the one big chunk for now.
            # True token-by-token streaming with static batching requires modifying generate() to yield.
            if req.stream:
                req.future.put_nowait(result)
                req.future.put_nowait(None)
            else:
                # Provide the result back to the waiting HTTP handler
                # Note: call_soon_threadsafe is required since we are in `to_thread`
                asyncio.get_event_loop().call_soon_threadsafe(req.future.set_result, result)

engine_instance = InferenceEngine()

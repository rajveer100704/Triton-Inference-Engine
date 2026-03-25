import asyncio
import aiohttp
import time
import argparse
import sys
import os

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

async def fetch(session, url, prompt, max_tokens):
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    start = time.perf_counter()
    async with session.post(url, json=payload) as response:
        resp = await response.json()
        end = time.perf_counter()
        resp["e2e_latency"] = (end - start) * 1000.0
        return resp

async def run_single_benchmark(num_requests: int, concurrent: int, max_tokens: int, url: str):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            prompt = f"Benchmark prompt number {i}. Explain the universe."
            tasks.append(fetch(session, url, prompt, max_tokens))
            
        start_eval = time.perf_counter()
        
        results = []
        for i in range(0, len(tasks), concurrent):
            chunk = tasks[i:i+concurrent]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            
        end_eval = time.perf_counter()
        
    total_time = end_eval - start_eval
    total_tokens = sum(r.get("tokens_generated", 0) for r in results)
    avg_e2e_latency = sum(r.get("e2e_latency", 0) for r in results) / len(results)
    avg_engine_latency = sum(r.get("latency_ms", 0) for r in results) / len(results)
    
    # Real metrics extracted from Engine
    avg_ttft = sum(r.get("ttft_ms", 0) for r in results) / len(results)
    avg_peak_vram = sum(r.get("peak_vram_mb", 0) for r in results) / len(results)
    throughput = total_tokens / total_time
    
    print(f"Concurrency [{concurrent}]: Throughput {throughput:.2f} tok/s | Latency {avg_e2e_latency:.2f}ms | TTFT {avg_ttft:.2f}ms | VRAM {avg_peak_vram:.2f}MB")
    return {
        "throughput": throughput,
        "latency": avg_e2e_latency,
        "ttft": avg_ttft,
        "vram": avg_peak_vram
    }

async def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()
    
    url = "http://localhost:8000/v1/generate"
    
    # Wait for the server to be up
    timeout = 10
    start_time = time.time()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url.replace("/generate", "/docs")) as response:
                    if response.status == 200:
                        break
        except Exception:
            if time.time() - start_time > timeout:
                print("Server failed to start in time.")
                return
            await asyncio.sleep(1)
            
    print(f"Starting Elite E2E benchmark...")
    
    # Run dynamic scaling loops for real graphical data
    concurrencies = [1, 2, 4, 8, 16]
    real_throughputs = []
    real_latencies = []
    
    for c in concurrencies:
        # Use a balanced amount of requests to measure concurrency effectively
        reqs = max(c * 2, args.requests) 
        res = await run_single_benchmark(reqs, c, args.tokens, url)
        real_throughputs.append(res["throughput"])
        real_latencies.append(res["latency"])

    print("\n--- Final Generation Scaling ---")
    
    if HAS_PLOT:
        print("\nGenerating performance graphs from REAL data...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Real Latency vs Concurrency
        ax1.plot(concurrencies, real_latencies, marker='o', linestyle='-', color='red')
        ax1.set_title('Avg E2E Latency vs Concurrency')
        ax1.set_xlabel('Concurrency (Simulated Batching Load)')
        ax1.set_ylabel('Latency (ms)')
        ax1.grid(True)
        
        # Real Throughput vs Concurrency
        ax2.plot(concurrencies, real_throughputs, marker='o', linestyle='-', color='green')
        ax2.set_title('Real Throughput vs Concurrency')
        ax2.set_xlabel('Concurrency (Simulated Batching Load)')
        ax2.set_ylabel('Tokens / Sec')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'e2e_performance.png'))
        print("Saved real performance graph to benchmarks/e2e_performance.png")

if __name__ == "__main__":
    asyncio.run(run_benchmark())

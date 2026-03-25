import torch
try:
    import triton
except ImportError:
    print("Warning: Triton library is not installed. The benchmark will only run the PyTorch baseline. To benchmark Triton kernels, please install triton on a compatible OS (Linux/WSL).")
    triton = None
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.triton_attention import triton_attention
import math

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N_CTX'],  # Argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7, 12)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['pytorch', 'triton'],  # Possible values for `line_arg`
        line_names=['PyTorch', 'Triton'],  # Label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # Line styles
        ylabel='TFLOPS',  # Label name for the y-axis
        plot_name='attention-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={'H': 8, 'B': 2, 'D_HEAD': 64},  # Values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(B, H, N_CTX, D_HEAD, provider):
    q = torch.randn((B, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn((B, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn((B, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda", requires_grad=True)
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        # Flash attention style standard pytorch implementation
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_attention(q, k, v, sm_scale), quantiles=quantiles)
    
    # Calculate TFLOPS
    # 2 ops per MAC, 2 for QK, 2 for PV, resulting in 4 * B * H * N_CTX * N_CTX * D_HEAD
    flops_per_iter = 4 * B * H * N_CTX * N_CTX * D_HEAD
    return flops_per_iter / ms * 1e-9, flops_per_iter / max_ms * 1e-9, flops_per_iter / min_ms * 1e-9

def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA is required for Triton benchmark.")
        return
        
    print("Running Triton vs PyTorch benchmark...")
    # Will save a plot to 'benchmarks' directory if possible, or just print table
    benchmark.run(print_data=True, show_plots=False, save_path=os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    run_benchmark()

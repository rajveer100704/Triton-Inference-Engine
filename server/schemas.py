from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text prompt.")
    max_tokens: int = Field(50, description="Maximum number of tokens to generate.")
    temperature: float = Field(1.0, description="Sampling temperature.")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter.")

class GenerateResponse(BaseModel):
    text: str = Field(..., description="The generated output text.")
    latency_ms: float = Field(..., description="Total request latency in milliseconds.")
    tokens_generated: int = Field(..., description="Number of tokens generated.")
    tokens_per_sec: float = Field(..., description="Throughput for this request in tokens/sec.")
    cache_hits: int = Field(default=0, description="Number of times KV cache was successfully hit/used.")
    ttft_ms: float = Field(..., description="Time to First Token (TTFT) in milliseconds.")
    peak_vram_mb: float = Field(..., description="Peak GPU VRAM allocated during this request in MB.")

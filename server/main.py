from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
from server.schemas import GenerateRequest, GenerateResponse
from server.engine import InferenceEngine

app = FastAPI(title="End-to-End Optimized Transformer Inference Engine")
engine = InferenceEngine()

@app.on_event("startup")
async def startup_event():
    await engine.start()

@app.on_event("shutdown")
async def shutdown_event():
    await engine.stop()

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Standard generation endpoint taking advantage of background dynamic batching.
    """
    try:
        result = await engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        return GenerateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/generate_stream")
async def generate_stream(request: GenerateRequest):
    """
    Streaming generation endpoint using Server-Sent Events (SSE).
    """
    async def event_generator():
        try:
            async for chunk in engine.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

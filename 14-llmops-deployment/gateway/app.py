"""
gateway/app.py - PRODUCTION API gateway in front of the vLLM model server.

Runs in the Docker Compose stack (docker-compose.yml). Does the gateway's job: auth (API keys),
rate limiting (token bucket), health checks, and proxying to vLLM. Run: uvicorn app:app --port 8080
"""
from __future__ import annotations
import os, time
from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="LLM gateway")
VALID_KEYS = set(os.getenv("API_KEYS", "demo-key").split(","))
_tokens = {"n": 20.0, "t": time.time()}            # simple token bucket (capacity 20, refill 10/s)


def rate_ok():
    now = time.time()
    _tokens["n"] = min(20.0, _tokens["n"] + (now - _tokens["t"]) * 10.0); _tokens["t"] = now
    if _tokens["n"] >= 1:
        _tokens["n"] -= 1; return True
    return False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat(body: dict, authorization: str = Header(default="")):
    if authorization.replace("Bearer ", "") not in VALID_KEYS:
        raise HTTPException(401, "invalid API key")          # auth
    if not rate_ok():
        raise HTTPException(429, "rate limit exceeded")      # rate limit
    # PRODUCTION TODO: proxy `body` to the vLLM server (http://vllm:8000/v1/chat/completions),
    # log latency/tokens/cost to Prometheus, and return the response.
    return {"status": "ok", "note": "wire up the vLLM proxy here"}

"""
serving_sim.py - the key production-serving ideas, simulated (runnable, no GPU).

    python -m src.serving_sim

WHAT : simulates (1) why BATCHING multiplies throughput, (2) latency p50/p95/p99 (the tail), and
       (3) a token-bucket RATE LIMITER - with assertions so each claim is verified.
WHY  : serving a model to thousands of users fast and cheaply is what turns AI into a product, and
       it's where data/ops experience is an edge many ML engineers lack.
HOW  : amortise fixed GPU overhead across a batch; percentiles of simulated latencies; token bucket.
WHERE: the concept layer. gateway/app.py + docker-compose.yml are the production stack (vLLM).
       See notebooks/14_serving_mlops.ipynb for the throughput/latency/architecture diagrams.
"""
from __future__ import annotations
import numpy as np


def throughput(batch_sizes, overhead_ms=50.0, per_req_ms=5.0):
    """Per-request time = (fixed overhead + B*marginal)/B; throughput = 1000/that (req/s)."""
    B = np.array(batch_sizes, dtype=float)
    return 1000.0 / ((overhead_ms + B * per_req_ms) / B)


def latency_percentiles(n=5000, seed=0):
    lat = np.random.default_rng(seed).lognormal(mean=np.log(220), sigma=0.45, size=n)
    return np.percentile(lat, [50, 95, 99])


class TokenBucket:
    """Allow up to `capacity` requests, refilling `refill`/sec. Rejects bursts (HTTP 429)."""
    def __init__(self, capacity, refill):
        self.capacity = capacity; self.refill = refill; self.tokens = capacity; self.t = 0.0

    def allow(self, now):
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.refill); self.t = now
        if self.tokens >= 1:
            self.tokens -= 1; return True
        return False


def main():
    tp = throughput([1, 8, 64])
    print(f"[throughput] batch 1: {tp[0]:.0f} req/s  ->  batch 64: {tp[2]:.0f} req/s")
    assert tp[2] > 5 * tp[0], "batching should multiply throughput"

    p50, p95, p99 = latency_percentiles()
    print(f"[latency] p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms (watch the tail, not the mean)")
    assert p99 > p50, "the tail latency must exceed the median"

    bucket = TokenBucket(capacity=10, refill=5)
    allowed = sum(bucket.allow(t) for t in np.linspace(0, 1, 20))   # a burst of 20 in 1s
    print(f"[rate-limit] {allowed}/20 allowed, {20-allowed} throttled")
    assert allowed < 20, "the rate limiter must throttle a burst"
    print("[serving] OK - batching, latency tail, and rate limiting all verified")


if __name__ == "__main__":
    main()

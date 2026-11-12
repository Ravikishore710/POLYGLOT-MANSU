"""
src/polyglot_mansu/eval/roi_benchmark.py
Enterprise Edge ROI & Latency Benchmarking:
Compares self-hosted 4-bit Tiny Aya edge deployment vs. closed frontier APIs + guardrails.
"""

import time
from typing import Dict, Any

def benchmark_throughput(
    generate_fn,
    num_tokens: int = 128,
    warmup: int = 2
) -> Dict[str, float]:
    """Measures generation throughput in tokens per second."""
    # Warmup
    for _ in range(warmup):
        generate_fn(max_new_tokens=16)

    start = time.perf_counter()
    tokens_generated = generate_fn(max_new_tokens=num_tokens)
    elapsed = time.perf_counter() - start

    actual_tok = tokens_generated if isinstance(tokens_generated, int) else num_tokens
    tok_per_sec = actual_tok / max(1e-5, elapsed)

    return {
        "tokens_generated": actual_tok,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tok_per_sec, 2)
    }

def calculate_roi_comparison(
    measured_tok_per_sec: float = 85.0,
    gpu_hourly_cost: float = 0.35,  # e.g. T4 / L4 hourly rate on cloud
    closed_api_cost_per_million: float = 5.00,  # e.g. GPT-4o / Claude 3.5
    guardrail_tax_per_million: float = 0.75     # dual-pass moderation classifier cost
) -> Dict[str, Any]:
    """
    Computes enterprise cost per 1M tokens and monthly savings.
    """
    # Self-hosted cost per 1M tokens
    seconds_for_1m = 1_000_000.0 / max(1.0, measured_tok_per_sec)
    hours_for_1m = seconds_for_1m / 3600.0
    self_hosted_cost_1m = hours_for_1m * gpu_hourly_cost

    total_closed_api_cost_1m = closed_api_cost_per_million + guardrail_tax_per_million

    # For 50M tokens/month enterprise workload
    monthly_tokens = 50_000_000
    monthly_cost_self = (monthly_tokens / 1_000_000.0) * self_hosted_cost_1m
    monthly_cost_api = (monthly_tokens / 1_000_000.0) * total_closed_api_cost_1m
    monthly_savings = monthly_cost_api - monthly_cost_self
    savings_pct = (monthly_savings / max(1e-5, monthly_cost_api)) * 100.0

    return {
        "measured_throughput_tok_s": measured_tok_per_sec,
        "self_hosted_cost_per_1m_tokens": round(self_hosted_cost_1m, 4),
        "closed_api_cost_per_1m_tokens": round(total_closed_api_cost_1m, 4),
        "cost_reduction_ratio": round(total_closed_api_cost_1m / max(1e-5, self_hosted_cost_1m), 2),
        "monthly_tokens_volume": monthly_tokens,
        "monthly_self_hosted_cost": round(monthly_cost_self, 2),
        "monthly_closed_api_cost": round(monthly_cost_api, 2),
        "monthly_savings_dollars": round(monthly_savings, 2),
        "savings_percentage": round(savings_pct, 1)
    }

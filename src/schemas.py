
from typing import Optional

from pydantic import BaseModel


class ChunkTelemetry(BaseModel):
    """Token usage, performance, and cost metrics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_cost_usd: float = 0.0


class StreamChunk(BaseModel):
    """A normalized chunk returned by an LLM provider."""

    delta_text: str = ""
    provider: str
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None
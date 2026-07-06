"""Model size specifications from Table 1 of the assignment handout.

Throughout assignment 2 we benchmark/profile a fixed family of Transformer LMs
(mostly GPT-2 configs). Shared defaults: vocab_size=10000, batch_size=4,
context_length=512 unless otherwise specified.
"""

from __future__ import annotations

from dataclasses import dataclass

# Shared benchmarking defaults (handout §2.1.2).
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


# Table 1: specifications of different model sizes.
MODEL_SIZES: dict[str, ModelConfig] = {
    "small":  ModelConfig(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  ModelConfig(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

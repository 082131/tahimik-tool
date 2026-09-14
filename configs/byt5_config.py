# Configuration for standard uncompressed ByT5 baseline.

from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class ByT5Config(BaseConfig):
    """ByT5 configuration without byte compression."""

    variant_name: str = "byt5_baseline"
    model_name: str = "google/byt5-small"

    # No compression — every byte is processed by every encoder layer.
    use_compression: bool = False

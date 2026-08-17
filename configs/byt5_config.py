# =============================================================================
# Configuration for the ByT5 baseline (no compression).
#
# This is the first baseline model. It uses the full ByT5 architecture
# without any byte-level compression, establishing the normalization-accuracy
# ceiling against which MrT5 and TAHIMIK are compared.
# =============================================================================

from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class ByT5Config(BaseConfig):
    """ByT5 with no compression — the accuracy ceiling baseline."""

    variant_name: str = "byt5_baseline"

    # No compression — every byte is processed by every encoder layer.
    use_compression: bool = False

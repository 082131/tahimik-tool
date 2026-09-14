from pathlib import Path
import pytest
from configs.base import BaseConfig
from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig


def test_base_config_pins_mrt5_small_as_authoritative():
    base = BaseConfig()
    assert base.model_name == "stanfordnlp/mrt5-small"


def test_byt5_uses_google_small_while_compressed_variants_use_mrt5_small():
    assert ByT5Config().model_name == "google/byt5-small"
    assert MrT5Config().model_name == "stanfordnlp/mrt5-small"
    assert TAHIMIKConfig().model_name == "stanfordnlp/mrt5-small"

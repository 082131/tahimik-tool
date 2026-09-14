from pathlib import Path
import pytest
from configs.base import BaseConfig
from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig


def test_base_config_defaults_to_the_stanford_mrt5_small_backbone():
    base = BaseConfig()
    assert base.model_name == "stanfordnlp/mrt5-small"


def test_variants_resolve_to_their_intended_small_backbones():
    byt5 = ByT5Config()
    mrt5 = MrT5Config()
    tahimik = TAHIMIKConfig()

    assert byt5.model_name == "google/byt5-small"
    assert mrt5.model_name == "stanfordnlp/mrt5-small"
    assert tahimik.model_name == "stanfordnlp/mrt5-small"

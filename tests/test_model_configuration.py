from pathlib import Path
import pytest
from configs.base import BaseConfig
from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig


def test_base_config_pins_byt5_base_as_authoritative():
    base = BaseConfig()
    assert base.model_name == "google/byt5-base"


def test_all_variants_use_byt5_base():
    configs = [ByT5Config(), MrT5Config(), TAHIMIKConfig()]
    assert {config.model_name for config in configs} == {"google/byt5-base"}


def test_experiment_has_no_hardcoded_small_identifier():
    source = Path("scripts/run_experiment.py").read_text(encoding="utf-8")
    assert 'from_pretrained("google/byt5-small")' not in source
    assert 'google/byt5-small' not in source

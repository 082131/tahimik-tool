from pathlib import Path
import pytest
from configs.base import BaseConfig
from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig


def test_base_config_pins_byt5_base_as_authoritative():
    base = BaseConfig()
    assert base.model_name == "google/byt5-base"


def test_all_variants_inherit_byt5_base():
    byt5 = ByT5Config()
    mrt5 = MrT5Config()
    tahimik = TAHIMIKConfig()

    assert byt5.model_name == "google/byt5-base"
    assert mrt5.model_name == "google/byt5-base"
    assert tahimik.model_name == "google/byt5-base"


def test_run_experiment_script_contains_no_byt5_small():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py"
    content = script_path.read_text(encoding="utf-8")
    assert "google/byt5-small" not in content, "run_experiment.py still hardcodes google/byt5-small"

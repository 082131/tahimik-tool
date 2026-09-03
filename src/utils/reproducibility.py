import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone

def _git(command):
    try: return subprocess.run(["git", *command], capture_output=True, text=True, check=True).stdout.strip()
    except Exception: return "unknown"

def collect_run_metadata(seed=None, config=None):
    dirty = bool(_git(["status", "--porcelain"]))
    packages = {}
    for name in ("numpy", "scipy", "torch", "transformers", "sacrebleu", "nltk", "editdistance"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_sha": _git(["rev-parse", "HEAD"]), "dirty": dirty, "seed": seed, "resolved_config": config or {}, "python": platform.python_version(), "platform": platform.platform(), "packages": packages}

def configure_determinism(seed: int):
    import os, random
    import numpy as np
    random.seed(seed); np.random.seed(seed); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        return True
    except ImportError: return False

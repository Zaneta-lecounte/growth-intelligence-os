"""Central configuration: model name, paths, demo-mode detection."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass

ROOT = Path(__file__).resolve().parent.parent

# Current Claude Sonnet model (see Anthropic models overview).
MODEL = os.getenv("GIOS_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 16000

REPO_URL = os.getenv("GIOS_REPO_URL", "https://github.com/Zaneta-lecounte/growth-intelligence-os")

PROMPTS_DIR = ROOT / "prompts"
DEMO_DIR = ROOT / "demo"
DATA_DIR = ROOT / "data" / "synthetic"
SPECS_DIR = ROOT / "specs"
DB_PATH = Path(os.getenv("GIOS_DB_PATH", ROOT / "gios.db"))


def is_demo_mode() -> bool:
    """DEMO MODE is on whenever no Anthropic API key is configured."""
    return not os.getenv("ANTHROPIC_API_KEY", "").strip()

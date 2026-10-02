import json

from gios import config
from scripts import build_demo_cache


def test_offline_demo_cache_is_fresh(tmp_path, monkeypatch):
    """Rebuilding the offline cache must reproduce the committed demo/ files exactly."""
    monkeypatch.setattr(config, "DEMO_DIR", tmp_path)
    for builder in build_demo_cache.BUILDERS.values():
        builder(False)
    built = sorted(p.name for p in tmp_path.glob("*.json"))
    assert built, "no demo files built"
    for name in built:
        committed = config.ROOT / "demo" / name
        assert committed.exists(), f"demo/{name} missing; run scripts/build_demo_cache.py"
        assert json.loads(committed.read_text()) == json.loads((tmp_path / name).read_text()), \
            f"demo/{name} is stale; run scripts/build_demo_cache.py"

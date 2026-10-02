"""Streamlit Community Cloud readiness: entry point, pinned dependencies, committed data and caches."""
import re
import subprocess

from gios import config

ROOT = config.ROOT


def test_entrypoint_is_app_py_with_navigation():
    app = (ROOT / "app.py").read_text()
    assert "st.navigation(" in app and ".run()" in app


def test_runtime_requirements_are_pinned():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines and all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[0-9][\w.]*", l) for l in lines), lines
    names = {l.split("==")[0].lower() for l in lines}
    assert {"streamlit", "pydantic", "pandas", "scipy", "plotly", "anthropic", "numpy"} <= names
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert "-r requirements.txt" in dev and re.search(r"pytest==", dev)


def test_pins_match_the_installed_versions():
    from importlib.metadata import version

    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, pinned = line.strip().split("==")
            assert version(name) == pinned, f"{name}: installed {version(name)} but pinned {pinned}"


def test_data_and_demo_cache_are_committed():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for name in ["funnel_by_source", "web_behavior", "customer_evidence", "sales_feedback", "experiments",
                 "experiment_segments", "lead_velocity"]:
        assert f"data/synthetic/{name}.csv" in tracked
    assert len([l for l in tracked.splitlines() if l.startswith("demo/") and l.endswith(".json")]) >= 20
    assert ".streamlit/config.toml" in tracked


def test_secrets_and_local_state_are_ignored():
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in [".env", "*.db", "__pycache__/", ".streamlit/secrets.toml"]:
        assert pattern in ignore


def test_no_absolute_paths_in_code():
    for path in list(ROOT.glob("gios/**/*.py")) + list(ROOT.glob("views/*.py")) + [ROOT / "app.py", ROOT / "home.py"]:
        assert "/home/" not in path.read_text(), path

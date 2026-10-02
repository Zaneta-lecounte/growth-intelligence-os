from gios import config
from gios.modules import MODULES

README = (config.ROOT / "README.md").read_text()


def test_readme_lists_every_module_in_its_layer():
    for m in MODULES:
        row = next((l for l in README.splitlines() if l.startswith(f"| {m.name} |")), None)
        assert row, m.name
        assert f"| {m.layer.title()} |" in row


def test_readme_has_required_sections():
    for heading in ["## The problem", "## The solution", "## Architecture", "## Modules", "## Design principles",
                    "## Synthetic data", "## Screenshots", "## How to run", "## Running with an API key", "## Tech",
                    "## Limitations", "## Future improvements"]:
        assert heading in README
    assert "```mermaid" in README
    for item in ["Warehouse / SQL source", "GA4 connector", "CRM sync", "Slack alerts"]:
        assert item in README

"""Command line interfaces for hazard.

`osc_hazard` (hazard.cli.main) onboards hazards generically; `os_climate_hazard`
(hazard.cli.cli_legacy) supports the legacy per-indicator commands.
"""

from hazard.cli.main import app

__all__ = ["app"]

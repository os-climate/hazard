"""Cli for hazard."""

import json
from typing import List, Optional

from hazard import get_hazards_onboarding
import hazard.utilities.cli_utilities as cli_u
from hazard.onboard.general_onboarding import onboard_hazards
import typer


app = typer.Typer(no_args_is_help=True)

credentials_option_default = typer.Option(
    False,
    help="If you want to make the onboard to your local machine",
)

hazards_option_default = typer.Argument(
    ...,
    help=f"Hazards to be onboarded: {cli_u.hazards_help_text()}",
    autocompletion=cli_u.autocomplete_hazards,
)


@app.command()
def list_hazards():
    """List all available hazards."""
    hazards = get_hazards_onboarding().keys()
    for haz in sorted(hazards):
        print(haz)


@app.command()
def onboard(
    local: Optional[bool] = credentials_option_default,
    credentials_path: Optional[str] = typer.Option(
        None, help="Path to the credentials for S3 access."
    ),
    source_dir_base: Optional[str] = typer.Option(None, help="Path to the data."),
    hazards: List[str] = hazards_option_default,
    download_dir: Optional[str] = typer.Option(None, help="Path to the data."),
    store: Optional[str] = typer.Option(
        None,
        help="Path of a local directory to onboard to (zarr store and inventory). "
        "Takes precedence over --local and --credentials-path.",
    ),
    hazard_kwargs: Optional[str] = typer.Option(
        None,
        help="JSON object of keyword arguments for the constructor of each hazard's onboarder, e.g. "
        '\'{"run": {"gcms": ["ACCESS-CM2"], "scenarios": ["ssp585"], "central_years": [2030]}, '
        '"dask_cluster_kwargs": {"n_workers": 1, "threads_per_worker": 2}}\'.',
    ),
):
    """Onboard desired hazards."""
    local = bool(local)

    kwargs = None
    if hazard_kwargs is not None:
        try:
            kwargs = json.loads(hazard_kwargs)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(
                f"not valid JSON: {e}", param_hint="--hazard-kwargs"
            )
        if not isinstance(kwargs, dict):
            raise typer.BadParameter(
                "must be a JSON object", param_hint="--hazard-kwargs"
            )

    onboard_hazards(
        local,
        credentials_path,
        source_dir_base,
        hazards,
        download_dir,
        store=store,
        hazard_kwargs=kwargs,
    )


if __name__ == "__main__":
    app()

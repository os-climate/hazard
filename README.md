# Quick start

## Installation

Clone the repository :

```
git clone git@github.com:os-climate/hazard.git
cd hazard
```

Then use either `pdm` (recommended):

```
pip install pdm
pdm config venv.with_pip True
pdm install
```

Or `virtualenv`:

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

A command line interface is exposed with the package. For example, this code snippet will run a cut-down version of a "days above temperature" indicator and write the output to `$HOME/hazard_example` :

```
source .venv/bin/activate
mkdir -p $HOME/hazard_example
os_climate_hazard days_tas_above_indicator --store $HOME/hazard_example
```

# Contributing

Patches may be contributed via pull requests from forks to
https://github.com/os-climate/hazard.

All changes must pass the automated test suite, along with various static checks.

The easiest way to run these is via:
```
pdm run all
```

# Hazard modelling

For more modelling-specific information, see `HAZARD.md`.

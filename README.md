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

### In a virtual environment

A command line interface is exposed with the package. For example, this code snippet will run a cut-down version of a "days above temperature" indicator and write the output to `$HOME/hazard_example` :

```
source .venv/bin/activate
mkdir -p $HOME/hazard_example
os_climate_hazard days_tas_above_indicator --store $HOME/hazard_example
```

### In a docker container

First, build the image.

```
docker build -t os-hazard-indicator -f dockerfiles/Dockerfile .
```

Then, you can run an example the following way. In the example, we save the data locally to /data/hazard-test-container in the container. To have access to the output once the container finished running, we are mounting `/data` from the container to `$HOME/data` locally.

```
docker run -it -v $HOME/data:/data os-hazard-indicator os_climate_hazard days_tas_above_indicator --store /data/hazard-test-container
```

### In a CWL (Common Workflow Language) workflow

## Arguments parsing in CLI

The CLI for this package is built on top of google's `fire` package. Arguments passed to the command line are parsed not based on their declared type in python but their string value at runtime. For complex types such as lists, this can lead to confusion if internal list elements have special characters like hyphens.

This is an example of command that _works_ for the argument `gcm_list` (note the single and double quotes in that argument value)

```
os_climate_hazard degree_days_indicator --store $HOME/data/hazard-test --scenario_list [ssp126,ssp585] --central_year_list [2070,2080] --window_years 1 --gcm_list "['ACCESS-CM2','NorESM2-MM']"
```

And this is an example that does not :

```
os_climate_hazard degree_days_indicator --store $HOME/data/hazard-test --scenario_list [ssp126,ssp585] --central_year_list [2070,2080] --window_years 1 --gcm_list [ACCESS-CM2,NorESM2-MM]
```

# Contributing

Patches may be contributed via pull requests from forks to
<https://github.com/os-climate/hazard>.

All changes must pass the automated test suite, along with various static checks.

The easiest way to run these is via:

```
pdm run all
```

# Hazard modelling

For more modelling-specific information, see `HAZARD.md`.

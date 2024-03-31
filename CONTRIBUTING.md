# Contributing to hazard

## Getting started
To get set up, clone and enter the repo.
```
git clone git@github.com:os-climate/hazard.git
cd hazard
```

We recommend using [pdm](https://pdm-project.org/latest/) for a
consistent working environment. Install via, e.g.:
```
pip install pdm
```

For use with Jupyter and mypy, the configuration is needed:
```
pdm config venv.with_pip True
```

The command:
```
pdm install
```
will create a virtual environment (typically .venv folder in the project folder) and install the dependencies.
We recommend that the IDE workspace uses this virtual environment when developing.

When adding a package for use in new or improved functionality,
`pdm add <package-name>`. Or, when adding something helpful for
testing or development, `pdm add -dG <group> <package-name>`.

### JupyterHub and requirements.txt
It may be useful to generate a requirements.txt file:
```
pdm export -o requirements.txt --without-hashes
```

## Development
Patches may be contributed via pull requests to
https://github.com/os-climate/hazard.

All changes must pass the automated test suite, along with various static
checks.

The easiest way to run these is via:
```
pdm run all
```

[Black](https://black.readthedocs.io/) code style and
[isort](https://pycqa.github.io/isort/) import ordering are enforced
and enabling automatic formatting via [pre-commit](https://pre-commit.com/)
is recommended:
```
pre-commit install
```

To ensure compliance with static check tools, developers may wish to run black and isort against modified files.

E.g.,
```
# auto-sort imports
isort .
# auto-format code
black .
```

## IDE set-up
For those using VS Code, configure tests ('Python: Configure Tests') to use 'pytest'
to allow running of tests within the IDE.

## Releasing
Actions are configured to release to PyPI on pushing a tag. In order to do this:
- Update VERSION
- Create new annotated tag and push
```
git tag -a v1.0.0 -m "v1.0.0"
git push --follow-tags
```

## Forking workflow
This is a useful clarification of the forking workflow:
https://gist.github.com/Chaser324/ce0505fbed06b947d962

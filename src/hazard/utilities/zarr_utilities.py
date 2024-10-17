"""zarr_utilities is used to load environment variables from a credentials dotenv file."""

import os
import pathlib

from dotenv import load_dotenv


def set_credential_env_variables():
    """Load environment variables from a credentials dotenv file.

    This function looks for a `credentials.env` file in the directory specified
    by the `CREDENTIAL_DOTENV_DIR` environment variable. If `CREDENTIAL_DOTENV_DIR`
    is not set, it defaults to the current working directory. The environment variables
    from the file are then loaded and can override existing variables.

    Notes
        - The `credentials.env` file must exist in the specified directory.
        - Environment variables are loaded using `load_dotenv` with `override=True`.

    Environment Variables:
        CREDENTIAL_DOTENV_DIR (str, optional):
            Path to the directory containing the `credentials.env` file.
            Defaults to the current working directory.

    Examples
        To load credentials from a custom directory:
        >>> import os
        >>> os.environ["CREDENTIAL_DOTENV_DIR"] = "/path/to/credentials"
        >>> set_credential_env_variables()

    """
    dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
    dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)

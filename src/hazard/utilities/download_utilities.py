"""Module with utilities for downloading files."""

import os
import re
import tarfile
import zipfile
from typing import Optional

import requests


def download_file(
    url: str, directory: str, filename: Optional[str] = None, force_download=False
):
    """Download a file in chunks."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with requests.get(url, stream=True, verify=False) as r:
        if filename is None:
            filename = get_filename_from_cd(r.headers["content-disposition"])
            if not filename:
                raise ValueError(
                    "filename not provided and cannot infer from content-disposition"
                )

        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path) and not force_download:
            print(f"The file {filename} already exists.")
            return filename

        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return filename


def download_and_unzip(
    url: str,
    dir: str,
    archive_name: str,
    overwrite: bool = False,
    extension: str = ".zip",
):
    """Download a file and unzip."""
    unzip_file = os.path.join(dir, archive_name)
    if not overwrite and os.path.exists(unzip_file):
        return
    file = os.path.join(dir, archive_name + extension)
    if not os.path.exists(file):
        download_file(url, dir, filename=archive_name + extension)

    if tarfile.is_tarfile(file):
        with tarfile.open(file, "r") as t:
            t.extractall(unzip_file)
    else:
        with zipfile.ZipFile(file, "r") as z:
            z.extractall(unzip_file)


def get_filename_from_cd(content_disp):
    """Get filename from content-disposition."""
    if not content_disp:
        return None
    filename = re.findall("filename=(.+)", content_disp)
    if len(filename) == 0:
        return None
    return filename[0]

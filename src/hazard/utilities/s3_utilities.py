"""Module for managing the transfer and synchronization of Zarr arrays between S3 buckets.

This module contains functions for copying, synchronizing, and listing objects
in different S3 storage environments, including development, production, and public
buckets. It utilizes the `boto3` and `s3fs` libraries for interacting with Amazon S3
and handles parallelized copy operations.
"""

import asyncio
import concurrent.futures
from glob import iglob
import logging
import os
import pathlib
import sys
from typing import Callable, Optional, Sequence

try:
    ## python >3.11 includes tomllib
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import boto3
import botocore.client
import s3fs
from fsspec import FSMap

logger = logging.getLogger(__name__)

# We are considereing two main storage buckets, dev and production.
# The env vars for accessing them are
#     __access_key = "OSC_S3_ACCESS_KEY"
#     __endpoint_url = "OSC_S3_ENDPOINT"
#     __secret_key = "OSC_S3_SECRET_KEY"
#     __token = "OSC_S3_TOKEN"
# appending _DEV for the dev bucket.


def get_s3_fs(
    use_dev: bool = True, extra_s3fs_kwargs: Optional[dict] = None, **kwargs
) -> s3fs.S3FileSystem:
    """Return a S3FileSystem object.

    Args:
        use_dev : bool=True
            Use the "_DEV" ending env vars.

        extra_s3fs_kwargs : Optional[dict]
            Extra keyword arguments that will be passed to S3FileSystem.
            They will override the parameters extracted from envvars.
        kwargs: dict
            Extra keyword arguments. extra_s3fs_kwargs will be updated with kwargs.

    """
    __access_key = "OSC_S3_ACCESS_KEY"
    __endpoint_url = "OSC_S3_ENDPOINT"
    __secret_key = "OSC_S3_SECRET_KEY"
    __token = "OSC_S3_TOKEN"

    suffix = "_DEV" if use_dev else ""

    s3fsparams = {
        "key": os.environ.get(f"{__access_key}{suffix}", None),
        "secret": os.environ.get(f"{__secret_key}{suffix}", None),
        "token": os.environ.get(f"{__token}{suffix}", None),
        "endpoint_url": os.environ.get(f"{__endpoint_url}{suffix}", None),
    }
    if extra_s3fs_kwargs is None:
        extra_s3fs_kwargs = {}
    extra_s3fs_kwargs.update(kwargs)

    s3fsparams.update(extra_s3fs_kwargs)
    return s3fs.S3FileSystem(**s3fsparams)


def get_store(
    s3: Optional[s3fs.S3FileSystem] = None,
    use_dev: bool = True,
    extra_s3fs_kwargs: Optional[dict] = None,
    bucket: Optional[str] = None,
    group_path_suffix: str = "hazard/hazard.zarr",
    *_,
) -> FSMap:
    """Return the FSMap object from s3fs.S3Map.

    Args:
        s3: Optional[s3fs.S3FileSystem] = None
            S3Filesystem to use.
        use_dev: bool=True
            Use the "_DEV" ending env vars.
        extra_s3fs_kwargs: dict
            Extra keyword arguments that will be passed to S3FileSystem.
            They will override the parameters extracted from envvars.
        bucket: Optional[str] = None
            bucket to use. If not provided, the value from the envvar
            (OSC_S3_BUCKET or OSC_S3_BUCKET_DEV) will be used.
        group_path_suffix: str = "hazard/hazard.zarr"
            The root zarr group is by convention `${bucket}/hazard/hazard.zarr`.
            This argument allows changing the `hazard/hazard.zarr` part.

    """
    __s3_bucket = "OSC_S3_BUCKET"

    if extra_s3fs_kwargs is None:
        extra_s3fs_kwargs = {}

    if not s3:
        s3 = get_s3_fs(use_dev=use_dev, **extra_s3fs_kwargs)

    if not bucket:
        suffix = "_DEV" if use_dev else ""
        bucket = os.environ.get(f"{__s3_bucket}{suffix}", "")

    group_path = str(pathlib.PurePosixPath(bucket, group_path_suffix))
    store = s3fs.S3Map(root=group_path, s3=s3, check=False)

    return store


def load_s3_parameters(toml_path: Optional[str] = None) -> dict:
    """Load the parameters (mostly credentials) for the S3 utilities from a TOML file.

    The env_vars_override dictionary will be poped and use to override the environment.

    Args:
        toml_path (str): path to the toml file.

    Returns:
        Dict[str, Any]: Dict that can be passed via ** to s3_utilities.get_s3_fs and s3_utilities.get_store.

    Example:
        toml_file contents:

        ```
            # Parameters for s3
            bucket = "my-hazard-bucket"

            [extra_s3fs_kwargs]
            key = "000000000000000000000"
            secret = "secretsecretkey"
            endpoint_url = "https://s3.example.com"

            # env vars to override
            [env_vars_override]
            OSC_S3_ACCESS_KEY = "000000000000000000000"
            OSC_S3_SECRET_KEY = "secretsecretkey"
            OSC_S3_BUCKET = "my-hazard-bucket"

            WISC_CDSAPI_URL = "https://copernicus.example.com"
            WISC_CDSAPI_KEY = "cdskeytoaccess"
        ```

    """
    if toml_path is None or toml_path == "":
        return {}

    with open(toml_path, "rb") as f:
        tomlcontents = tomllib.load(f)

    ## Override the environment with the contents of tomlcontents[env_vars_override]
    dot_env_dict = tomlcontents.pop("env_vars_override", {})

    os.environ.update(dot_env_dict)

    return tomlcontents


def copy_local_to_dev(zarr_dir: str, array_path: str, dry_run=False):
    """Copy zarr array from a local directory to the development bucket.
    Requires environment variables:
    OSC_S3_BUCKET_DEV=physrisk-hazard-indicators-dev01
    OSC_S3_ACCESS_KEY_DEV=...
    OSC_S3_SECRET_KEY_DEV=...

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_path (str): The path of the array or data set within the group (i.e. all files that are children are to be copied).
        dry_run (bool, optional): If True, log the action that would
        be taken without actually executing. Defaults to False.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename="batch.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    s3_target_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
        config=botocore.client.Config(max_pool_connections=32),
    )
    target_bucket_name = os.environ["OSC_S3_BUCKET_DEV"]
    logger.info(f"Source path {zarr_dir}; target bucket {target_bucket_name}")

    root_path = pathlib.Path(zarr_dir)
    relative_file_paths = [
        pathlib.Path(f).as_posix()
        for f in iglob(
            str(pathlib.PurePosixPath(array_path) / "**"),
            root_dir=str(root_path),
            recursive=True,
        )
        if (root_path / f).is_file()
    ]
    relative_file_paths = relative_file_paths + [
        pathlib.Path(f).as_posix()
        for f in iglob(
            str(pathlib.PurePosixPath(array_path) / "**/.*"),
            root_dir=str(root_path),
            recursive=True,
        )
        if (root_path / f).is_file()
    ]

    # files = [f for f in pathlib.Path(zarr_dir, array_path).iterdir() if f.is_file()]
    logger.info(f"Copying {len(relative_file_paths)} files in array {array_path}")

    def copy_file(path: pathlib.Path):
        # path is the relative path with respect to hazard.zarr
        with open(root_path / path, "rb") as f:
            data = f.read()
            target_key = str(pathlib.PurePosixPath("hazard", "hazard.zarr", path))
            s3_target_client.put_object(
                Body=data, Bucket=target_bucket_name, Key=target_key
            )

    async def copy_all():
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(executor, copy_file, path)
            for path in relative_file_paths
        ]

        completed = []
        for coro in asyncio.as_completed(futures):
            completed.append(await coro)
            if len(completed) % 100 == 0:
                logger.info(f"Completed {len(completed)}/{len(relative_file_paths)}")

    if not dry_run:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(copy_all())


def copy_dev_to_prod(prefix: str, dry_run=False, sync=True):
    """Copy files with a specified prefix from development S3 to production S3.

    Args:
        prefix : str
            The prefix of the files to copy.
        dry_run : bool, optional
            If True, log actions without executing them. Defaults to False.
        sync : bool, optional
            If True, perform synchronization based on ETag differences. Defaults to True.

    """

    def rename(source_prefix: str):
        return source_prefix.replace(
            "hazard/hazard.zarr", "hazard-indicators/hazard.zarr", 1
        )

    s3_source_client = boto3.client(
        service_name="s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
        endpoint_url=os.environ.get("OSC_S3_ENDPOINT_DEV", None),
        config=botocore.client.Config(max_pool_connections=32),
    )
    s3_target_client = boto3.client(
        service_name="s3",
        endpoint_url=os.environ.get("OSC_S3_ENDPOINT", None),
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY", None),
        config=botocore.client.Config(max_pool_connections=32),
    )

    source_bucket_name = os.environ["OSC_S3_BUCKET_DEV"]
    target_bucket_name = os.environ["OSC_S3_BUCKET"]
    if (
        source_bucket_name != "physrisk-hazard-indicators-dev01"
        or target_bucket_name != "os-climate-physical-risk"
    ):
        # double check on environment variables
        raise ValueError("unexpected bucket")
    if sync:
        sync_buckets(
            s3_source_client,
            source_bucket_name,
            s3_target_client,
            target_bucket_name,
            prefix=prefix,
            dry_run=dry_run,
            rename=rename,
        )
    else:
        keys, size = list_objects(s3_source_client, source_bucket_name, prefix)
        logger.info(
            f"Prefix {prefix} {len(keys)} objects with total size {size / 1e9}GB"
        )
        logger.info(
            f"Copying from bucket {source_bucket_name} to bucket {target_bucket_name}"
        )
        if not dry_run:
            copy_objects(
                keys,
                s3_source_client,
                source_bucket_name,
                s3_target_client,
                target_bucket_name,
                rename=rename,
            )


def copy_prod_to_public(prefix: str, dry_run=False, sync=True):
    """Copy files with a specified prefix from production S3 to public S3.

    Args:
        prefix : str
            The prefix of the files to copy.
        dry_run : bool, optional
            If True, log actions without executing them. Defaults to False.
        sync : bool, optional
            If True, perform synchronization based on ETag differences. Defaults to True.

    """
    s3_source_client = boto3.client(
        service_name="s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY", None),
    )
    s3_target_client = boto3.client(
        service_name="s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY_PUBLIC", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY_PUBLIC", None),
    )

    source_bucket_name = os.environ["OSC_S3_BUCKET"]
    target_bucket_name = os.environ["OSC_S3_BUCKET_PUBLIC"]
    if (
        source_bucket_name != "physrisk-hazard-indicators"
        or target_bucket_name != "os-climate-public-data"
    ):
        # double check on environment variables
        raise ValueError("unexpected bucket")
    if sync:
        sync_buckets(
            s3_source_client,
            source_bucket_name,
            s3_target_client,
            target_bucket_name,
            prefix=prefix,
            dry_run=dry_run,
        )
    else:
        keys, size = list_objects(s3_source_client, source_bucket_name, prefix)
        logger.info(
            f"Prefix {prefix} {len(keys)} objects with total size {size / 1e9}GB"
        )
        logger.info(
            f"Copying from bucket {source_bucket_name} to bucket {target_bucket_name}"
        )
        if not dry_run:
            copy_objects(
                keys,
                s3_source_client,
                source_bucket_name,
                s3_target_client,
                target_bucket_name,
            )


def list_objects(client, bucket_name, prefix):
    """List objects in an S3 bucket with a given prefix.

    Args:
        client : boto3.Client
            The S3 client used for listing objects.
        bucket_name : str
            The name of the S3 bucket.
        prefix : str
            The prefix to filter objects by.

    Returns:
        list
            List of object keys matching the prefix.
        int
            Total size of the listed objects in bytes.

    """
    paginator = client.get_paginator(
        "list_objects_v2"
    )  # , PaginationConfig={"MaxItems": 10000})
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # get the list of keys with the given prefix
    keys = []
    size = 0
    for page in pages:
        for objs in page.get("Contents", []):
            if isinstance(objs, list):
                for obj in objs:
                    keys.append(obj["Key"])
                    size += obj["Size"]
            else:
                keys.append(objs["Key"])
                size += objs["Size"]
    return keys, size


def list_object_etags(client, bucket_name, prefix):
    """List ETags of objects in an S3 bucket with a given prefix.

    Args:
        client : boto3.Client
            The S3 client used for listing objects.
        bucket_name : str
            The name of the S3 bucket.
        prefix : str
            The prefix to filter objects by.

    Returns:
        dict
            A dictionary of object keys and their ETags.

    """
    paginator = client.get_paginator(
        "list_objects_v2"
    )  # , PaginationConfig={'MaxItems': 10000})
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # get the list of keys with the given prefix
    etags = {}
    for page in pages:
        contents = page.get("Contents", [])
        for objs in contents:
            if isinstance(objs, list):
                for obj in objs:
                    etags[obj["Key"]] = obj["ETag"]
            else:
                etags[objs["Key"]] = objs["ETag"]
    return etags


def copy_objects(
    keys: Sequence[str],
    s3_source_client,
    source_bucket_name: str,
    s3_target_client,
    target_bucket_name: str,
    rename: Optional[Callable[[str], str]] = None,
):
    """Copy objects from one S3 bucket to another.

    Args:
        keys : Sequence[str]
            List of keys to copy.
        s3_source_client : boto3.Client
            Source S3 client.
        source_bucket_name : str
            Source S3 bucket name.
        s3_target_client : boto3.Client
            Target S3 client.
        target_bucket_name : str
            Target S3 bucket name.
        rename : Optional[Callable[[str], str]], optional
            Function to rename keys during the copy. Defaults to None.

    """
    logger.info(
        f"Source bucket {source_bucket_name}; target bucket {target_bucket_name}"
    )

    def copy_object(key):
        obj = s3_source_client.get_object(Bucket=source_bucket_name, Key=key)
        data = obj["Body"].read()
        target_key = rename(key) if rename is not None else key
        # target_key = key.replace('hazard_test/hazard.zarr', 'hazard/hazard.zarr')
        return s3_target_client.put_object(
            Body=data, Bucket=target_bucket_name, Key=target_key
        )

    async def copy_all(keys: Sequence[str]):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(executor, copy_object, key) for key in keys]

        completed = []
        for coro in asyncio.as_completed(futures):
            completed.append(await coro)
            if len(completed) % 100 == 0:
                logger.info(f"Completed {len(completed)}/{len(keys)}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(copy_all(keys))
    logger.info("Completed.")


def remove_objects(keys: Sequence[str], s3_client, bucket_name: str):
    """Remove objects from an S3 bucket.

    Args:
        keys : Sequence[str]
            List of object keys to remove.
        s3_client : boto3.Client
            S3 client for deletion.
        bucket_name : str
            Name of the S3 bucket.

    """
    for i, key in enumerate(keys):
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        if i % 100 == 0:
            logger.info(f"Completed {i}/{len(keys)}")


def remove_from_prod(prefix: str, dry_run=True):
    """Remove objects from the production S3 bucket based on a prefix.

    Args:
        prefix : str
            The prefix to filter objects for removal.
        dry_run : bool, optional
            If True, log actions without executing them. Defaults to True.

    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY", None),
    )
    bucket_name = os.environ["OSC_S3_BUCKET"]
    keys, size = list_objects(s3_client, bucket_name, prefix)
    logger.info(f"Prefix {prefix} {len(keys)} objects with total size {size / 1e9}GB")
    logger.info(f"Removing from bucket {bucket_name}")
    if not dry_run:
        remove_objects(keys, s3_client, bucket_name)


def sync_buckets(
    s3_source_client,
    source_bucket_name: str,
    s3_target_client,
    target_bucket_name: str,
    prefix: str,
    dry_run=True,
    rename: Optional[Callable[[str], str]] = None,
):
    """Synchronize files between two S3 buckets based on ETag differences.

    Args:
        s3_source_client : boto3.Client
            Source S3 client.
        source_bucket_name : str
            Source S3 bucket name.
        s3_target_client : boto3.Client
            Target S3 client.
        target_bucket_name : str
            Target S3 bucket name.
        prefix : str
            Prefix of objects to synchronize.
        dry_run : bool, optional
            If True, log actions without executing them. Defaults to True.

    """
    if rename is None:

        def _rename(s):
            return s

        rename = _rename
    logger.info(
        f"Syncing target bucket {target_bucket_name} to source {source_bucket_name}."
    )

    prefix_target = rename(prefix) if rename is not None else prefix
    source_etags = list_object_etags(s3_source_client, source_bucket_name, prefix)
    target_etags = list_object_etags(
        s3_target_client, target_bucket_name, prefix_target
    )
    # look
    # look for objects that are 1) missing in target and 2) different in target
    all_diffs = set(
        key
        for key, etag in source_etags.items()
        if target_etags.get(rename(key), "") != etag
    )
    missing = set(key for key in source_etags if rename(key) not in target_etags)
    different = set(key for key in all_diffs if key not in missing)
    logger.info(
        f"Checked {len(source_etags)} files from {source_bucket_name} against {target_bucket_name}"  # noqa:W503
    )
    logger.info(
        f"Copying {len(missing)} missing files from {source_bucket_name} to {target_bucket_name}: "
        + _first_5_last_5(list(missing))  # noqa:W503
    )
    if not dry_run:
        copy_objects(
            list(missing),
            s3_source_client,
            source_bucket_name,
            s3_target_client,
            target_bucket_name,
            rename=rename,
        )
    logger.info(
        f"Copying {len(different)} different files from {source_bucket_name} to {target_bucket_name}: "
        + _first_5_last_5(list(different))  # noqa:W503
    )
    if not dry_run:
        copy_objects(
            list(different),
            s3_source_client,
            source_bucket_name,
            s3_target_client,
            target_bucket_name,
            rename=rename,
        )


def fast_copy(
    s3_public_client,
    keys,
    source_bucket_name,
    target_bucket_name,
    source_zarr_path="hazard/hazard.zarr",
    target_zarr_path="hazard/hazard.zarr",
):
    """Efficiently copy objects from one S3 bucket to another.

    Args:
        s3_public_client : boto3.Client
            S3 client for the public bucket.
        keys : list
            List of object keys to copy.
        source_bucket_name : str
            Source S3 bucket name.
        target_bucket_name : str
            Target S3 bucket name.
        source_zarr_path : str, optional
            The source Zarr path. Defaults to "hazard/hazard.zarr".
        target_zarr_path : str, optional
            The target Zarr path. Defaults to "hazard/hazard.zarr".

    """
    for i, key in enumerate(keys):
        copy_source = {"Bucket": source_bucket_name, "Key": key}
        target_key = key.replace(source_zarr_path, target_zarr_path)
        # target_key = key
        # print(f"{key} to {target_key} for bucket {bucket_name}")
        s3_public_client.copy_object(
            CopySource=copy_source, Bucket=target_bucket_name, Key=target_key
        )
        if i % 100 == 0:
            logger.info(f"Completed {i}/{len(keys)}")


def _first_5_last_5(items):
    return ", ".join(items[0:5]) + ", ..., " + ", ".join(items[-5:-1])


def get_input_onboarding_data(download_dir=None):
    """Download inputs for IRISIndicator and Jupiter hazards for the onboarding from the production bucket.

    They will be downloaded in the downloads folder.

    """
    s3prod = s3fs.S3FileSystem(
        key=os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        secret=os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
    )
    if download_dir is None:
        download_dir = str(os.path.join(pathlib.Path.home(), "Downloads"))
    jupiter_path = str(
        pathlib.PurePosixPath(
            "physrisk-hazard-indicators-dev01",
            "inputs",
            "all_hazards",
            "jupiter",
            "osc-main.zip",
        )
    )
    iris_path = str(
        pathlib.PurePosixPath("physrisk-hazard-indicators-dev01", "inputs", "wind")
    )
    s3prod.get(jupiter_path, download_dir)
    s3prod.get(iris_path, download_dir, recursive=True)

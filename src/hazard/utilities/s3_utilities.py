"""Module for managing the transfer and synchronization of Zarr arrays between S3 buckets.

This module contains functions for copying, synchronizing, and listing objects
in different S3 storage environments, including development, production, and public
buckets. It utilizes the `boto3` and `s3fs` libraries for interacting with Amazon S3
and handles parallelized copy operations.
"""

import concurrent.futures
from glob import iglob
import hashlib
import logging
import os
import pathlib
import sys
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple, Union

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


def _get_boto3_client_and_bucket(use_dev: bool):
    """Build a boto3 S3 client and resolve the target bucket name + zarr group prefix.

    Uses OSC_S3_ACCESS_KEY/OSC_S3_SECRET_KEY/OSC_S3_BUCKET/OSC_S3_ENDPOINT, with a
    "_DEV" suffix on all four when use_dev is True. OSC_S3_ENDPOINT(_DEV) is optional
    and only needed for a non-AWS / custom S3-compatible endpoint.

    The returned zarr group prefix is the location of the hazard.zarr group *within*
    the bucket, i.e. "<bucket>/<zarr_group_prefix>/...". This defaults to
    "hazard-indicators/hazard.zarr" for both dev and prod (matching physrisk-api's
    convention), and can be overridden per-bucket via OSC_S3_HAZARD_PATH /
    OSC_S3_HAZARD_PATH_DEV -- e.g. if a contributor's own dev bucket happens to use
    the older "hazard/hazard.zarr" layout instead.
    """
    suffix = "_DEV" if use_dev else ""

    client_kwargs: dict = {
        "aws_access_key_id": os.environ.get(f"OSC_S3_ACCESS_KEY{suffix}", None),
        "aws_secret_access_key": os.environ.get(f"OSC_S3_SECRET_KEY{suffix}", None),
        "config": botocore.client.Config(max_pool_connections=32),
    }
    endpoint_url = os.environ.get(f"OSC_S3_ENDPOINT{suffix}", None)
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    s3_target_client = boto3.client("s3", **client_kwargs)
    target_bucket_name = os.environ[f"OSC_S3_BUCKET{suffix}"]
    default_zarr_group_prefix = "hazard-indicators/hazard.zarr"
    zarr_group_prefix = os.environ.get(
        f"OSC_S3_HAZARD_PATH{suffix}", default_zarr_group_prefix
    )
    return s3_target_client, target_bucket_name, zarr_group_prefix


def _local_zarr_relative_file_paths(
    root_path: pathlib.Path, array_path: str
) -> List[str]:
    """List files (including dotfiles) under array_path within a local Zarr group.

    Returned paths are relative to root_path, using posix separators.
    """
    relative_file_paths = [
        pathlib.Path(f).as_posix()
        for f in iglob(
            str(pathlib.PurePosixPath(array_path) / "**"),
            root_dir=str(root_path),
            recursive=True,
        )
        if (root_path / f).is_file()
    ]
    relative_file_paths += [
        pathlib.Path(f).as_posix()
        for f in iglob(
            str(pathlib.PurePosixPath(array_path) / "**/.*"),
            root_dir=str(root_path),
            recursive=True,
        )
        if (root_path / f).is_file()
    ]
    return relative_file_paths


def _hazard_zarr_key(path: str, zarr_group_prefix: str) -> str:
    """Build the S3 key for a file at a path relative to a local hazard.zarr group.

    zarr_group_prefix is the bucket-side location of the zarr group (e.g.
    "hazard/hazard.zarr" for dev, "hazard-indicators/hazard.zarr" for prod -- see
    _get_boto3_client_and_bucket), which does not necessarily match the local
    directory name path is relative to.
    """
    return str(pathlib.PurePosixPath(zarr_group_prefix, path))


def _local_etag(path: pathlib.Path) -> str:
    """Compute the S3 ETag a local file would get if PUT to S3 unencrypted/with SSE-S3.

    S3's ETag is the hex MD5 digest of the object body, double-quoted, for any
    object created by a single (non-multipart) PUT -- which is the only kind of
    upload this module makes -- as long as the bucket does not use SSE-KMS or
    SSE-C encryption (those produce an ETag that is not an MD5 digest). If this
    bucket uses either, every file will be reported as "different" even when its
    content is unchanged.
    """
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    return f'"{digest}"'


def _copy_local_to_bucket(
    zarr_dir: str,
    array_path: str,
    s3_target_client,
    target_bucket_name: str,
    zarr_group_prefix: str,
    dry_run: bool = False,
    relative_file_paths: Optional[Sequence[str]] = None,
):
    """Copy a local Zarr array to the given bucket using the given S3 client.

    Shared implementation for copy_local_to_dev, copy_local_to_prod, and (for the
    missing/different subset of files) _sync_local_to_bucket.

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_path (str): The path of the array or data set within the group (i.e. all files that are children are to be copied).
        s3_target_client: boto3 S3 client to copy to.
        target_bucket_name (str): Name of the target bucket.
        zarr_group_prefix (str): Bucket-side location of the zarr group (see
        _get_boto3_client_and_bucket) -- not necessarily the same as the local
        "hazard/hazard.zarr" directory name zarr_dir points to.
        dry_run (bool, optional): If True, log the action that would
        be taken without actually executing. Defaults to False.
        relative_file_paths: If given, copy only these paths (relative to zarr_dir)
        instead of discovering every file under array_path.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename="batch.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(f"Source path {zarr_dir}; target bucket {target_bucket_name}")

    root_path = pathlib.Path(zarr_dir)
    if relative_file_paths is None:
        relative_file_paths = _local_zarr_relative_file_paths(root_path, array_path)

    logger.info(f"Copying {len(relative_file_paths)} files in array {array_path}")

    def copy_file(path: str):
        # path is the relative path with respect to hazard.zarr
        with open(root_path / path, "rb") as f:
            data = f.read()
            s3_target_client.put_object(
                Body=data,
                Bucket=target_bucket_name,
                Key=_hazard_zarr_key(path, zarr_group_prefix),
            )

    if not dry_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(copy_file, path) for path in relative_file_paths]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Completed {completed}/{len(relative_file_paths)}")


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
    s3_target_client, target_bucket_name, zarr_group_prefix, _ = (
        _get_boto3_client_and_bucket(use_dev=True)
    )
    _copy_local_to_bucket(
        zarr_dir,
        array_path,
        s3_target_client,
        target_bucket_name,
        zarr_group_prefix,
        dry_run,
    )


def copy_local_to_prod(zarr_dir: str, array_path: str, dry_run=False):
    """Copy zarr array from a local directory to the production (public) bucket.

    Note the bucket-side zarr group is at "hazard-indicators/hazard.zarr", not
    "hazard/hazard.zarr" like the local directory convention -- see
    _get_boto3_client_and_bucket.

    Requires environment variables:
    OSC_S3_BUCKET=os-climate-physical-risk
    OSC_S3_ACCESS_KEY=...
    OSC_S3_SECRET_KEY=...

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_path (str): The path of the array or data set within the group (i.e. all files that are children are to be copied).
        dry_run (bool, optional): If True, log the action that would
        be taken without actually executing. Defaults to False.
    """
    s3_target_client, target_bucket_name, zarr_group_prefix, _ = (
        _get_boto3_client_and_bucket(use_dev=False)
    )
    _copy_local_to_bucket(
        zarr_dir,
        array_path,
        s3_target_client,
        target_bucket_name,
        zarr_group_prefix,
        dry_run,
    )


def _sync_local_to_bucket(
    zarr_dir: str,
    array_paths: Union[str, Iterable[str]],
    s3_target_client,
    target_bucket_name: str,
    zarr_group_prefix: str,
    dry_run: bool = True,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Synchronize local Zarr arrays to the given bucket based on ETag differences.

    Like sync_buckets, but the source is a local directory rather than another S3
    bucket: only files that are missing from the bucket, or whose ETag differs from
    the local file's computed ETag (see _local_etag), are uploaded. The total size of
    the files to copy, across all array paths, is logged.

    Shared implementation for sync_local_to_dev and sync_local_to_prod.

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_paths (Union[str, Iterable[str]]): The path(s) of the arrays or data sets
        within the group.
        s3_target_client: boto3 S3 client to sync to.
        target_bucket_name (str): Name of the target bucket.
        zarr_group_prefix (str): Bucket-side location of the zarr group (see
        _get_boto3_client_and_bucket) -- not necessarily the same as the local
        "hazard/hazard.zarr" directory name zarr_dir points to.
        dry_run (bool, optional): If True, log actions without executing them. Defaults to True.

    Returns:
        Tuple[Set[str], Set[str], Set[str]]: (checked, missing, different), each a
        set of paths relative to zarr_dir. checked is every local file considered;
        missing is present locally but not in the bucket; different is present in
        both but with differing content.
    """
    if isinstance(array_paths, str):
        array_paths = [array_paths]
    root_path = pathlib.Path(zarr_dir)

    def key(path: str) -> str:
        return _hazard_zarr_key(path, zarr_group_prefix)

    all_checked: Set[str] = set()
    all_missing: Set[str] = set()
    all_different: Set[str] = set()
    for array_path in array_paths:
        relative_file_paths = _local_zarr_relative_file_paths(root_path, array_path)
        prefix = key(array_path)
        target_etags = list_object_etags(s3_target_client, target_bucket_name, prefix)

        checked = set(relative_file_paths)
        missing = set(
            path for path in relative_file_paths if key(path) not in target_etags
        )
        different = set(
            path
            for path in relative_file_paths
            if path not in missing
            and target_etags[key(path)] != _local_etag(root_path / path)
        )

        logger.info(
            f"Checked {len(checked)} local files from {zarr_dir}/{array_path} against bucket {target_bucket_name}"
        )
        logger.info(
            f"Copying {len(missing)} missing files to {target_bucket_name}: "
            + _first_5_last_5(list(missing))
        )
        logger.info(
            f"Copying {len(different)} different files to {target_bucket_name}: "
            + _first_5_last_5(list(different))
        )
        logger.info(
            f"Size of files to copy: {format_size(local_files_size(zarr_dir, missing | different))}"
        )

        if not dry_run:
            _copy_local_to_bucket(
                zarr_dir,
                array_path,
                s3_target_client,
                target_bucket_name,
                zarr_group_prefix,
                dry_run=False,
                relative_file_paths=list(missing | different),
            )

        all_checked |= checked
        all_missing |= missing
        all_different |= different

    logger.info(
        f"Total size of files to copy: {format_size(local_files_size(zarr_dir, all_missing | all_different))}"
    )
    return all_checked, all_missing, all_different


def sync_local_to_dev(
    zarr_dir: str, array_paths: Union[str, Iterable[str]], dry_run: bool = True
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Synchronize local Zarr arrays to the development bucket based on ETag
    differences, only uploading files that are missing or different.
    Requires environment variables:
    OSC_S3_BUCKET_DEV=physrisk-hazard-indicators-dev01
    OSC_S3_ACCESS_KEY_DEV=...
    OSC_S3_SECRET_KEY_DEV=...

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_paths (Union[str, Iterable[str]]): The path(s) of the arrays or data sets
        within the group.
        dry_run (bool, optional): If True, log actions without executing them. Defaults to True.

    Returns:
        Tuple[Set[str], Set[str], Set[str]]: (checked, missing, different) relative
        file paths, as described in _sync_local_to_bucket.
    """
    s3_target_client, target_bucket_name, zarr_group_prefix = (
        _get_boto3_client_and_bucket(use_dev=True)
    )
    return _sync_local_to_bucket(
        zarr_dir,
        array_paths,
        s3_target_client,
        target_bucket_name,
        zarr_group_prefix,
        dry_run,
    )


def sync_local_to_prod(
    zarr_dir: str, array_paths: Union[str, Iterable[str]], dry_run: bool = True
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Synchronize local Zarr arrays to the production (public) bucket based on
    ETag differences, only uploading files that are missing or different.

    Note the bucket-side zarr group is at "hazard-indicators/hazard.zarr", not
    "hazard/hazard.zarr" like the local directory convention -- see
    _get_boto3_client_and_bucket.

    Requires environment variables:
    OSC_S3_BUCKET=os-climate-physical-risk
    OSC_S3_ACCESS_KEY=...
    OSC_S3_SECRET_KEY=...

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_paths (Union[str, Iterable[str]]): The path(s) of the arrays or data sets
        within the group.
        dry_run (bool, optional): If True, log actions without executing them. Defaults to True.

    Returns:
        Tuple[Set[str], Set[str], Set[str]]: (checked, missing, different) relative
        file paths, as described in _sync_local_to_bucket.
    """
    s3_target_client, target_bucket_name, zarr_group_prefix = (
        _get_boto3_client_and_bucket(use_dev=False)
    )
    return _sync_local_to_bucket(
        zarr_dir,
        array_paths,
        s3_target_client,
        target_bucket_name,
        zarr_group_prefix,
        dry_run,
    )


def _local_file_matches(path: pathlib.Path, etag: str, size: int) -> bool:
    """Check whether a local file has the same content as an S3 object with the given ETag and size.

    The size is checked first so that the (more expensive) MD5 is only computed when
    the sizes agree. An ETag containing "-" comes from a multipart upload and is not
    an MD5 of the body, so for those objects matching size is taken as matching content.
    """
    if path.stat().st_size != size:
        return False
    if "-" in etag:
        return True
    return _local_etag(path) == etag


def _sync_bucket_to_local(
    s3_source_client,
    source_bucket_name: str,
    zarr_group_prefix: str,
    zarr_dir: str,
    array_path: str = "",
    dry_run: bool = True,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Synchronize a bucket's Zarr group (or an array within it) to a local directory.

    The reverse of _sync_local_to_bucket: only objects that are missing locally, or whose
    local copy differs (see _local_file_matches), are downloaded. The bucket is only ever
    read from. Local files with no counterpart in the bucket are reported but not removed.

    Args:
        s3_source_client: boto3 S3 client to sync from.
        source_bucket_name (str): Name of the source bucket.
        zarr_group_prefix (str): Bucket-side location of the zarr group (see
        _get_boto3_client_and_bucket).
        zarr_dir (str): Local directory of the Zarr group, i.e. /<path>/hazard.zarr.
        array_path (str, optional): Path of the array or data set within the group to
        sync. Defaults to "", i.e. the whole group.
        dry_run (bool, optional): If True, log actions without executing them. Defaults to True.

    Returns:
        Tuple[Set[str], Set[str], Set[str]]: (checked, missing, different), each a
        set of paths relative to zarr_dir. checked is every object in the bucket under
        array_path; missing is present in the bucket but not locally; different is
        present in both but with differing content.
    """
    root_path = pathlib.Path(zarr_dir)
    # trailing "/" so that e.g. "hazard.zarr" does not also match "hazard.zarr_old"
    prefix = _hazard_zarr_key(array_path, zarr_group_prefix).rstrip("/") + "/"

    logger.info(
        f"Syncing Bucket {source_bucket_name}/{prefix} to local path {zarr_dir}/{array_path} (dry_run={dry_run})"
    )

    objects: dict = {}  # relative path -> (key, etag, size)
    for obj in list_objects_parallel(s3_source_client, source_bucket_name, prefix):
        if obj["Key"].endswith("/"):
            continue  # "directory" marker
        path = pathlib.PurePosixPath(obj["Key"]).relative_to(zarr_group_prefix)
        objects[path.as_posix()] = (obj["Key"], obj["ETag"], obj["Size"])

    checked = set(objects)
    missing = set(path for path in checked if not (root_path / path).is_file())
    different = set(
        path
        for path in checked - missing
        if not _local_file_matches(root_path / path, *objects[path][1:])
    )
    local_root = root_path / array_path
    local_only = (
        set(
            (pathlib.Path(dir_path) / f).relative_to(root_path).as_posix()
            for dir_path, _, files in os.walk(local_root)
            for f in files
        )
        - checked
        if local_root.exists()
        else set()
    )

    def size(paths: Iterable[str]) -> int:
        return sum(objects[path][2] for path in paths)

    logger.info(
        f"Bucket {source_bucket_name}/{prefix} contains {len(checked)} files, total size {format_size(size(checked))}"
    )
    logger.info(
        f"Downloading {len(missing)} missing files ({format_size(size(missing))}) to {zarr_dir}: "
        + _first_5_last_5(sorted(missing))
    )
    logger.info(
        f"Downloading {len(different)} different files ({format_size(size(different))}) to {zarr_dir}: "
        + _first_5_last_5(sorted(different))
    )
    logger.info(
        f"Total to download: {len(missing | different)} files, {format_size(size(missing | different))}"
    )
    if local_only:
        logger.info(
            f"{len(local_only)} local files are not in the bucket (not removed): "
            + _first_5_last_5(sorted(local_only))
        )

    if not dry_run:
        to_download = sorted(missing | different)

        def download_file(path: str):
            local_path = root_path / path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # download_file writes to a temporary file and renames it on completion,
            # so an interrupted run does not leave partial files behind
            s3_source_client.download_file(
                source_bucket_name, objects[path][0], str(local_path)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(download_file, path) for path in to_download]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                if completed % 1000 == 0:
                    logger.info(f"Completed {completed}/{len(to_download)}")
        logger.info(f"Completed {completed}/{len(to_download)}")

    return checked, missing, different


def sync_prod_to_local(
    zarr_dir: str, array_path: str = "", dry_run: bool = True
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Synchronize the production bucket's Zarr group to a local directory, e.g. as a back-up.

    Only files that are missing locally or differ from the bucket are downloaded, so the
    first run copies everything and subsequent runs bring the copy up to date. Local files
    no longer in the bucket are reported but kept.

    The production bucket is public, so no credentials are needed: an anonymous (unsigned)
    client is used, which also guarantees that the bucket can only be read from. The bucket
    and zarr group prefix default to "os-climate-physical-risk" and
    "hazard-indicators/hazard.zarr" and can be overridden by OSC_S3_BUCKET and
    OSC_S3_HAZARD_PATH.

    Args:
        zarr_dir (str): Local directory of the Zarr group, i.e. /<path>/hazard.zarr.
        array_path (str, optional): Path of the array or data set within the group to
        sync. Defaults to "", i.e. the whole group.
        dry_run (bool, optional): If True, log actions without executing them. Defaults to True.

    Returns:
        Tuple[Set[str], Set[str], Set[str]]: (checked, missing, different) relative
        file paths, as described in _sync_bucket_to_local.
    """
    s3_source_client = boto3.client(
        "s3",
        config=botocore.client.Config(
            signature_version=botocore.UNSIGNED, max_pool_connections=32
        ),
    )
    source_bucket_name = os.environ.get("OSC_S3_BUCKET", "os-climate-physical-risk")
    zarr_group_prefix = os.environ.get(
        "OSC_S3_HAZARD_PATH", "hazard-indicators/hazard.zarr"
    )
    return _sync_bucket_to_local(
        s3_source_client,
        source_bucket_name,
        zarr_group_prefix,
        zarr_dir,
        array_path,
        dry_run,
    )


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


def list_objects_parallel(
    client, bucket_name: str, prefix: str, max_workers: int = 32
) -> List[dict]:
    """List all objects in an S3 bucket with a given prefix, fetching pages in parallel.

    list_objects_v2 returns at most 1000 keys per page and each page needs the previous
    page's continuation token, so a single listing of a large prefix is sequential and
    slow (around 6 minutes for the ~1M objects of the production hazard.zarr). Instead,
    the listing is split by sub-prefix: prefixes are listed with Delimiter="/", and each
    sub-prefix ("directory") is queued to be listed in the same way as soon as it is
    found. The result is the same as a single listing of prefix, but in no particular
    order. Note that parsing the responses is CPU-bound (around 0.1s per 1000 keys), which
    limits what can be gained from threads: around 3 minutes for the production bucket.
    For a prefix containing many small "directories" (e.g. many small Zarr arrays) this
    makes more requests than a single listing and can be slower, so it is only worth
    using for large listings, such as the whole bucket.

    Args:
        client : boto3.Client
            The S3 client used for listing objects.
        bucket_name : str
            The name of the S3 bucket.
        prefix : str
            The prefix to filter objects by.
        max_workers : int, optional
            Number of listings to run concurrently. Defaults to 32.

    Returns:
        List[dict]
            The listed objects, as returned in the "Contents" of list_objects_v2
            (i.e. with "Key", "Size", "ETag" etc.).

    """

    def list_prefix(prefix: str) -> Tuple[List[dict], List[str]]:
        objects: List[dict] = []
        sub_prefixes: List[str] = []
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket_name, Prefix=prefix, Delimiter="/"
        ):
            objects.extend(page.get("Contents", []))
            sub_prefixes.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
        return objects, sub_prefixes

    all_objects: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(list_prefix, prefix)}
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                objects, sub_prefixes = future.result()
                all_objects.extend(objects)
                pending |= {
                    executor.submit(list_prefix, sub_prefix)
                    for sub_prefix in sub_prefixes
                }
    return all_objects


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

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(copy_object, key) for key in keys]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0:
                logger.info(f"Completed {completed}/{len(keys)}")
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


def _first_5_last_5(items):
    return ", ".join(items[0:5]) + ", ..., " + ", ".join(items[-5:-1])


def local_files_size(zarr_dir: str, relative_file_paths: Iterable[str]) -> int:
    """Total size in bytes of the given files, as paths relative to zarr_dir."""
    root_path = pathlib.Path(zarr_dir)
    return sum((root_path / path).stat().st_size for path in relative_file_paths)


def format_size(num_bytes: float) -> str:
    """Format a size in bytes for display, e.g. 1536 -> '1.5 KiB'."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TiB"


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

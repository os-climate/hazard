import asyncio
import concurrent.futures
import logging
import os
import pathlib
import sys
from typing import Callable, Optional, Sequence

import boto3
import botocore.client

logger = logging.getLogger(__name__)


def copy_local_to_dev(zarr_dir: str, array_path: str, dry_run=False):
    """Copy zarr array from a local directory to the development bucket.
    Requires environment variables:
    OSC_S3_BUCKET_DEV=physrisk-hazard-indicators-dev01
    OSC_S3_ACCESS_KEY_DEV=...
    OSC_S3_SECRET_KEY_DEV=...

    Args:
        zarr_dir (str): Directory of the Zarr group, i.e. /<path>/hazard/hazard.zarr.
        array_path (str): The path of the array within the group.
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

    files = [f for f in pathlib.Path(zarr_dir, array_path).iterdir() if f.is_file()]
    logger.info(f"Copying {len(files)} files in array {array_path}")

    def copy_file(file: pathlib.Path):
        with open(file, "rb") as f:
            data = f.read()
            target_key = str(
                pathlib.PurePosixPath("hazard", "hazard.zarr", array_path, file.name)
            )
            s3_target_client.put_object(
                Body=data, Bucket=target_bucket_name, Key=target_key
            )

    async def copy_all():
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(executor, copy_file, file) for file in files]

        completed = []
        for coro in asyncio.as_completed(futures):
            completed.append(await coro)
            if len(completed) % 100 == 0:
                logger.info(f"Completed {len(completed)}/{len(files)}")

    if not dry_run:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(copy_all())


def copy_dev_to_prod(prefix: str, dry_run=False, sync=True):
    """Use this script to copy files with the prefix specified from
    dev S3 to prod S3.
    OSC_S3_BUCKET_DEV=physrisk-hazard-indicators-dev01
    OSC_S3_ACCESS_KEY_DEV=...
    OSC_S3_SECRET_KEY_DEV=...
    OSC_S3_BUCKET=physrisk-hazard-indicators
    OSC_S3_ACCESS_KEY=...
    OSC_S3_SECRET_KEY=...
    """
    s3_source_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
        config=botocore.client.Config(max_pool_connections=32),
    )
    s3_target_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY", None),
        config=botocore.client.Config(max_pool_connections=32),
    )

    source_bucket_name = os.environ["OSC_S3_BUCKET_DEV"]
    target_bucket_name = os.environ["OSC_S3_BUCKET"]
    if (
        source_bucket_name != "physrisk-hazard-indicators-dev01"
        or target_bucket_name != "physrisk-hazard-indicators"
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


def copy_prod_to_public(prefix: str, dry_run=False, sync=True):
    """Use this script to copy files with the prefix specified from
    prod S3 to public S3.
    OSC_S3_BUCKET=physrisk-hazard-indicators
    OSC_S3_ACCESS_KEY=...
    OSC_S3_SECRET_KEY=...
    OSC_S3_BUCKET_PUBLIC=os-climate-public-data
    OSC_S3_ACCESS_KEY_PUBLIC=...
    OSC_S3_SECRET_KEY_PUBLIC=...
    """
    s3_source_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("OSC_S3_ACCESS_KEY", None),
        aws_secret_access_key=os.environ.get("OSC_S3_SECRET_KEY", None),
    )
    s3_target_client = boto3.client(
        "s3",
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
    paginator = client.get_paginator(
        "list_objects_v2"
    )  # , PaginationConfig={"MaxItems": 10000})
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # get the list of keys with the given prefix
    keys = []
    size = 0
    for page in pages:
        for objs in page["Contents"]:
            if isinstance(objs, list):
                for obj in objs:
                    keys.append(obj["Key"])
                    size += obj["Size"]
            else:
                keys.append(objs["Key"])
                size += objs["Size"]
    return keys, size


def list_object_etags(client, bucket_name, prefix):
    paginator = client.get_paginator(
        "list_objects_v2"
    )  # , PaginationConfig={'MaxItems': 10000})
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # get the list of keys with the given prefix
    etags = {}
    for page in pages:
        for objs in page["Contents"]:
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
    """Form of copy that allows separate credentials for source and target buckets."""

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
    for i, key in enumerate(keys):
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        if i % 100 == 0:
            logger.info(f"Completed {i}/{len(keys)}")


def remove_from_prod(prefix: str, dry_run=True):
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
    prefix,
    dry_run=True,
):
    logger.info(
        f"Syncing target bucket {target_bucket_name} to source {source_bucket_name}."
    )

    source_etags = list_object_etags(s3_source_client, source_bucket_name, prefix)
    target_etags = list_object_etags(s3_target_client, target_bucket_name, prefix)
    # look for objects that are 1) missing in target and 2) different in target
    all_diffs = set(
        key for key, etag in source_etags.items() if target_etags.get(key, "") != etag
    )
    missing = set(key for key in source_etags if key not in target_etags)
    different = set(key for key in all_diffs if key not in missing)
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
        )


def fast_copy(
    s3_public_client,
    keys,
    source_bucket_name,
    target_bucket_name,
    source_zarr_path="hazard/hazard.zarr",
    target_zarr_path="hazard/hazard.zarr",
):
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

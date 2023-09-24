import logging
import os
from typing import Sequence

import boto3

logger = logging.getLogger(__name__)


def copy_dev_to_prod(prefix: str, dry_run = False):
    """ Use this script to copy files with the prefix specified from
    dev S3 to prod S3. 
    OSC_S3_BUCKET=physrisk-hazard-indicators
    OSC_S3_ACCESS_KEY=...
    OSC_S3_SECRET_KEY=...
    OSC_S3_BUCKET_DEV=physrisk-hazard-indicators-dev01
    OSC_S3_ACCESS_KEY_DEV=...
    OSC_S3_SECRET_KEY_DEV=...
    """
    s3_source_client = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_ACCESS_KEY_DEV"], 
                                    aws_secret_access_key=os.environ["OSC_S3_SECRET_KEY_DEV"])
    s3_target_client = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_ACCESS_KEY"], 
                                    aws_secret_access_key=os.environ["OSC_S3_SECRET_KEY"])

    source_bucket_name = os.environ["OSC_S3_BUCKET_DEV"]
    target_bucket_name = os.environ["OSC_S3_BUCKET"]
    if source_bucket_name != "physrisk-hazard-indicators-dev01" or target_bucket_name != "physrisk-hazard-indicators":
        # double check on environment variables
        raise ValueError("unexpected bucket")
    keys, size = list_objects(s3_source_client, source_bucket_name, prefix)
    logger.info(f"Prefix {prefix} {len(keys)} objects with total size {size / 1e9}GB")
    logger.info(f"Copying from bucket {source_bucket_name} to bucket {target_bucket_name}")
    if not dry_run:
        copy_objects(keys, s3_source_client, source_bucket_name, s3_target_client, target_bucket_name)


def list_objects(client, bucket_name, prefix):
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # get the list of keys with the given prefix
    keys = []
    size = 0
    for page in pages:
        for objs in page['Contents']:
            if isinstance(objs, list):
                for obj in objs:
                    keys.append(obj['Key'])
                    size += obj['Size']
            else:
                keys.append(objs['Key'])
                size += objs['Size']
    return keys, size


def copy_objects(keys: Sequence[str], s3_source_client, source_bucket_name: str,
              s3_target_client, target_bucket_name: str):
    """Form of copy that allows separate credentials for source and target buckets."""
    
    for i, key in enumerate(keys):
        obj = s3_source_client.get_object(
            Bucket=source_bucket_name,
            Key=key
        )
        data = obj['Body'].read()
        target_key = key
        #target_key = key.replace('hazard_test/hazard.zarr', 'hazard/hazard.zarr')
        s3_target_client.put_object(
            Body=data,
            Bucket=target_bucket_name,
            Key=target_key
        )
        if i % 100 == 0:
            logger.info(f"Completed {i}/{len(keys)}")


def fast_copy(s3_public_client, keys, source_bucket_name, target_bucket_name, 
              source_zarr_path="hazard/hazard.zarr", target_zarr_path="hazard/hazard.zarr"):
    for i, key in enumerate(keys):
        copy_source = {
            'Bucket': source_bucket_name,
            'Key': key
            }
        target_key = key.replace(source_zarr_path, target_zarr_path)
        #target_key = key
        #print(f"{key} to {target_key} for bucket {bucket_name}")
        s3_public_client.copy_object(CopySource=copy_source, Bucket=target_bucket_name, Key=target_key)
        if i % 100 == 0:
            logger.info(f"Completed {i}/{len(keys)}")
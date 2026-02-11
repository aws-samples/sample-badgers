"""
S3-based configuration and prompt loading with caching.

Provides cached loading of manifests and prompts from S3 for Lambda functions.
"""

import json
import logging
from functools import lru_cache
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class S3ConfigError(Exception):
    """Raised when S3 config loading fails."""


@lru_cache(maxsize=128)
def load_manifest_from_s3(
    bucket: str, analyzer_name: str, custom: bool = False
) -> Dict[str, Any]:
    """
    Load analyzer manifest from S3 with caching.

    Cache persists across Lambda warm invocations for performance.

    Args:
        bucket: S3 bucket name
        analyzer_name: Name of the analyzer (e.g., 'full_text_analyzer')
        custom: Whether this is a custom analyzer (uses custom_analyzers/ prefix)

    Returns:
        Manifest dictionary

    Raises:
        S3ConfigError: If manifest cannot be loaded
    """
    try:
        s3 = boto3.client("s3")
        if custom:
            key = f"custom-analyzers/manifests/{analyzer_name}.json"
        else:
            key = f"manifests/{analyzer_name}.json"

        logger.info("Loading manifest from s3://%s/%s", bucket, key)
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        manifest = json.loads(content)
        logger.info("Loaded manifest for %s (%d bytes)", analyzer_name, len(content))

        return manifest

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            raise S3ConfigError(f"Manifest not found: s3://{bucket}/{key}") from e
        raise S3ConfigError(f"Failed to load manifest from S3: {error_code}") from e
    except json.JSONDecodeError as e:
        raise S3ConfigError(f"Invalid JSON in manifest s3://{bucket}/{key}: {e}") from e
    except Exception as e:
        raise S3ConfigError(f"Unexpected error loading manifest: {e}") from e


@lru_cache(maxsize=256)
def load_prompt_from_s3(
    bucket: str, analyzer_name: str, prompt_file: str, custom: bool = False
) -> str:
    """
    Load prompt file from S3 with caching.

    Cache persists across Lambda warm invocations for performance.

    Args:
        bucket: S3 bucket name
        analyzer_name: Name of the analyzer
        prompt_file: Relative path to prompt file
        custom: Whether this is a custom analyzer (uses custom-analyzers/ prefix)

    Returns:
        Prompt content as string

    Raises:
        S3ConfigError: If prompt cannot be loaded
    """
    try:
        s3 = boto3.client("s3")
        if custom:
            key = f"custom-analyzers/prompts/{analyzer_name}/{prompt_file}"
        else:
            key = f"prompts/{analyzer_name}/{prompt_file}"

        logger.debug("Loading prompt from s3://%s/%s", bucket, key)
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        logger.debug("Loaded prompt %s (%d bytes)", prompt_file, len(content))

        return content

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            raise S3ConfigError(f"Prompt not found: s3://{bucket}/{key}") from e
        raise S3ConfigError(f"Failed to load prompt from S3: {error_code}") from e
    except UnicodeDecodeError as e:
        raise S3ConfigError(f"Failed to decode prompt s3://{bucket}/{key}: {e}") from e
    except Exception as e:
        raise S3ConfigError(f"Unexpected error loading prompt: {e}") from e


@lru_cache(maxsize=32)
def load_wrapper_from_s3(
    bucket: str, wrapper_name: str = "prompt_system_wrapper.xml"
) -> str:
    """
    Load system prompt wrapper from S3 with caching.

    Args:
        bucket: S3 bucket name
        wrapper_name: Name of wrapper file

    Returns:
        Wrapper content as string

    Raises:
        S3ConfigError: If wrapper cannot be loaded
    """
    try:
        s3 = boto3.client("s3")
        key = f"wrappers/{wrapper_name}"

        logger.debug("Loading wrapper from s3://%s/%s", bucket, key)
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        logger.info("Loaded wrapper %s (%d bytes)", wrapper_name, len(content))

        return content

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            raise S3ConfigError(f"Wrapper not found: s3://{bucket}/{key}") from e
        raise S3ConfigError(f"Failed to load wrapper from S3: {error_code}") from e
    except UnicodeDecodeError as e:
        raise S3ConfigError(f"Failed to decode wrapper s3://{bucket}/{key}: {e}") from e
    except Exception as e:
        raise S3ConfigError(f"Unexpected error loading wrapper: {e}") from e


def clear_s3_cache():
    """Clear all S3 config caches. Useful for testing or forced refresh."""
    load_manifest_from_s3.cache_clear()
    load_prompt_from_s3.cache_clear()
    load_wrapper_from_s3.cache_clear()
    logger.info("Cleared S3 config caches")

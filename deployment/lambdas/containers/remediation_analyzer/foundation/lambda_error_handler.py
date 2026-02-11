"""Standardized error handling for Lambda functions called via AgentCore Gateway."""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LambdaError(Exception):
    """Base class for Lambda errors with structured responses."""

    def __init__(
        self,
        message: str,
        error_type: str,
        status_code: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}


class ValidationError(LambdaError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="ValidationError",
            status_code=400,
            details=details,
        )


class ResourceNotFoundError(LambdaError):
    """Raised when a required resource is not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="ResourceNotFound",
            status_code=404,
            details=details,
        )


class ModelUnavailableError(LambdaError):
    """Raised when Bedrock model is unavailable."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="ModelUnavailable",
            status_code=503,
            details=details,
        )


class TimeoutError(LambdaError):
    """Raised when processing exceeds time limits."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="TimeoutError",
            status_code=504,
            details=details,
        )


class InternalError(LambdaError):
    """Raised for unexpected internal errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="InternalError",
            status_code=500,
            details=details,
        )


def create_error_response(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error response for AgentCore Gateway.

    Args:
        error: The exception that occurred

    Returns:
        Dict with statusCode and body for Lambda response
    """
    if isinstance(error, LambdaError):
        # Structured error with known type
        status_code = error.status_code
        error_body = {
            "error_type": error.error_type,
            "message": error.message,
            "details": error.details,
        }
        logger.error(
            "Lambda error: %s - %s",
            error.error_type,
            error.message,
            extra={"details": error.details},
        )
    else:
        # Unexpected error - log full trace but return generic message
        logger.error("Unexpected error: %s", str(error), exc_info=True)
        status_code = 500
        error_body = {
            "error_type": "InternalError",
            "message": "An unexpected error occurred during processing.",
            "details": {"error_class": error.__class__.__name__},
        }

    return {"statusCode": status_code, "body": json.dumps(error_body)}


def handle_bedrock_error(error: Exception, model_id: str) -> LambdaError:
    """
    Convert Bedrock-specific errors to structured Lambda errors.

    Args:
        error: The Bedrock exception
        model_id: The model ID that was being invoked

    Returns:
        Appropriate LambdaError subclass
    """
    error_str = str(error)

    # Throttling errors
    if "ThrottlingException" in error_str or "TooManyRequestsException" in error_str:
        return ModelUnavailableError(
            message="Bedrock model is currently throttling requests. Please retry.",
            details={"model_id": model_id, "retry_after_seconds": 5},
        )

    # Model not available
    if (
        "ModelNotReadyException" in error_str
        or "ServiceUnavailableException" in error_str
    ):
        return ModelUnavailableError(
            message="Bedrock model temporarily unavailable. Please retry.",
            details={"model_id": model_id, "retry_after_seconds": 10},
        )

    # Validation errors
    if "ValidationException" in error_str:
        return ValidationError(
            message="Invalid request to Bedrock model.",
            details={"model_id": model_id, "error": error_str},
        )

    # Access denied
    if "AccessDeniedException" in error_str:
        return InternalError(
            message="Insufficient permissions to invoke Bedrock model.",
            details={"model_id": model_id},
        )

    # Generic Bedrock error
    return InternalError(
        message="Failed to invoke Bedrock model.",
        details={"model_id": model_id, "error": error_str},
    )


def handle_s3_error(error: Exception, bucket: str, key: str) -> LambdaError:
    """
    Convert S3-specific errors to structured Lambda errors.

    Args:
        error: The S3 exception
        bucket: The S3 bucket name
        key: The S3 object key

    Returns:
        Appropriate LambdaError subclass
    """
    error_str = str(error)

    # Object not found
    if "NoSuchKey" in error_str or "404" in error_str:
        return ResourceNotFoundError(
            message="Image not found in S3 bucket.",
            details={"bucket": bucket, "key": key},
        )

    # Access denied
    if "AccessDenied" in error_str or "403" in error_str:
        return InternalError(
            message="Insufficient permissions to access S3 object.",
            details={"bucket": bucket, "key": key},
        )

    # Generic S3 error
    return InternalError(
        message="Failed to retrieve image from S3.",
        details={"bucket": bucket, "key": key, "error": error_str},
    )

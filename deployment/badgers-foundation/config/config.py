"""Configuration management for the private MCP server."""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""

    log_level: str = "INFO"
    log_streaming: bool = True
    log_tools: bool = True


def configure_logging(config: LoggingConfig) -> None:
    """Configure logging based on environment variables.

    Args:
        config: LoggingConfig instance with logging settings

    Sets up logging with the specified level and conditionally disables
    streaming and tool logs based on configuration.
    """
    import sys

    # Set base logging level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    # Logs go to stderr to keep stdout clean
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Conditionally disable streaming logs if LOG_STREAMING=false
    if not config.log_streaming:
        logging.getLogger("strands").setLevel(logging.WARNING)
        logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
        logging.getLogger("agent").setLevel(logging.WARNING)

    # Conditionally disable tool logs if LOG_TOOLS=false
    if not config.log_tools:
        logging.getLogger("badgers_foundation.tools").setLevel(logging.WARNING)
        logging.getLogger("task_executor").setLevel(logging.WARNING)
        logging.getLogger("tools").setLevel(logging.WARNING)


class CredentialsNotConfiguredError(Exception):
    """Raised when AWS credentials are not properly configured."""


# Absolute path to analyzer configuration - single source of truth
# Resolved relative to this config.py file location
_CONFIG_DIR = Path(__file__).parent
ANALYZER_CONFIG_PATH = str(_CONFIG_DIR / "analyzer_config.json")


@dataclass
class ServerConfig:
    """Configuration class for the MCP server."""

    server_name: str = "Private MCP Server"
    log_level: str = "INFO"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    timeout: int = 300  # 5 minutes default timeout in seconds
    aws_profile: Optional[str] = None  # AWS profile for Bedrock/AWS services
    logging: Optional[LoggingConfig] = None

    def __post_init__(self):
        if self.logging is None:
            self.logging = LoggingConfig()

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> "ServerConfig":
        """Load configuration from TOML file.

        Environment variables override TOML values:
        - MCP_SERVER_NAME overrides [server].name
        - MCP_HOST overrides [server].host
        - MCP_PORT overrides [server].port
        - MCP_TIMEOUT overrides [server].timeout
        - MCP_DEBUG overrides [server].debug
        - AWS_PROFILE overrides [aws].profile
        - LOG_LEVEL overrides [logging].level
        - LOG_STREAMING overrides [logging].streaming
        - LOG_TOOLS overrides [logging].tools

        Returns:
            ServerConfig: Configuration instance with values from TOML or defaults.
        """
        # Find config file - look in badgers-foundation directory
        config_file = Path(__file__).parent.parent / config_path

        if not config_file.exists():
            logger.warning("Config file not found: %s, using defaults", config_file)
            config = cls()
            config.validate()
            return config

        # Load TOML
        with open(config_file, "rb") as f:
            toml_config = tomllib.load(f)

        # Extract server values with environment variable overrides
        server_section = toml_config.get("server", {})
        server_name = os.getenv("MCP_SERVER_NAME") or server_section.get(
            "name", "Private MCP Server"
        )
        host = os.getenv("MCP_HOST") or server_section.get("host", "127.0.0.1")

        # Parse port with validation
        port_str = os.getenv("MCP_PORT") or str(server_section.get("port", 8000))
        try:
            port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid port number: '{port_str}'. Must be an integer."
            ) from exc

        # Parse timeout with validation
        timeout_str = os.getenv("MCP_TIMEOUT") or str(
            server_section.get("timeout", 300)
        )
        try:
            timeout = int(timeout_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid timeout: '{timeout_str}'. Must be an integer."
            ) from exc

        # Parse debug flag
        debug_env = os.getenv("MCP_DEBUG")
        if debug_env:
            debug = debug_env.lower().strip() in ("true", "1", "yes", "on")
        else:
            debug = server_section.get("debug", False)

        # AWS profile
        aws_profile = os.getenv("AWS_PROFILE") or toml_config.get("aws", {}).get(
            "profile", "rbpotter+nlp-rbpotter-nlp-role"
        )

        # Logging config
        logging_section = toml_config.get("logging", {})
        log_level = os.getenv("LOG_LEVEL") or logging_section.get("level", "INFO")
        log_streaming = (
            os.getenv(
                "LOG_STREAMING", str(logging_section.get("streaming", True))
            ).lower()
            == "true"
        )
        log_tools = (
            os.getenv("LOG_TOOLS", str(logging_section.get("tools", True))).lower()
            == "true"
        )

        logging_config = LoggingConfig(
            log_level=log_level, log_streaming=log_streaming, log_tools=log_tools
        )

        config = cls(
            server_name=server_name,
            log_level=log_level,
            debug=debug,
            host=host,
            port=port,
            timeout=timeout,
            aws_profile=aws_profile,
            logging=logging_config,
        )

        # Validate configuration before returning
        config.validate()
        return config

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from TOML file (legacy method for compatibility).

        This method now calls from_toml() for backward compatibility.
        """
        return cls.from_toml()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate server name
        if not self.server_name or not self.server_name.strip():
            raise ValueError("Server name cannot be empty or whitespace only")

        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"Invalid log level: '{self.log_level}'. "
                f"Must be one of: {', '.join(sorted(valid_log_levels))}"
            )

        # Validate host
        if not self.host or not self.host.strip():
            raise ValueError("Host cannot be empty or whitespace only")

        # Validate port
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ValueError(
                f"Invalid port: {self.port}. Must be an integer between 1 and 65535"
            )

        # Validate timeout
        if not isinstance(self.timeout, int) or self.timeout < 1:
            raise ValueError(
                f"Invalid timeout: {self.timeout}. Must be a positive integer"
            )

        # Normalize log level to uppercase
        self.log_level = self.log_level.upper()
        self.host = self.host.strip()

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"ServerConfig(server_name='{self.server_name}', "
            f"log_level='{self.log_level}', debug={self.debug}, "
            f"host='{self.host}', port={self.port}, timeout={self.timeout}, "
            f"aws_profile='{self.aws_profile}')"
        )


@dataclass
class KnowledgeBaseConfig:
    """Configuration for a single knowledge base."""

    id: str
    description: str
    region: str = "us-west-2"
    max_results: int = 10
    confidence_threshold: float = 0.7

    def validate(self) -> None:
        """Validate knowledge base configuration.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not self.id or not self.id.strip():
            raise ValueError("Knowledge base ID cannot be empty")

        if not self.description or not self.description.strip():
            raise ValueError("Knowledge base description cannot be empty")

        if self.max_results < 1 or self.max_results > 100:
            raise ValueError(
                f"max_results must be between 1 and 100, got {self.max_results}"
            )

        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )


@dataclass
class ToolsConfig:
    """Configuration class for tools including knowledge bases."""

    knowledge_bases: dict[str, KnowledgeBaseConfig]
    aws_knowledge_base: Optional[dict] = None

    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> "ToolsConfig":
        """Load tools configuration from JSON file.

        Args:
            config_path: Optional path to config file. If not provided, uses default.

        Returns:
            ToolsConfig: Configuration instance loaded from file.

        Raises:
            ValueError: If configuration file is invalid.
            FileNotFoundError: If configuration file doesn't exist.
        """
        import json
        from pathlib import Path as PathLib

        config_file_path: PathLib
        if config_path is None:
            # Default to config/tools_config.json
            config_dir = PathLib(__file__).parent
            config_file_path = config_dir / "tools_config.json"
        else:
            config_file_path = PathLib(config_path)

        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e

        # Parse knowledge bases
        kb_data = data.get("knowledge_bases", {})
        knowledge_bases = {}

        for kb_name, kb_config in kb_data.items():
            knowledge_bases[kb_name] = KnowledgeBaseConfig(
                id=kb_config["id"],
                description=kb_config.get("description", ""),
                region=kb_config.get("region", "us-west-2"),
                max_results=kb_config.get("max_results", 10),
                confidence_threshold=kb_config.get("confidence_threshold", 0.7),
            )
            # Validate each KB config
            knowledge_bases[kb_name].validate()

        # Get legacy aws_knowledge_base config if present
        aws_kb = data.get("aws_knowledge_base")

        config = cls(knowledge_bases=knowledge_bases, aws_knowledge_base=aws_kb)
        return config

    def get_kb_config(self, kb_name: str) -> Optional[KnowledgeBaseConfig]:
        """Get configuration for a specific knowledge base.

        Args:
            kb_name: Name of the knowledge base.

        Returns:
            KnowledgeBaseConfig if found, None otherwise.
        """
        return self.knowledge_bases.get(kb_name)

    def list_kb_names(self) -> list[str]:
        """Get list of all configured knowledge base names.

        Returns:
            List of knowledge base names.
        """
        return list(self.knowledge_bases.keys())

    def __str__(self) -> str:
        """String representation of the configuration."""
        kb_names = ", ".join(self.knowledge_bases.keys())
        return f"ToolsConfig(knowledge_bases=[{kb_names}])"


def ensure_aws_credentials_configured(config: ServerConfig) -> bool:
    """
    Ensure AWS credentials are properly configured for the MCP server.

    Checks if AWS_PROFILE environment variable is set. If not, sets it from config.
    Then validates that the profile exists and has valid credentials.

    Args:
        config: ServerConfig instance with aws_profile setting

    Returns:
        True if credentials are properly configured

    Raises:
        CredentialsNotConfiguredError: If credentials cannot be configured or validated
    """
    try:
        # Step 1: Ensure AWS_PROFILE is set
        if "AWS_PROFILE" not in os.environ:
            if not config.aws_profile:
                raise CredentialsNotConfiguredError(
                    "No AWS profile configured. Set AWS_PROFILE environment variable "
                    "or configure aws_profile in badgers-foundation/config/config.py"
                )

            os.environ["AWS_PROFILE"] = config.aws_profile
            logger.info("Set AWS_PROFILE from config: %s", config.aws_profile)
        else:
            logger.info(
                "Using AWS_PROFILE from environment: %s", os.environ["AWS_PROFILE"]
            )

        # Step 2: Validate credentials by attempting to create a boto3 session
        try:
            import boto3

            session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])

            # Try to get credentials to validate they exist
            credentials = session.get_credentials()
            if credentials is None:
                raise CredentialsNotConfiguredError(
                    f"AWS profile '{os.environ['AWS_PROFILE']}' exists but has no credentials. "
                    f"Run 'aws configure --profile {os.environ['AWS_PROFILE']}' to set up credentials."
                )

            # Validate credentials are not expired by checking if we can get them
            _ = credentials.get_frozen_credentials()

            logger.info("✅ AWS Profile Validated!")

            return True

        except Exception as e:
            error_msg = str(e)
            if (
                "could not be found" in error_msg.lower()
                or "profile" in error_msg.lower()
            ):
                raise CredentialsNotConfiguredError(
                    f"AWS profile '{os.environ['AWS_PROFILE']}' not found in ~/.aws/credentials or ~/.aws/config. "
                    f"Run 'aws configure --profile {os.environ['AWS_PROFILE']}' to create it."
                ) from e
            else:
                raise CredentialsNotConfiguredError(
                    f"Failed to validate AWS credentials: {error_msg}"
                ) from e

    except CredentialsNotConfiguredError:
        # Re-raise our custom exception
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise CredentialsNotConfiguredError(
            f"Unexpected error while configuring AWS credentials: {e}"
        ) from e

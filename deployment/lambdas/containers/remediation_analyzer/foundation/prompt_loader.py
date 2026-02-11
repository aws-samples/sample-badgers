"""
Prompt loading and caching functionality for analyzer system.

Handles loading, combining, and caching of prompt files for different analyzer types.
Supports both local file system and S3-based loading.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Find project root
# This file is in the foundation/ directory, so parent.parent gets us to the container root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class PromptLoadError(Exception):
    """Raised when prompt loading fails."""


class PromptLoader:
    """Handles the loading of prompt files from local filesystem or S3."""

    def __init__(
        self,
        config_source: str = "local",
        s3_bucket: Optional[str] = None,
        analyzer_name: Optional[str] = None,
        custom: bool = False,
    ):
        """
        Initialize the prompt loader.

        Args:
            config_source: 'local' or 's3'
            s3_bucket: S3 bucket name (required if config_source='s3')
            analyzer_name: Analyzer name (required if config_source='s3')
            custom: Whether this is a custom analyzer (affects S3 paths)
        """
        self.logger = logging.getLogger(__name__)
        self.config_source = config_source
        self.s3_bucket = s3_bucket
        self.analyzer_name = analyzer_name
        self.custom = custom

        if config_source == "s3":
            if not s3_bucket:
                raise PromptLoadError("s3_bucket required when config_source='s3'")
            if not analyzer_name:
                raise PromptLoadError("analyzer_name required when config_source='s3'")

            # Import S3 loader only when needed
            from foundation.s3_config_loader import (
                load_prompt_from_s3,
                load_wrapper_from_s3,
                S3ConfigError,
            )

            self._load_prompt_from_s3 = load_prompt_from_s3
            self._load_wrapper_from_s3 = load_wrapper_from_s3
            self._S3ConfigError = S3ConfigError

            self.logger.info(
                "Initialized S3 prompt loader: bucket=%s, analyzer=%s",
                s3_bucket,
                analyzer_name,
            )

    def load_system_prompt(
        self,
        analyzer_config: Dict[str, Any],
        placeholders: Optional[Dict[str, str]] = None,
        audit_mode: bool = False,
    ) -> str:
        """
        Load and compose the complete system prompt.

        Args:
            analyzer_config: Configuration dictionary for the analyzer
            placeholders: Optional dictionary of placeholder replacements (e.g., {"PIXEL_WIDTH": "1024"})
            audit_mode: Whether to include confidence assessment instructions

        Returns:
            Complete system prompt with all sections composed
        """
        try:
            # Step 1: Load all core system files
            core_prompt_files = self.load_core_system_files()

            # Step 2: Load and compose analyzer-specific prompts
            # Get the prompt base path from config (could be prompt_base_path or prompt_analyzer_prompt_base_path)
            prompt_base_key = None
            for key in ["prompt_base_path", "prompt_analyzer_prompt_base_path"]:
                if key in analyzer_config:
                    prompt_base_key = key
                    break

            if not prompt_base_key:
                raise PromptLoadError("No prompt base path found in analyzer config")

            analyzer_prompt_base_path = analyzer_config[prompt_base_key]
            composed_analyzer_prompt = self.load_prompt_files(
                analyzer_prompt_base_path,
                analyzer_config["prompt_files"],
            )

            # Step 3: Start with the wrapper template
            system_prompt = core_prompt_files["prompt_system_wrapper"]

            # Step 4: Inject all the content into the wrapper
            system_prompt = system_prompt.replace(
                "{core_rules}", core_prompt_files["core_rules_rules"]
            )
            system_prompt = system_prompt.replace(
                "{composed_prompt}", composed_analyzer_prompt
            )

            # Step 4.5: Inject audit mode content if enabled
            audit_content = ""
            if audit_mode:
                audit_key = "audit_confidence_assessment"
                if audit_key in core_prompt_files and core_prompt_files[audit_key]:
                    audit_content = core_prompt_files[audit_key]
                    self.logger.info(
                        "Audit mode enabled - injecting confidence assessment prompt"
                    )
            system_prompt = system_prompt.replace("{audit_mode}", audit_content)

            system_prompt = system_prompt.replace(
                "{error_handler_general}",
                core_prompt_files["error_handling_error_handler"],
            )
            system_prompt = system_prompt.replace(
                "{error_handler_not_found}",
                core_prompt_files["error_handling_not_found_handler"],
            )

            # Step 5: Replace custom placeholders if provided
            if placeholders:
                system_prompt = self.replace_placeholders(system_prompt, placeholders)

            self.logger.info(
                "Loaded system prompt for %s: %d characters",
                analyzer_config.get("name", "unknown"),
                len(system_prompt),
            )

            return system_prompt

        except Exception as e:
            raise PromptLoadError(f"Failed to load system prompt: {e}") from e

    def replace_placeholders(self, text: str, placeholders: Dict[str, str]) -> str:
        """
        Replace placeholders in text with actual values.

        Args:
            text: Text containing placeholders in format [[PLACEHOLDER_NAME]]
            placeholders: Dictionary mapping placeholder names to values

        Returns:
            Text with placeholders replaced
        """
        result = text
        for key, value in placeholders.items():
            placeholder = f"[[{key}]]"
            result = result.replace(placeholder, str(value))
            self.logger.debug("Replaced placeholder %s with %s", placeholder, value)
        return result

    def load_prompt_files(self, base_path: str, files: List[str]) -> str:
        """
        Load and combine multiple prompt files from local or S3.

        Args:
            base_path: Base directory path for prompt files (ignored for S3)
            files: List of prompt file paths relative to base_path

        Returns:
            Combined prompt content

        Raises:
            PromptLoadError: If prompt files cannot be loaded
        """
        try:
            if self.config_source == "s3":
                return self._load_prompt_files_from_s3(files)
            else:
                return self._load_prompt_files_from_local(base_path, files)

        except Exception as e:
            raise PromptLoadError(f"Failed to load prompt files: {e}") from e

    def _load_prompt_files_from_s3(self, files: List[str]) -> str:
        """Load and combine prompt files from S3."""
        combined_prompt = ""
        loaded_files = 0

        for file_path in files:
            try:
                content = self._load_prompt_from_s3(
                    self.s3_bucket, self.analyzer_name, file_path, self.custom
                )
                if content.strip():
                    combined_prompt += content + "\n\n"
                    loaded_files += 1
                    self.logger.debug("Loaded prompt file from S3: %s", file_path)
                else:
                    self.logger.warning("Empty prompt file: %s", file_path)
            except self._S3ConfigError as e:
                self.logger.warning("Could not load prompt file %s: %s", file_path, e)

        if loaded_files == 0:
            raise PromptLoadError("No valid prompt files were loaded from S3")

        self.logger.info(
            "Combined %d prompt files from S3 into %d characters",
            loaded_files,
            len(combined_prompt),
        )

        return combined_prompt.strip()

    def _load_prompt_files_from_local(self, base_path: str, files: List[str]) -> str:
        """Load and combine prompt files from local filesystem."""
        base_path_obj = Path(base_path)
        if not base_path_obj.exists():
            raise PromptLoadError(f"Prompt base path not found: {base_path}")

        combined_prompt = ""
        loaded_files = 0
        for file_path in files:
            full_path = base_path_obj / file_path

            if full_path.exists():
                content = self._read_file(full_path)
                if content.strip():
                    combined_prompt += content + "\n\n"
                    loaded_files += 1
                    self.logger.debug("Loaded prompt file: %s", file_path)
                else:
                    self.logger.warning("Empty prompt file: %s", file_path)
            else:
                self.logger.warning("Prompt file not found: %s", full_path)

        if loaded_files == 0:
            raise PromptLoadError("No valid prompt files were loaded")

        self.logger.info(
            "Combined %d prompt files into %d characters",
            loaded_files,
            len(combined_prompt),
        )

        return combined_prompt.strip()

    def _read_file(self, file_path: Path) -> str:
        """
        Read file content safely with error handling.

        Args:
            file_path: Path to the file to read

        Returns:
            File content as string

        Raises:
            PromptLoadError: If file cannot be read
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                self.logger.warning("File is empty: %s", file_path)

            return content

        except FileNotFoundError as exc:
            raise PromptLoadError(f"File not found: {file_path}") from exc
        except PermissionError as exc:
            raise PromptLoadError(
                f"Permission denied reading file: {file_path}"
            ) from exc
        except UnicodeDecodeError as e:
            raise PromptLoadError(f"Failed to decode file {file_path}: {e}") from e
        except Exception as e:
            raise PromptLoadError(f"Failed to read file {file_path}: {e}") from e

    def load_core_system_files(self) -> Dict[str, str]:
        """
        Load all core system prompt files into a dictionary.

        Returns:
            Dictionary with file contents keyed by relative paths
        """
        if self.config_source == "s3":
            return self._load_core_system_files_from_s3()
        else:
            return self._load_core_system_files_from_local()

    def _load_core_system_files_from_s3(self) -> Dict[str, str]:
        """Load core system files from S3."""
        result = {}

        # Load the wrapper file
        try:
            wrapper_content = self._load_wrapper_from_s3(self.s3_bucket)
            result["prompt_system_wrapper"] = wrapper_content
        except self._S3ConfigError as e:
            self.logger.warning("Could not load wrapper from S3: %s", e)
            result["prompt_system_wrapper"] = ""

        # For S3, we use simplified core rules stored in wrapper
        # These are placeholders that get replaced in load_system_prompt
        result["core_rules_rules"] = ""
        result["error_handling_error_handler"] = ""
        result["error_handling_not_found_handler"] = ""

        return result

    def _load_core_system_files_from_local(self) -> Dict[str, str]:
        """Load core system files from local filesystem."""
        # Try Lambda layer path first, then fall back to dev path
        core_system_path = PROJECT_ROOT / "prompts" / "core_system_prompts"
        if not core_system_path.exists():
            core_system_path = PROJECT_ROOT / "foundation" / "core_system_prompts"
        result = {}

        # Recursively find all .xml files
        for xml_file in core_system_path.rglob("*.xml"):
            # Use relative path from core_system_prompts as key
            relative_path = xml_file.relative_to(core_system_path)
            key = str(relative_path).replace("/", "_").replace(".xml", "")

            try:
                result[key] = self._read_file(xml_file)
            except PromptLoadError as e:
                self.logger.warning("Could not load %s: %s", key, e)
                result[key] = ""

        return result

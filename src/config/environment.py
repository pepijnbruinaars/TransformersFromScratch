"""Environment-aware configuration for cloud/local deployments.

Provides automatic detection of RunPod environment and resolves paths
to the network volume for persistent storage across spot instance restarts.
"""
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EnvironmentConfig:
    """Environment-aware configuration for cloud/local deployments.

    Supports RunPod by detecting /runpod-volume/ and adjusting all paths
    to use persistent network storage.

    Usage:
        env = EnvironmentConfig()  # Auto-detects RunPod
        checkpoint_dir = env.resolve_path("checkpoints/")
        # On RunPod: /runpod-volume/checkpoints/
        # Locally: ./checkpoints/
    """

    RUNPOD_VOLUME = "/runpod-volume"

    def __init__(self, base_path: Optional[str] = None):
        """Initialize environment configuration.

        Args:
            base_path: Override base path. If None, auto-detects RunPod.
        """
        if base_path is not None:
            self._base_path = Path(base_path)
            logger.info(f"Using configured base path: {self._base_path}")
        elif self._is_runpod():
            self._base_path = Path(self.RUNPOD_VOLUME)
            logger.info(f"RunPod environment detected, using base path: {self._base_path}")
        else:
            self._base_path = Path(".")
            logger.info("Local environment detected, using current directory as base")

    @staticmethod
    def _is_runpod() -> bool:
        """Detect if running on RunPod by checking for volume mount."""
        return os.path.exists(EnvironmentConfig.RUNPOD_VOLUME)

    @staticmethod
    def is_runpod() -> bool:
        """Public method to check RunPod environment."""
        return EnvironmentConfig._is_runpod()

    @property
    def base_path(self) -> Path:
        """Get the base path for all storage."""
        return self._base_path

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a relative path to absolute path under base_path.

        Args:
            relative_path: Path relative to project root (e.g., "checkpoints/")

        Returns:
            Absolute path string
        """
        # Remove leading ./ if present
        clean_path = relative_path.lstrip("./")
        resolved = self._base_path / clean_path
        return str(resolved)

    def get_checkpoint_dir(self, config_dir: str) -> str:
        """Get checkpoint directory, prepending base path if relative.

        Args:
            config_dir: Checkpoint directory from config

        Returns:
            Resolved checkpoint directory path
        """
        if os.path.isabs(config_dir):
            return config_dir
        return self.resolve_path(config_dir)

    def get_log_dir(self, config_dir: str) -> str:
        """Get log directory, prepending base path if relative.

        Args:
            config_dir: Log directory from config

        Returns:
            Resolved log directory path
        """
        if os.path.isabs(config_dir):
            return config_dir
        return self.resolve_path(config_dir)

    def get_tokenizer_path(self, config_path: str) -> str:
        """Get tokenizer path, prepending base path if relative.

        Args:
            config_path: Tokenizer path from config

        Returns:
            Resolved tokenizer path
        """
        if os.path.isabs(config_path):
            return config_path
        return self.resolve_path(config_path)

    def get_data_dir(self, config_dir: str) -> str:
        """Get data directory, prepending base path if relative.

        Args:
            config_dir: Data directory from config

        Returns:
            Resolved data directory path
        """
        if os.path.isabs(config_dir):
            return config_dir
        return self.resolve_path(config_dir)


def resolve_env_vars(value: str) -> str:
    """Resolve environment variables in a string.

    Supports ${VAR} and $VAR syntax.

    Args:
        value: String potentially containing env vars

    Returns:
        String with env vars replaced by their values

    Example:
        resolve_env_vars("${HOME}/data")  # Returns "/home/user/data"
    """
    return os.path.expandvars(value)

"""Logger management and main logging functionality.

This module contains the LoggerManager class and convenience functions
for creating and configuring loggers.
"""

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

from .config import LogFormat, LoggerConfig, StructuredFormatter
from .filters import PerformanceFilter


class LoggerManager:
    """Centralized logger management."""

    _instances: ClassVar[dict[str, logging.Logger]] = {}
    _configured_loggers: ClassVar[set[str]] = set()

    @classmethod
    def get_logger(
        cls, name: str = "app", config: LoggerConfig | None = None
    ) -> logging.Logger:
        """Get or create a logger with the specified configuration.

        Args:
            name: Logger name
            config: Logger configuration

        Returns:
            Configured logger instance

        """
        if name in cls._instances and name in cls._configured_loggers:
            return cls._instances[name]

        if config is None:
            config = LoggerConfig(name=name)

        logger = logging.getLogger(name)

        # Avoid reconfiguring if already configured
        if name not in cls._configured_loggers:
            cls._configure_logger(logger, config)
            cls._configured_loggers.add(name)

        cls._instances[name] = logger
        return logger

    @classmethod
    def _configure_logger(cls, logger: logging.Logger, config: LoggerConfig):
        """Configure a logger with the specified configuration."""
        # Clear existing handlers
        logger.handlers.clear()

        # Set level
        logger.setLevel(getattr(logging, config.level.value))
        logger.propagate = config.propagate

        # Create formatters
        console_formatter = StructuredFormatter(
            format_type=config.console_format, include_extra=True
        )

        file_formatter = StructuredFormatter(
            format_type=config.file_format, include_extra=True
        )

        json_formatter = StructuredFormatter(
            format_type=LogFormat.JSON, include_extra=True
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # File handlers
        if config.enable_file_logging:
            config.log_dir.mkdir(parents=True, exist_ok=True)

            # Regular log file
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_dir / f"{config.name}.log",
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                config.log_dir / f"{config.name}_errors.log",
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            error_handler.setFormatter(file_formatter)
            error_handler.setLevel(logging.ERROR)
            logger.addHandler(error_handler)

        # JSON log file
        if config.enable_json_logging:
            config.log_dir.mkdir(parents=True, exist_ok=True)
            json_handler = logging.handlers.RotatingFileHandler(
                config.log_dir / f"{config.name}.json",
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            json_handler.setFormatter(json_formatter)
            logger.addHandler(json_handler)

        # Performance filter
        if config.enable_performance_tracking:
            perf_filter = PerformanceFilter(track_memory=True)
            for handler in logger.handlers:
                handler.addFilter(perf_filter)

    @classmethod
    def configure_from_dict(cls, config_dict: dict[str, Any]) -> logging.Logger:
        """Configure logger from dictionary configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configured logger

        """
        name = config_dict.get("name", "app")
        config = LoggerConfig(
            name=name,
            level=config_dict.get("level", "INFO"),
            log_dir=config_dict.get("log_dir"),
            console_format=LogFormat(config_dict.get("console_format", "console")),
            file_format=LogFormat(config_dict.get("file_format", "file")),
            enable_file_logging=config_dict.get("enable_file_logging", True),
            enable_json_logging=config_dict.get("enable_json_logging", False),
            max_file_size=config_dict.get("max_file_size", 10 * 1024 * 1024),
            backup_count=config_dict.get("backup_count", 5),
            enable_performance_tracking=config_dict.get(
                "enable_performance_tracking", False
            ),
            propagate=config_dict.get("propagate", False),
        )
        return cls.get_logger(name, config)

    @classmethod
    def configure_from_env(
        cls, name: str = "app", env_prefix: str = "LOG"
    ) -> logging.Logger:
        """Configure logger from environment variables.

        Args:
            name: Logger name
            env_prefix: Prefix for environment variables

        Returns:
            Configured logger

        Environment variables:
            {env_prefix}_LEVEL: Log level (default: INFO)
            {env_prefix}_DIR: Log directory (default: ./logs)
            {env_prefix}_CONSOLE_FORMAT: Console format (default: console)
            {env_prefix}_FILE_FORMAT: File format (default: file)
            {env_prefix}_ENABLE_FILE: Enable file logging (default: true)
            {env_prefix}_ENABLE_JSON: Enable JSON logging (default: false)
            {env_prefix}_MAX_SIZE: Max file size in bytes (default: 10485760)
            {env_prefix}_BACKUP_COUNT: Number of backup files (default: 5)
            {env_prefix}_PERFORMANCE: Enable performance tracking (default: false)

        """
        log_dir_str = os.getenv(f"{env_prefix}_DIR")
        log_dir: Path | None = Path(log_dir_str) if log_dir_str else Path("./logs")

        config = LoggerConfig(
            name=name,
            level=os.getenv(f"{env_prefix}_LEVEL", "INFO"),
            log_dir=log_dir,
            console_format=LogFormat(
                os.getenv(f"{env_prefix}_CONSOLE_FORMAT", "console")
            ),
            file_format=LogFormat(os.getenv(f"{env_prefix}_FILE_FORMAT", "file")),
            enable_file_logging=os.getenv(f"{env_prefix}_ENABLE_FILE", "true").lower()
            == "true",
            enable_json_logging=os.getenv(f"{env_prefix}_ENABLE_JSON", "false").lower()
            == "true",
            max_file_size=int(os.getenv(f"{env_prefix}_MAX_SIZE", "10485760")),
            backup_count=int(os.getenv(f"{env_prefix}_BACKUP_COUNT", "5")),
            enable_performance_tracking=os.getenv(
                f"{env_prefix}_PERFORMANCE", "false"
            ).lower()
            == "true",
        )
        return cls.get_logger(name, config)

    @classmethod
    def reset_all(cls):
        """Reset all loggers and configurations."""
        for logger_name in cls._instances:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)

        cls._instances.clear()
        cls._configured_loggers.clear()


# Convenience functions
def get_logger(
    name: str = "app",
    level: str = "INFO",
    enable_file_logging: bool = True,
    enable_json_logging: bool = False,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Get a pre-configured logger with sensible defaults.

    Args:
        name: Logger name
        level: Logging level
        enable_file_logging: Whether to enable file logging
        enable_json_logging: Whether to enable JSON logging
        log_dir: Directory for log files

    Returns:
        Configured logger

    """
    config = LoggerConfig(
        name=name,
        level=level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_json_logging=enable_json_logging,
    )
    return LoggerManager.get_logger(name, config)


def configure_logging_from_file(config_file: str | Path) -> logging.Logger:
    """Configure logging from a JSON configuration file.

    Args:
        config_file: Path to configuration file

    Returns:
        Configured logger

    """
    config_path = Path(config_file)
    if not config_path.exists():
        msg = f"Configuration file not found: {config_file}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    return LoggerManager.configure_from_dict(config_dict)

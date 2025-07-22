"""
Configuration classes and formatters for the logging module.

This module contains:
- LogLevel and LogFormat enumerations
- LoggerConfig configuration class
- StructuredFormatter for different output formats
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LogLevel(Enum):
    """Enumeration for log levels."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


class LogFormat(Enum):
    """Enumeration for log formats."""
    CONSOLE = "console"
    FILE = "file"
    JSON = "json"
    DETAILED = "detailed"


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that adds structured data and context to log records.
    
    Supports both JSON and human-readable formats with automatic field detection.
    """
    
    def __init__(self, 
                 format_type: LogFormat = LogFormat.CONSOLE,
                 include_extra: bool = True,
                 timestamp_format: str = "%Y-%m-%d %H:%M:%S",
                 **kwargs):
        """
        Initialize the structured formatter.
        
        Args:
            format_type: Type of format to use
            include_extra: Whether to include extra fields from log records
            timestamp_format: Format string for timestamps
        """
        self.format_type = format_type
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format
        
        # Define format strings for different types
        self.formats = {
            LogFormat.CONSOLE: "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            LogFormat.FILE: "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
            LogFormat.DETAILED: "%(asctime)s | %(name)s | %(levelname)s | %(pathname)s:%(lineno)d | %(funcName)s | PID:%(process)d | Thread:%(thread)d | %(message)s",
        }
        
        format_string = self.formats.get(format_type, self.formats[LogFormat.CONSOLE])
        super().__init__(format_string, timestamp_format, **kwargs)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        # Add runtime context
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        if self.format_type == LogFormat.JSON:
            return self._format_json(record)
        else:
            formatted = super().format(record)
            
            # Add extra fields if available and requested
            if self.include_extra and hasattr(record, '__dict__'):
                extra_fields = self._extract_extra_fields(record)
                if extra_fields:
                    extra_str = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
                    formatted += f" | {extra_str}"
            
            return formatted
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        log_data = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = self._extract_extra_fields(record)
            log_data.update(extra_fields)
        
        return json.dumps(log_data, default=str, ensure_ascii=False)
    
    def _extract_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields from log record."""
        # Standard fields that shouldn't be included as extra
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'exc_info', 'exc_text', 'stack_info', 'timestamp'
        }
        
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in standard_fields and not key.startswith('_'):
                extra_fields[key] = value
        
        return extra_fields


class LoggerConfig:
    """Configuration class for logger setup."""
    
    def __init__(self,
                 name: str = "venti",
                 level: Union[str, LogLevel] = LogLevel.INFO,
                 log_dir: Optional[Path] = None,
                 console_format: LogFormat = LogFormat.CONSOLE,
                 file_format: LogFormat = LogFormat.FILE,
                 enable_file_logging: bool = True,
                 enable_json_logging: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_performance_tracking: bool = False,
                 propagate: bool = False):
        """
        Initialize logger configuration.
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Directory for log files
            console_format: Format for console output
            file_format: Format for file output
            enable_file_logging: Whether to enable file logging
            enable_json_logging: Whether to enable JSON log files
            max_file_size: Maximum size per log file
            backup_count: Number of backup files to keep
            enable_performance_tracking: Whether to track performance metrics
            propagate: Whether to propagate to parent loggers
        """
        self.name = name
        self.level = level if isinstance(level, LogLevel) else LogLevel(level)
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.console_format = console_format
        self.file_format = file_format
        self.enable_file_logging = enable_file_logging
        self.enable_json_logging = enable_json_logging
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_performance_tracking = enable_performance_tracking
        self.propagate = propagate
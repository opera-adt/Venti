"""Custom filters for the logging module.

This module contains filters that enhance log records with additional
information such as performance metrics and memory usage.
"""

import logging
import time
import warnings


class PerformanceFilter(logging.Filter):
    """Filter that adds performance metrics to log records."""

    def __init__(self, track_memory: bool = False):
        """Initialize performance filter.

        Args:
            track_memory: Whether to track memory usage (requires psutil)

        """
        super().__init__()
        self.track_memory = track_memory
        self.start_time = time.time()

        if track_memory:
            try:
                import psutil

                self.process = psutil.Process()
            except ImportError:
                warnings.warn(
                    "psutil not available, memory tracking disabled", stacklevel=2
                )
                self.track_memory = False

    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to the record."""
        record.elapsed_time = time.time() - self.start_time

        if self.track_memory and hasattr(self, "process"):
            try:
                memory_info = self.process.memory_info()
                record.memory_mb = memory_info.rss / 1024 / 1024
            except Exception:
                # Don't fail logging if memory info unavailable
                pass

        return True


class SecurityFilter(logging.Filter):
    """Filter that sanitizes sensitive information from log records."""

    def __init__(self, sensitive_fields=None):
        """Initialize security filter.

        Args:
            sensitive_fields: List of field names to sanitize
            (default: common sensitive fields)

        """
        super().__init__()
        self.sensitive_fields = sensitive_fields or [
            "password",
            "token",
            "key",
            "secret",
            "credential",
            "authorization",
            "cookie",
            "session_id",
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize sensitive information from the record."""
        # Sanitize the message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for field in self.sensitive_fields:
                if field.lower() in record.msg.lower():
                    # Simple sanitization - replace with asterisks
                    record.msg = record.msg.replace(field, "*" * len(field))

        # Sanitize extra fields
        for key, value in list(record.__dict__.items()):
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                if isinstance(value, str):
                    record.__dict__[key] = "*" * min(len(value), 8)
                else:
                    record.__dict__[key] = "***"

        return True


class ThrottleFilter(logging.Filter):
    """Filter that throttles repeated log messages."""

    def __init__(self, max_rate: int = 10, time_window: int = 60):
        """Initialize throttle filter.

        Args:
            max_rate: Maximum number of messages per time window
            time_window: Time window in seconds

        """
        super().__init__()
        self.max_rate = max_rate
        self.time_window = time_window
        self.message_counts: dict[tuple[str, int], int] = {}
        self.last_reset = time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether the log record should be emitted.

        Returns True if the record should be logged, False if it is throttled.
        """
        now = time.time()

        # Reset counts if time window has passed
        if now - self.last_reset > self.time_window:
            self.message_counts.clear()
            self.last_reset = now

        key: tuple[str, int] = (record.msg, record.levelno)
        count: int = self.message_counts.get(key, 0)

        if count >= self.max_rate:
            return False

        self.message_counts[key] = count + 1
        return True

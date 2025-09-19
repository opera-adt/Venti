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
                warnings.warn("psutil not available, memory tracking disabled", stacklevel=2)
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
            sensitive_fields: List of field names to sanitize (default: common sensitive fields)

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
        self.message_counts = {}
        self.last_reset = time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        """Throttle repeated messages."""
        current_time = time.time()

        # Reset counts if time window has passed
        if current_time - self.last_reset > self.time_window:
            self.message_counts.clear()
            self.last_reset = current_time

        # Create message key (combination of logger name and message)
        message_key = f"{record.name}:{record.getMessage()}"

        # Count this message
        self.message_counts[message_key] = self.message_counts.get(message_key, 0) + 1

        # Allow message if under the rate limit
        if self.message_counts[message_key] <= self.max_rate:
            return True
        elif self.message_counts[message_key] == self.max_rate + 1:
            # Log a throttling message on the first dropped message
            record.msg = f"[THROTTLED] Message rate exceeded for: {record.getMessage()}"
            return True
        else:
            # Drop the message
            return False

"""Utility functions and context managers for the logging module.

This module contains context managers for temporary logging configuration
and performance monitoring, as well as demo and helper functions.
"""

import logging
import logging.handlers
import time

from .logger import get_logger


class TemporaryLogLevel:
    """Context manager for temporarily changing log level."""

    def __init__(self, logger: logging.Logger, level: str):
        """Initialize context with logger and level."""
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


class LogPerformance:
    """Context manager for logging performance metrics."""

    def __init__(self, logger: logging.Logger, operation_name: str):
        """Initialize performance logging context."""
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(
                f"{self.operation_name} failed after {duration:.3f}s",
                exc_info=(exc_type, exc_val, exc_tb),
            )
        else:
            self.logger.info(
                f"{self.operation_name} completed in {duration:.3f}s",
                extra={"duration": duration, "operation": self.operation_name},
            )


class LogContext:
    """Context manager for adding consistent context to all log messages."""

    def __init__(self, logger: logging.Logger, **context):
        """Initialize context logging with additional metadata."""
        self.logger = logger
        self.context = context
        self.original_makeRecord = None

    def __enter__(self):
        # Store original makeRecord method
        self.original_makeRecord = self.logger.makeRecord

        # Create wrapper that adds context
        def make_record_with_context(*args, **kwargs):
            record = self.original_makeRecord(*args, **kwargs)
            # Add context to the record
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        # Replace makeRecord method
        self.logger.makeRecord = make_record_with_context
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original makeRecord method
        if self.original_makeRecord:
            self.logger.makeRecord = self.original_makeRecord


def log_function_calls(
    logger: logging.Logger | None = None,
    log_args: bool = True,
    log_result: bool = True,
    log_level: str = "DEBUG",
):
    """Configure request logging for web applications.

    Args:
        logger: Logger to use (if None, creates one based on function module)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_level: Log level to use

    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create logger
            func_logger = logger or get_logger(func.__module__)
            level = getattr(logging, log_level.upper())

            # Log function entry
            log_data = {
                "function": func.__name__,
                "module": func.__module__,
                "action": "entry",
            }

            if log_args:
                log_data["args"] = str(args) if args else None
                log_data["kwargs"] = kwargs if kwargs else None

            func_logger.log(level, f"Entering function {func.__name__}", extra=log_data)

            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log function exit
                exit_data = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "action": "exit",
                    "duration": duration,
                    "success": True,
                }

                if log_result:
                    exit_data["result"] = str(result) if result is not None else None

                func_logger.log(
                    level, f"Exiting function {func.__name__}", extra=exit_data
                )
            except Exception as e:
                # Log function exception
                duration = time.time() - start_time
                error_data = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "action": "exception",
                    "duration": duration,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

                func_logger.error(
                    f"Exception in function {func.__name__}",
                    extra=error_data,
                    exc_info=True,
                )
                raise
            else:
                return result

        return wrapper

    return decorator


def setup_request_logging(logger: logging.Logger):
    """Configure request logging for web applications..

    Args:
        logger: Logger to configure

    Returns:
        Configured logger with request context

    """
    # This is a helper function that can be used with web frameworks
    # The actual implementation would depend on the specific framework

    class RequestFilter(logging.Filter):
        def __init__(self, get_request_id_func):
            super().__init__()
            self.get_request_id = get_request_id_func

        def filter(self, record):
            # Add request ID to all log records
            request_id = self.get_request_id()
            if request_id:
                record.request_id = request_id
            return True

    # This would be customized based on the web framework being used
    def get_request_id():
        # Placeholder - implement based on your web framework
        return None

    request_filter = RequestFilter(get_request_id)
    for handler in logger.handlers:
        handler.addFilter(request_filter)

    return logger


def demo_logging():
    """Demonstrate the logging module capabilities."""
    print("=== Logging Module Demo ===\n")

    # Basic logger
    logger = get_logger("demo", level="DEBUG")

    # Basic logging
    logger.debug("This is a debug message")
    logger.info("Application started successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Structured logging with extra fields
    logger.info(
        "User action",
        extra={
            "user_id": "12345",
            "action": "login",
            "ip_address": "192.168.1.1",
            "success": True,
        },
    )

    # Performance logging
    with LogPerformance(logger, "data_processing"):
        time.sleep(0.1)  # Simulate work

    # Temporary log level change
    with TemporaryLogLevel(logger, "ERROR"):
        logger.info("This won't be logged")
        logger.error("This will be logged")

    # Context logging
    with LogContext(logger, service="demo_service", version="1.0.0"):
        logger.info("Service operation completed")

    # Function decorator demo
    @log_function_calls(logger)
    def sample_function(x, y):
        """Sample function for demonstrating decorator."""
        return x + y

    result = sample_function(5, 3)
    logger.info(f"Function result: {result}")

    logger.info("Demo completed")


def create_rotating_file_logger(
    name: str,
    filename: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    level: str = "INFO",
) -> logging.Logger:
    """Create a simple rotating file logger.

    Args:
        name: Logger name
        filename: Log file path
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        level: Log level

    Returns:
        Configured logger

    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Create rotating file handler
    handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=max_bytes, backupCount=backup_count
    )

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:  # Avoid duplicate handlers
        logger.addHandler(handler)

    return logger


def silence_logger(logger_name: str):
    """Silence a specific logger by setting it to CRITICAL level.

    Args:
        logger_name: Name of the logger to silence

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)


def get_memory_usage() -> float | None:
    """Get current memory usage in MB.

    Returns:
        Memory usage in MB or None if psutil is not available

    """
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None

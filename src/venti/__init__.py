"""Venti package for calibrating OPERA DISP with GNSS."""

import sys
from typing import Any

# Lazy imports to avoid pyproj initialization errors
_LAZY_MODULES = {
    'models': '.models',
    'unwrap': '.unwrap',
}


def __getattr__(name: str) -> Any:
    """Lazily import submodules on first access."""
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name], package=__name__)
        # Cache the module
        globals()[name] = module
        return module

    if name == '__version__':
        try:
            from ._version import __version__
            globals()['__version__'] = __version__
            return __version__
        except ImportError:
            return "unknown"

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes."""
    return list(_LAZY_MODULES.keys()) + ['__version__']


__all__ = ['models', 'unwrap', '__version__']
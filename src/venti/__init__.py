"""Venti package for calibrating OPERA DISP with GNSS."""

from typing import Any

# Lazy imports to avoid pyproj initialization errors
_LAZY_MODULES = {
    "models": ".models",
    "unwrap": ".unwrap",
    "workflow": ".workflow",
    "gnss": ".gnss",
    "io": ".io",
    "interpolation": ".interpolation",
    "raster": ".raster",
    "spatial": ".spatial",
}


def __getattr__(name: str) -> Any:
    """Lazily import submodules on first access."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], package=__name__)
        # Cache the module
        globals()[name] = module
        return module

    if name == "__version__":
        try:
            from ._version import __version__

            globals()["__version__"] = __version__
        except ImportError:
            return "unknown"
        else:
            return __version__

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    """List available attributes."""
    return [*list(_LAZY_MODULES.keys()), "__version__"]


__all__ = [
    "__version__",
    "io",
    "models",
    "unwrap",
    "workflow",
]  # , 'gnss', 'interpolation', 'raster', 'spatial']

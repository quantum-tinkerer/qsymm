"""Version helper used by package runtime imports."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qsymm")
except PackageNotFoundError:
    try:
        from . import _static_version

        __version__ = getattr(_static_version, "__version__")
    except Exception:  # pragma: no cover - fallback for uninstalled source trees
        __version__ = "0+unknown"

__all__ = ["__version__"]

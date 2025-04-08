"""MD Contact Analysis package."""
__version__ = "0.1.0"

try:
    from . import contact_map_cython
except ImportError:
    pass

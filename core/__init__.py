from .data import (
    DEFAULT_PARTICLE_FEATURES,
    build_binary_labels,
    discover_root_files,
    iter_dense_batches,
    load_split_arrays,
)

__all__ = [
    "DEFAULT_PARTICLE_FEATURES",
    "SimpleParT",
    "build_binary_labels",
    "discover_root_files",
    "iter_dense_batches",
    "load_split_arrays",
]


def __getattr__(name: str):
    if name == "SimpleParT":
        from .model import SimpleParT

        return SimpleParT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

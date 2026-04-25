"""
csnewton — Newton-GMRES/LGMRES with complex-step Jacobian-vector products.

Reference:
    D. Mitsotakis, "The complex-step Newton method and its convergence",
    Numerische Mathematik 157(3), 993-1021, 2025.
    https://link.springer.com/article/10.1007/s00211-025-01471-w
"""

from ._csnewton import (  # noqa: F401
    gmres,
    gmres_z,
    lgmres,
    lgmres_z,
    csnewton,
)

__all__ = ["gmres", "gmres_z", "lgmres", "lgmres_z", "csnewton"]

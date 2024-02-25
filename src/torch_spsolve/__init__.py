"""A pytorch package that can perform sparse solves on the CPU and GPU seamlessly including autograd functionality.
"""
from .torch_spsolve import spsolve, TorchSparseOp
__version__ = "1.0"
__author__ = "Thomas Grandits"

__all__ = ["spsolve", "TorchSparseOp"]

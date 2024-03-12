import torch
from torch.nn import Module
from torch.autograd.function import once_differentiable
from torch.autograd.function import FunctionCtx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve as spsolve_np
from typing import Callable

_cupy_available = True
try:
    import cupy as cp
    from cupyx.scipy.sparse import coo_matrix as coo_matrix_cp
    from cupyx.scipy.sparse.linalg import splu as splu_cp
    from cupyx.scipy.sparse.linalg import spsolve as spsolve_cp
except ModuleNotFoundError as err:
    import warnings
    warnings.warn("Cupy not found, GPU version of torch_spsolve not available. Error message: ")
    warnings.warn(err.msg)
    _cupy_available = False

_torch_dtype_map = {torch.float32: np.float32,
                    torch.float64: np.float64,
                    torch.complex64: np.complex64,
                    torch.complex128: np.complex128}


def _solve_f(tens : np.ndarray) -> Callable[[np.ndarray], np.ndarray]: return (spsolve_cp if tens.is_cuda else spsolve_np)


def _convert_f(x : torch.Tensor):
    # Converts a tensor to a cupy or numpy array
    if x.is_cuda:
        return cp.asarray(x.detach())
    else:
        return x.detach().numpy()


def _inv_convert_f(x : np.ndarray, dtype : torch.dtype = None, device : torch.device = None) -> torch.Tensor:
    # Converts a cupy or numpy array to a tensor. The provided tensor will be converted to the desired dtype and device.
    if device is not None and "cuda" in device.type:
        return torch.as_tensor(x, dtype=dtype, device=device)
    else:
        return torch.from_numpy(x).to(dtype=dtype, device=device)


def _convert_sparse_to_lib(tens : torch.Tensor) -> coo_matrix:
    assert tens.is_coalesced(), f"Uncoalesced tensor found, usage is only supported through {TorchSparseOp.__name__}"
    indices = tens.indices()
    data = tens.values().detach()
    assert tens.dtype in _torch_dtype_map, (f"Unknown or unsupported dtype: {tens.dtype}."
                                            + " Only {list(_torch_dtype_map.keys())} are supported.")
    if tens.is_cuda:
        return coo_matrix_cp((cp.asarray(data), (cp.asarray(indices[0]), cp.asarray(indices[1]))),
                             shape=tens.shape, dtype=_torch_dtype_map[tens.dtype])
    else:
        return coo_matrix((data.numpy(), (indices[0].numpy(), indices[1].numpy())),
                          shape=tens.shape, dtype=_torch_dtype_map[tens.dtype])


class TorchSparseOp(Module):
    r"""A solver class that will convert given sparse tensor to allow for sparse linear system solutions of the type

    .. math::
        A(\theta_1) y = x(\theta_2),

    where the resulting tensor :math:`y` can be backpropgated w.r.t. :math:`\theta_1` and :math:`\theta_2`.
    The instance can be re-used to solve for multiple different :math:`x`.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        The sparse tensor :math:`A(\theta_1)` of the above linear system of shape [N, N].
    """
    def __init__(self, sparse_tensor : torch.Tensor) -> None:
        assert sparse_tensor.ndim == 2, "TorchSparseOp only provides support 2-dimensional sparse matrices"
        assert sparse_tensor.is_sparse, "TorchSparseOp expects a sparse tensor."
        assert not sparse_tensor.is_cuda or _cupy_available, ("Provided tensor is on the GPU, but cupy is not available. "
                                                              + "If you intend to use the CPU version, call .cpu() first.")
        self.tens = sparse_tensor.coalesce()
        self.tens_copy = _convert_sparse_to_lib(self.tens).tocsr()
        self.tens_copy_H = _convert_sparse_to_lib(self.tens.H.coalesce()).tocsr()
        self.factor = None
        self._convert_f = _convert_f
        self._inv_convert_f = lambda x: _inv_convert_f(x, sparse_tensor.dtype, sparse_tensor.device)
        self._solve_f = _solve_f(self)
        self._factorized = False

    def factorize(self):
        """Factorizes the system using a sparse LU decomposition.
        The system can be much more efficiently solved after the factorization, but factorization may take long.
        """
        tens_copy = self.tens_copy.tocsc()
        if self.is_cuda:
            self.factors = (splu_cp(tens_copy), splu_cp(self.tens_copy_H.tocsc()))
        else:
            self.factors = (splu(tens_copy), splu(self.tens_copy_H.tocsc()))
        self._factorized = True

    def solve(self, x : torch.Tensor) -> torch.Tensor:
        """Solves the system :math:`Ay = x`, for the initially provided sparse operator :math:`A` and
        the given parameter :math:`x`. The resulting :math:`y` will be returned.


        Parameters
        ----------
        x : torch.Tensor
            A dense tensor of shape [N], or [N, M]. Will be converted to the dtype of :math:`A`.

        Returns
        -------
        torch.Tensor
            A dense tensor of shape [N], or [N, M], which represents :math:`y`.
        """
        assert x.device == self.device, f"RHS resides on the wrong device {x.device} vs. {self.device}"
        assert not x.is_sparse, "RHS needs to be dense"
        return _TorchSparseOpF.apply(self.tens, x, self)

    def _solve_impl(self, x : torch.Tensor, transpose=False) -> torch.Tensor:
        # Internal solution function to call the correct spsolve of SuperLU function.
        assert x.device == self.device, f"RHS resides on the wrong device {x.device} vs. {self.device}"
        x_ = self._convert_f(x)
        if self._factorized:
            factor = (self.factors[1] if transpose else self.factors[0])
            sol = factor.solve(x_)
        else:
            if transpose:
                sol = self._solve_f(self.tens_copy_H, x_)
            else:
                sol = self._solve_f(self.tens_copy, x_)

        sol = self._inv_convert_f(sol)
        return sol

    def __getattr__(self, name):
        assert hasattr(self.tens, name), f"Method {name} unknown"
        return getattr(self.tens, name)


class _TorchSparseOpF(torch.autograd.Function):
    # Internal sparse operator function that takes the sparse operator :ref:`TorchSparseOp` and
    # returns the solution while keeping the results and operators for backpropagation.
    @staticmethod
    def forward(ctx : FunctionCtx, tens_op : torch.Tensor, rhs : torch.Tensor, sparse_op : TorchSparseOp):
        # Forward operator computing and returning the solution of the linear system.
        sol = sparse_op._solve_impl(rhs, False)
        sol.requires_grad = tens_op.requires_grad or rhs.requires_grad
        ctx.sparse_op = sparse_op
        ctx.save_for_backward(tens_op, sol)

        return sol

    @staticmethod
    @once_differentiable
    def backward(ctx : FunctionCtx, grad_in : torch.Tensor):
        # Backward operator, loading the results from the forward pass, and solving the adjoint
        # system w.r.t. the gradient. Additionally, the gradients w.r.t. the non-zero entries
        # of A will also be computed and returned.
        sparse_op = ctx.sparse_op
        tens_op, sol = ctx.saved_tensors
        sparse_op : TorchSparseOp
        sol : torch.Tensor
        grad_rhs = sparse_op._solve_impl(grad_in, True)

        indices = tens_op.indices()
        sol_inds = sol[indices[1]]
        # Complex gradient requires an additional conjugation
        if sol_inds.dtype.is_complex:
            sol_inds = torch.conj(sol_inds)

        sparse_op_diff_values = -(grad_rhs[indices[0]] * sol_inds)
        # Multi-rhs check
        if sparse_op_diff_values.ndim > 1:
            sparse_op_diff_values = sparse_op_diff_values.flatten()
            indices = indices[..., None].expand(list(indices.shape) + [sol.shape[-1]]).reshape([2, -1])

        grad_op = torch.sparse_coo_tensor(values=sparse_op_diff_values, indices=indices, size=tens_op.shape,
                                          dtype=tens_op.dtype, device=tens_op.device)

        return grad_op, grad_rhs, None


def spsolve(sparse_op : torch.Tensor, x : torch.Tensor) -> torch.Tensor:
    r"""The solution function to solve the linear system

    .. math::
        A(\theta_1) y = x(\theta_2).

    Internally, this creates an instance of the class :class:`TorchSparseOp` and will call  :meth:`TorchSparseOp.solve`,
    before freeing the instance immediately. For repeated solves of the same method, it is recommended to create and
    keep the instance, possibly factorizing using :meth:`TorchSparseOp.factorize` to significantly speed up any
    subsequent solutions.

    Parameters
    ----------
    sparse_op : torch.Tensor
        The sparse tensor :math:`A(\theta_1)` of the above linear system of shape [N, N].
    x : torch.Tensor
        A dense tensor of shape [N], or [N, M]. Will be converted to the dtype of :math:`A`.

    Returns
    -------
    torch.Tensor
        A dense tensor of shape [N], or [N, M], which represents :math:`y`.
    """
    return TorchSparseOp(sparse_op).solve(x)

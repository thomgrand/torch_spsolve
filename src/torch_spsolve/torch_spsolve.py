import torch
from torch.nn import Module
from torch.autograd.function import once_differentiable
from torch.autograd.function import FunctionCtx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
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
    warnings.warn("Cupy not found, GPU version of torch_spsolve not available")
    _cupy_available = False

_torch_dtype_map = {torch.float32: np.float32,
                    torch.float64: np.float64}

_solve_f = lambda tens: (spsolve_cp if tens.is_cuda else spsolve_np)

def _convert_f(x : torch.Tensor):
    if x.is_cuda:
        return cp.asarray(x.detach())
    else:
        return x.detach().numpy()
    
def _inv_convert_f(x : np.ndarray, tens : torch.Tensor):
    if tens.is_cuda:
        return torch.as_tensor(x, device=tens.device).to(dtype=tens.dtype)
    else:
        return torch.from_numpy(x).to(dtype=tens.dtype, device=tens.device)

def _convert_sparse_to_lib(tens : torch.Tensor):
    assert tens.is_coalesced(), f"Uncoalesced tensor found, usage is only supported through {TorchSparseOp.__name__}"
    indices = tens.indices()
    data = tens.values().detach()
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
        :math:`A(\theta_1)` of the above linear system
    """
    def __init__(self, sparse_tensor : torch.Tensor) -> None:
        #super().__init__()
        assert sparse_tensor.ndim == 2, "TorchSparseOp only provides support 2-dimensional sparse matrices"
        assert sparse_tensor.is_sparse, "TorchSparseOp expects a sparse tensor."
        assert not sparse_tensor.is_cuda or _cupy_available, ("Provided tensor is on the GPU, but cupy is not available. " 
                                                              + "If you intend to use the CPU version, call .cpu() first.")
        self.tens = sparse_tensor.coalesce()
        self.tens_copy = _convert_sparse_to_lib(self.tens).tocsr()
        self.tens_copy_T = _convert_sparse_to_lib(self.tens.T.coalesce()).tocsr()
        self.factor = None
        self._convert_f = _convert_f
        self._inv_convert_f = lambda x: _inv_convert_f(x, self)
        self._solve_f = _solve_f(self)
        self._factorized = False

    def factorize(self):
        """Factorizes the system using a sparse LU decomposition.
        The resulting system 
        """
        tens_copy = self.tens_copy.tocsc()
        if self.is_cuda:
            self.factors = (splu_cp(tens_copy), splu_cp(self.tens_copy.T))
        else:
            self.factors = (splu(tens_copy), splu(self.tens_copy.T))
        self._factorized = True

    def solve(self, x : torch.Tensor):
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assert x.device == self.device, f"RHS resides on the wrong device {x.device} vs. {self.device}"
        assert not x.is_sparse, f"RHS needs to be dense"
        return _TorchSparseOpF.apply(self.tens, x, self)
        
    def _solve_impl(self, x : torch.Tensor, transpose=False) -> torch.Tensor:
        #requires_grad = x.requires_grad
        assert x.device == self.device, f"RHS resides on the wrong device {x.device} vs. {self.device}"
        x_ = self._convert_f(x)
        if self._factorized:
            factor = (self.factors[1] if transpose else self.factors[0])
            sol = factor.solve(x_)
        else:
            if transpose:
                sol = self._solve_f(self.tens_copy_T, x_)
            else:
                sol = self._solve_f(self.tens_copy, x_)

        sol = self._inv_convert_f(sol)
        return sol

    @property
    def _dtype(self):
        return _torch_dtype_map[self.tens.dtype]

    @property
    def _lib(self):
        if self.is_cuda:
            return cp
        else:
            return np
        
    #def __getattribute__(self, __name: str) -> torch.Any:
    #    return self.tens.__getattribute__(__name)

    def __getattr__(self, name):
        assert hasattr(self.tens, name), f"Method {name} unknown"
        return getattr(self.tens, name)
        


class _TorchSparseOpF(torch.autograd.Function):
    @staticmethod
    def forward(ctx : FunctionCtx, tens_op : torch.Tensor, rhs : torch.Tensor,
                 sparse_op : TorchSparseOp #Variables not needed in the backprop
                ):
        #rhs_conv = _convert_f(rhs)
        #sol = _inv_convert_f(sparse_op._solve_f(sparse_op.tens_copy, rhs_conv), rhs)
        sol = sparse_op._solve_impl(rhs, False)
        sol.requires_grad = tens_op.requires_grad or rhs.requires_grad
        ctx.sparse_op = sparse_op
        ctx.save_for_backward(tens_op, sol)
        return sol
        

    @staticmethod
    @once_differentiable
    def backward(ctx : FunctionCtx, grad_in : torch.Tensor):
        sparse_op = ctx.sparse_op
        tens_op, sol = ctx.saved_tensors
        sparse_op : TorchSparseOp
        sol : torch.Tensor
        grad_rhs = sparse_op._solve_impl(grad_in, True)

        indices = tens_op.indices()
        sparse_op_diff_values = -(grad_rhs[indices[0]] * sol[indices[1]])
        #Multi-rhs check
        if sparse_op_diff_values.ndim > 1:
            sparse_op_diff_values = sparse_op_diff_values.flatten()
            indices = indices[..., None].expand(list(indices.shape) + [sol.shape[-1]]).reshape([2, -1])

        grad_op = torch.sparse_coo_tensor(values=sparse_op_diff_values, indices=indices, size=tens_op.shape, 
                                          dtype=tens_op.dtype, device=tens_op.device)
        return grad_op, grad_rhs, None

def spsolve(sparse_op : torch.Tensor, rhs : torch.Tensor) -> torch.Tensor:
    return TorchSparseOp(sparse_op).solve(rhs)
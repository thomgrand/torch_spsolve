""" tests for torch_sparse_solve """
import numpy as np
import torch
import pytest
import scipy.sparse
import torch_spsolve
from typing import Dict
from itertools import product
from scipy.sparse.linalg import spsolve as spsolve_ref
from scipy.sparse import coo_matrix

_cuda_available = torch.cuda.device_count() > 0

@pytest.fixture
def rand_gen():
    #Fix the random seed in each test
    return torch.manual_seed(0)


@pytest.fixture
def create_sparse_system(rand_gen, request : Dict): #size : int, dtype : torch.dtype, device : torch.device) -> torch.Tensor:
    #Creates a random sparse linear system
    size, dtype, device = request.param["size"], request.param["dtype"], request.param["device"]
    dense_op = torch.randn(size=[size, size], dtype=dtype, device=device)
    zero_mask = torch.rand(size=[size, size], dtype=dtype, device=device) < 0.2
    dense_op = dense_op * zero_mask
    dense_op += 1e-2 * torch.eye(size, dtype=dtype, device=device) #Make the system well-posed
    nnz = dense_op.nonzero()
    values = dense_op[nnz[:, 0], nnz[:, 1]]
    values.requires_grad = True
    sparse_op = torch.sparse_coo_tensor(values=values, indices=nnz.T, dtype=dtype, device=device)
    if "multi_rhs" in request.param and request.param["multi_rhs"]:
        rhs = torch.randn(size=[size, 10], dtype=dtype, device=device)
    else:
        rhs = torch.randn(size=[size], dtype=dtype, device=device)
    rhs.requires_grad = True
    if "ret_param" in request.param and request.param["ret_param"]:
        return sparse_op, rhs, values
    else:
        return sparse_op, rhs

def _sparse_tensor_to_np(sparse_op : torch.Tensor) -> coo_matrix:
    sparse_op = sparse_op.detach().cpu().coalesce()
    nnz = sparse_op.indices().numpy()
    values = sparse_op.values().numpy()
    return coo_matrix((values, nnz)).tocsr()

_sizes_sparse_op = [20, 50, 200]
_dtypes = [torch.float32, torch.float64]
_devices = [torch.device("cpu")] + ([torch.device("cuda")] if _cuda_available else [])
_multi_rhs = [False, True]
_sparse_op_param_dicts = [{"size": s, "dtype": dt, "device": de, "multi_rhs": mr} for s, dt, de, mr in product(_sizes_sparse_op, _dtypes, _devices, 
                                                                                                                  _multi_rhs)]
#For the gradient checks, we only allow double precision (single is too inaccurate for gradient estimation in our case)
_sparse_op_param_grad_dicts = [d | dict(ret_param=True) for d in _sparse_op_param_dicts if d["size"] < 50 and d["dtype"] == torch.float64]
_sparse_op_param_grad_multi_rhs_dicts = [d for d in _sparse_op_param_grad_dicts if d["multi_rhs"]]
_sparse_op_param_grad_single_rhs_dicts = [d for d in _sparse_op_param_grad_dicts if not d["multi_rhs"]]

def _test_against_scipy(sparse_op : torch.Tensor, rhs : torch.Tensor, sol_lib : torch.Tensor):
    dtype = rhs.dtype
    sol_lib = sol_lib.detach().cpu().numpy()
    sol_ref = spsolve_ref(_sparse_tensor_to_np(sparse_op), rhs.detach().cpu().numpy())
    np.testing.assert_almost_equal(sol_lib, sol_ref, decimal=(2 if dtype == torch.float32 else 6))

@pytest.mark.parametrize("create_sparse_system", _sparse_op_param_grad_single_rhs_dicts, indirect=True)
def test_gradcheck_single_rhs(create_sparse_system):
    #Check for correctness of the gradient
    sparse_op, rhs, _ = create_sparse_system
    assert torch.autograd.gradcheck(torch_spsolve.spsolve, [sparse_op, rhs], check_sparse_nnz=True, check_undefined_grad=False)

@pytest.mark.parametrize("create_sparse_system", _sparse_op_param_grad_multi_rhs_dicts, indirect=True)
def test_gradcheck_multi_rhs(create_sparse_system):
    #Check for correctness of the gradient for multiple rhs
    sparse_op, rhs, _ = create_sparse_system
    assert torch.autograd.gradcheck(torch_spsolve.spsolve, [sparse_op, rhs], check_sparse_nnz=True, check_undefined_grad=False)

#TODO: Is the test below really necessary?
@pytest.mark.parametrize("create_sparse_system", _sparse_op_param_grad_single_rhs_dicts, indirect=True)
def test_gradcheck_single(create_sparse_system):
    #Check for correctness of the gradient when only deriving w.r.t. one of the parameters
    sparse_op, rhs, sparse_op_params = create_sparse_system

    with torch.no_grad():
        target_sol = torch_spsolve.spsolve(sparse_op.detach(), rhs) + torch.randn(size=[rhs.numel()], 
                                                                                  dtype=rhs.dtype, device=rhs.device)
    loss_f = lambda x: 0.5 * torch.mean((x - target_sol)**2)

    sparse_op_params.grad = sparse_op.grad = rhs.grad = None
    #Test grad in rhs only
    loss_f(torch_spsolve.spsolve(sparse_op.detach(), rhs)).backward()
    assert sparse_op_params.grad is None and sparse_op.grad is None
    assert torch.any(rhs.grad != 0.)

    #Test grad in sparse_op only
    sparse_op_params.grad = sparse_op.grad = rhs.grad = None
    loss_f(torch_spsolve.spsolve(sparse_op, rhs.detach())).backward()
    assert rhs.grad is None
    assert torch.any(sparse_op_params.grad != 0.)

@pytest.mark.parametrize("create_sparse_system", _sparse_op_param_dicts, indirect=True)
def test_result(create_sparse_system):
    #Tests the result against scipy's reference implementation
    sparse_op, rhs = create_sparse_system
    sol_lib = torch_spsolve.spsolve(sparse_op, rhs)
    _test_against_scipy(sparse_op, rhs, sol_lib)

@pytest.mark.parametrize("create_sparse_system", _sparse_op_param_dicts, indirect=True)
def test_instance_call(create_sparse_system):
    sparse_op, rhs = create_sparse_system
    sparse_op_instance = torch_spsolve.TorchSparseOp(sparse_op)
    _test_against_scipy(sparse_op, rhs, sparse_op_instance.solve(rhs))

    #Change if the result is robust to changes in the rhs
    rhs = rhs + torch.randn(size=rhs.shape, dtype=rhs.dtype, device=rhs.device)
    _test_against_scipy(sparse_op, rhs, sparse_op_instance.solve(rhs))

@pytest.mark.parametrize("create_sparse_system", _sparse_op_param_dicts, indirect=True)
def test_factorization(create_sparse_system):
    sparse_op, rhs = create_sparse_system
    sparse_op_instance = torch_spsolve.TorchSparseOp(sparse_op)
    sparse_op_instance.factorize()
    _test_against_scipy(sparse_op, rhs, sparse_op_instance.solve(rhs))

@pytest.mark.parametrize("create_sparse_system", [dict(size=20, dtype=torch.float64, device=torch.device("cuda"), multi_rhs=True)], indirect=True)
def test_sparse_rhs(create_sparse_system):
    sparse_op, rhs = create_sparse_system
    with torch.no_grad():
        rhs[rhs < 0.2] = 0.
        rhs = rhs.to_sparse()

    with pytest.raises(AssertionError):
        torch_spsolve.spsolve(sparse_op, rhs)

@pytest.mark.skipif(not _cuda_available, reason="Cuda not available")
@pytest.mark.parametrize("create_sparse_system", [dict(size=20, dtype=torch.float64, device=torch.device("cuda"))], indirect=True)
def test_missing_cupy(create_sparse_system):
    #Checks if transmitting a cuda tensor without installed cupy will crash
    torch_spsolve.torch_spsolve._cupy_available = False
    sparse_op, rhs = create_sparse_system
    with pytest.raises(AssertionError):
        torch_spsolve.TorchSparseOp(sparse_op)
    
    with pytest.raises(AssertionError):
        torch_spsolve.spsolve(sparse_op, rhs)

if __name__ == "__main__":
    pytest.main([__file__])

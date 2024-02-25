# Torch spsolve

[![CI Tests](https://github.com/thomgrand/torch_spsolve/actions/workflows/python-package.yml/badge.svg)](https://github.com/thomgrand/torch_spsolve/actions/workflows/python-package.yml)

This library implements functionality similar to [spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html) in scipy to solve linear systems of the type `Ay = x`.
The provided functions work seamlessly with gradients in `A` (nonzeros only) and `y` and are implemented on the CPU (requires [scipy](https://scipy.org/)) and GPU (requires [cupy](https://cupy.dev/)).

# Installation

The library can easily be installed using pip
```bash
pip install git+https://github.com/thomgrand/torch_spsolve
```

Note that to use the library on the GPU, cupy needs to be installed. In case you want to install it from source, you can simply call

```bash
pip install git+https://github.com/thomgrand/torch_spsolve[gpu]
```

Pre-built binaries are also available for cupy though. Further information is available here [https://docs.cupy.dev/en/stable/install.html#installing-cupy](https://docs.cupy.dev/en/stable/install.html#installing-cupy).

# Usage

In the simplest case, you can simply call `torch_spsolve.spsolve` directly.
```python
import torch_spsolve
A = torch.randn(size=[50, 50])
A[A < 0] = 0.
A = A.to_sparse()
x = torch.randn(size=[50])
y = torch_spsolve.spsolve(A, x)
```

Internally, this creates an instance of `torch_spsolve.TorchSparseOp` and uses it solve method.
```python
solver = torch_spsolve.TorchSparseOp(A)
y = solver.solve(x)
```

In case you need to solve for multiple right hand sides (`x`), you can either keep the solver instance, or directly call any of the methods with a 2-dimensional `x`.

If you plan on using the same operator `A` for many solves, it probably pays off to pre-factorize the system. This can be significantly faster than using `spsolve` repeatedly. Internally, this uses the SuperLU library.

```python
solver.factorize()
y = solver.solve(x)
```

# Details

Internally, the function will convert the arrays and sparse operator to their corresponding scipy or cupy representations and solve them before converting them back to pytorch tensors.

More details on how the derivatives are computed can be found at [https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html](https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html).
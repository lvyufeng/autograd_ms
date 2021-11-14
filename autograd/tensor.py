import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union
import mindspore.ops.operations as P
from mindspore._c_expression import Tensor as _Tensor
from mindspore.common import dtype as mstype
from ._utils import _tensor_getitem

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[['Tensor'], _Tensor]

Arrayable = Union[float, list, _Tensor, np.ndarray]

Tensorable = Union['Tensor', float, np.ndarray, _Tensor]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

def ensure_array(arrayable: Arrayable, dtype) -> _Tensor:
    if isinstance(arrayable, _Tensor):
        return arrayable
    else:
        if dtype is None:
            arrayable = np.array(arrayable)
            if arrayable.dtype == np.float64:
                arrayable = arrayable.astype(np.float32)
            if arrayable.dtype == np.int64:
                arrayable = arrayable.astype(np.int32)
            return _Tensor(arrayable)
    return _Tensor(arrayable, dtype)

def ensure_dtype(*dtypes):
    if mstype.float32 in dtypes:
        return mstype.float32
    if mstype.int32 in dtypes:
        return mstype.int32
    return dtypes[0]

def ensure_matmul_candidates(t, dim):
    assert dim in [0, 1]
    shape = (1,) + tuple(t.shape) if dim == 0 else tuple(t.shape) + (1,)
    t_data = t if len(t.shape) == 2 else P.Reshape()(t, shape)
    return t_data

def matmul_postprocess(data, t1_ndim, t2_ndim):
    if t1_ndim == 1:
        data = P.Squeeze(0)(data)
    if t2_ndim == 1:
        data = P.Squeeze(1)(data)
    return data

class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None,
                 dtype = None
    ):
        self.data = ensure_array(data, dtype)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad : Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data.asnumpy()))

    def __repr__(self) -> str:
        return f"Tensor({self.data.asnumpy()}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """
        geys called if I do t + other
        """
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        """ gets called if I do other + t """
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        """
        when we do t += other
        """
        self.data = P.Add()(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __isub__(self, other) -> 'Tensor':
        self.data = P.Sub()(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __imul__(self, other) -> 'Tensor':
        self.data = P.Mul()(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, other)

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __sub__(self, other) -> 'Tensor':
        return _sub(self,ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def sum(self) -> 'Tensor':
        return tensor_sum(self)

    @property
    def ndim(self):
        return len(self.shape)

    def backward(self, grad: 'Tensor' = None):
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must specified for non-0-tensor")

        self.grad.data = P.Add()(self.grad.data, grad.data)
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

def tensor_sum(t: Tensor) -> Tensor:
    data_dtype = ensure_dtype(t.data.dtype)
    data = P.ReduceSum()(P.Cast()(t.data, mstype.float32))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: _Tensor) -> _Tensor:
            return grad * P.OnesLike()(t.data)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(P.Cast()(data, data_dtype), requires_grad, depends_on)

def _add(t1: Tensor, t2:Tensor) -> Tensor:
    data = P.Add()(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: _Tensor) -> _Tensor:
            # Idea: [1,2,3] + [4,5,6] => [5,7,9]
            # Handle the broadcasting properly
            # Sum out added dims
            grad_dtype = ensure_dtype(grad.dtype)
            grad = P.Cast()(grad, mstype.float32)
            ndims_added = len(grad.shape) - t1.ndim
            for _ in range(ndims_added):
                grad = P.ReduceSum()(grad, axis=0)

            # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = P.ReduceSum(True)(grad, axis=i)

            grad = P.Cast()(grad, grad_dtype)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: _Tensor) -> _Tensor:
            grad_dtype = ensure_dtype(grad.dtype)
            grad = P.Cast()(grad, mstype.float32)
            ndims_added = len(grad.shape) - t2.ndim
            for _ in range(ndims_added):
                grad = P.ReduceSum()(grad, axis=0)
             # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = P.ReduceSum(True)(grad, axis=i)

            grad = P.Cast()(grad, grad_dtype)
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
        requires_grad,
        depends_on
    )

def _mul(t1: Tensor, t2:Tensor) -> Tensor:
    """
    y = (a + eps) * b = a * b + (eps * b * dL/dy)
    gradient_y = 5
    have dL/dy
    dL/da = dL/dy * dy/da(b)
    """
    data = P.Mul()(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: _Tensor) -> _Tensor:
            grad_dtype = ensure_dtype(grad.dtype, t2.data.dtype)
            grad = P.Cast()(grad, mstype.float32)
            grad = P.Mul()(grad, t2.data)

            ndims_added = len(grad.shape) - t1.ndim
            for _ in range(ndims_added):
                grad = P.ReduceSum()(grad, axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = P.ReduceSum(True)(grad, axis=i)
            grad = P.Cast()(grad, grad_dtype)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: _Tensor) -> _Tensor:
            grad_dtype = ensure_dtype(grad.dtype, t1.data.dtype)
            grad = P.Cast()(grad, mstype.float32)
            grad = P.Mul()(grad, t1.data)
            ndims_added = len(grad.shape) - t2.ndim
            for _ in range(ndims_added):
                grad = P.ReduceSum()(grad, axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = P.ReduceSum(True)(grad, axis=i)
            grad = P.Cast()(grad, grad_dtype)
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
        requires_grad,
        depends_on
    )

def _neg(t: Tensor) -> Tensor:
    data = P.Neg()(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: P.Neg()(x))]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2

def _matmul(t1: Tensor, t2:Tensor) -> Tensor:
    """
    if t1 is (n1,m1) t2 is (m1,m2) then t1 @ t2 is (n1,m2)
    so grad3 is (n1,m2)
    if t3 = t1 @ t2 and grad3 is the gradient of some function wrt t3, then
        grad1 = grad @ t2.T
        grad2 = t1.T @ grad
    """
    assert len(t1.data.shape) <= 2, f"Tensor {t1} should not be greater 2 dims, but get {len(t1.data.shape)}"
    assert len(t2.data.shape) <= 2, f"Tensor {t2} should not be greater 2 dims, but get {len(t2.data.shape)}"
    data_dtype = ensure_dtype(t1.data.dtype, t2.data.dtype)
    t1_data = ensure_matmul_candidates(t1.data, 0)
    t2_data = ensure_matmul_candidates(t2.data, 1)
    data = P.MatMul()(P.Cast()(t1_data, mstype.float32), P.Cast()(t2_data, mstype.float32))
    data = matmul_postprocess(data, t1.ndim, t2.ndim)
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: _Tensor) -> _Tensor:
            grad_dtype = ensure_dtype(grad.dtype, t2.data.dtype)
            perm = tuple(range(t2.ndim - 1, -1, -1))
            t2_T = P.Transpose()(P.Cast()(t2.data, mstype.float32), perm)
            grad_ndim, t2_T_ndim = len(grad.shape), len(t2_T.shape)
            grad = ensure_matmul_candidates(grad, 0)
            t2_T = ensure_matmul_candidates(t2_T, 1)
            grad = P.MatMul()(P.Cast()(grad, mstype.float32), t2_T)
            grad = matmul_postprocess(grad, grad_ndim, t2_T_ndim)
            return P.Cast()(grad, grad_dtype)
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: _Tensor) -> _Tensor:
            grad_dtype = ensure_dtype(grad.dtype, t1.data.dtype)
            perm = tuple(range(t1.ndim - 1, -1, -1))
            t1_T = P.Transpose()(P.Cast()(t1.data, mstype.float32), perm)
            grad_ndim, t2_T_ndim = len(grad.shape), len(t1_T.shape)
            grad = ensure_matmul_candidates(grad, 1)
            t1_T = ensure_matmul_candidates(t1_T, 0)
            grad = P.MatMul()(t1_T, P.Cast()(grad, mstype.float32))
            grad = matmul_postprocess(grad, t2_T_ndim, grad_ndim)
            return P.Cast()(grad, grad_dtype)
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(P.Cast()(data, data_dtype),
        requires_grad,
        depends_on
    )

def _slice(t: Tensor, idx) -> Tensor:
    """
    t2 = t1[3:4,4:4]
    """
    data = _tensor_getitem(t.data, idx)
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: _Tensor) -> _Tensor:
            bigger_grad = P.ZerosLike()(data)
            # bigger_grad = _tensor_getitem(bigger_grad, idx, grad)
            return bigger_grad
        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data,requires_grad,depends_on)
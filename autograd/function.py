import mindspore.ops.operations as P
from autograd.tensor import Tensor, Dependency
from mindspore._c_expression import Tensor as _Tensor

def tanh(tensor: Tensor) -> Tensor:
    '''
    tanh = 
    '''
    data = P.Tanh()(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: _Tensor) -> _Tensor:
            return grad * (1 - data * data)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)
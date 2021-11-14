from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE)

from .tensor import Tensor
from .parameter import Parameter
from .optim import Optimizer, SGD
from .module import Module
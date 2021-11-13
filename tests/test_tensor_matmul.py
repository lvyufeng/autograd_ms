import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorMul(unittest.TestCase):
    def test_simple_matmul(self):
        t1 = Tensor([[1,2],[3,4],[5,6]],requires_grad=True)
        t2 = Tensor([[10],[20]],requires_grad=True)

        t3 = t1 @ t2

        assert t3.data.asnumpy().tolist() == [[50],[110],[170]]
        grad = Tensor([[-1],[-2],[-3]])
        t3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data.asnumpy(), grad.data.asnumpy() @ t2.data.asnumpy().T)
        np.testing.assert_array_equal(t2.grad.data.asnumpy(), t1.data.asnumpy().T @ grad.data.asnumpy())
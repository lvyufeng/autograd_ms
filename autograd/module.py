import inspect
from typing import Iterator
from autograd.parameter import Parameter

class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    
    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        result = self.forward(*input,**kwargs)
        return result
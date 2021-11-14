"""
Optimizers go there
"""

class Optimizer:
    def __init__(self, parameters):
        self.parameters = []
        self.add_param_group(parameters)
        
    def step(self):
        raise NotImplementedError

    def add_param_group(self, param_group):
        for group in param_group:
            self.parameters.append(group)

class SGD(Optimizer):
    def __init__(self, parameters, lr:float = 0.01) -> None:
        self.lr = lr
        super(SGD,self).__init__(parameters)
    def step(self) -> None:
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr
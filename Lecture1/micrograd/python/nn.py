import random
from micrograd.python.engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters():
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

class Layer(Module):
    pass

class MLP(Module):
    pass
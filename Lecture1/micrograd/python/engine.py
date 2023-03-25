
from math import tanh

class Value:
    """
    Store a single scalar value and its gradient
    """
    def __init__(self, data, _children=(), _op='') -> None:
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # The op that produced this node, for visualization / debugging etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f"^{other}")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        out = Value(tanh(self.data), (self, ), 'tanh')

        def _backward():
            self.grad += (1-(tanh(self.data))**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topological(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topological(child)
                topo.append(v)
        build_topological(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other
    
    def __sub__(self, other): # other - self
        return self + (-other)
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
            return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    

from abc import ABC

class BaseLayer(ABC):

    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass
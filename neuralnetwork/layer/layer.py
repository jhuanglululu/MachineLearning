class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError()

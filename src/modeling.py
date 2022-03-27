from typing import List
import numpy as np



def sigmoid(X: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-X))

def d_sigmoid(X: np.ndarray) -> np.ndarray:
    return X * (1-X)

def relu(X: np.ndarray) -> np.ndarray:
    return (np.abs(X) + X)/2

def d_relu(X: np.ndarray) -> np.ndarray:
    X[X>0] = 1
    X[X<=0] = 0
    return X




class BaseLayer(object):
    def __init__(self):
        self.parameters = {}
        self.gradients = {}

    @property
    def params(self):
        return self.parameters
    
    @property
    def grads(self):
        return self.gradients
    
    def forward(self, X):
        raise NotImplementedError()

    def backward(self, post_derivative):
        raise NotImplementedError()

scale = 1

class MLQPLayer(BaseLayer):
    def __init__(self, input_size, output_size, activation="sigmoid"):
        super().__init__()
        # u.shape = (o, i)
        self.parameters['u'] = np.random.randn(output_size, input_size)
        self.parameters['u'] /= np.prod(self.parameters['u'].shape)
        # v.shape = (o, i)
        self.parameters['v'] = np.random.randn(output_size, input_size)
        self.parameters['v'] /= np.prod(self.parameters['v'].shape)
        # b.shape = (o, 1)
        self.parameters['b'] = np.random.randn(output_size, 1)  # Batch size = 1 here
        self.parameters['b'] /= np.prod(self.parameters['b'].shape)
        # self.parameters['b'] = np.zeros([output_size, 1])


        self.gradients['u'] = np.zeros_like(self.parameters['u'])
        self.gradients['v'] = np.zeros_like(self.parameters['v'])
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        self._input = None
        self._output = None
        self._activation = activation
        self._prev_derivative = None
        if self._activation not in ["sigmoid", "relu"]:
            raise ValueError("Available activation functions: 'sigmoid', 'relu'.")


    def forward(self, X):
        # u.shape = (o, i)
        # v.shape = (o, i)
        # b.shape = (o, 1)
        # self.input.shape = (1, i)
        self._input = X
        # self.output.shape = (1, o)
        self._output= np.dot(self.parameters['u'], np.transpose(self._input**2)) + np.dot(self.parameters['v'], np.transpose(self._input)) + self.parameters['b']

        if self._activation == "sigmoid":
            self._output = np.transpose(sigmoid(self._output))
        if self._activation == "relu":
            self._output = np.transpose(relu(self._output))
        return self._output
    
    
    def backward(self, post_derivative):
        # post_derivative.shape = (1, o)
        # d_f.shape = (1, h)
        if self._activation == "sigmoid":
            d_f = d_sigmoid(self._output)
        if self._activation == "relu":
            d_f = d_relu(self._output)

        # ((batch_size, h) * (batch_size, h)).T @ (1, i) -> (h, i)
        self.gradients['u'] = np.dot(np.transpose(d_f*post_derivative), self._input**2)
        self.gradients['v'] = np.dot(np.transpose(d_f*post_derivative), self._input)
        self.gradients['b'] = np.transpose(d_f*post_derivative)
        self._prev_derivative = np.dot(d_f*post_derivative, (2*self.parameters['u']@np.diag(self._input[0, :]) + self.parameters['v']))

        return self._prev_derivative 




class LinearLayer(BaseLayer):
    def __init__(self, input_size, output_size, activation="sigmoid"):
        super().__init__()
        # w.shape = (o, i) output_dim = 1 here
        self.parameters['w'] = np.random.randn(output_size, input_size)
        self.parameters['w'] /= np.prod(self.parameters['w'].shape)
        # b.shape = (o, 1)
        self.parameters['b'] = np.random.randn(output_size, 1)
        self.parameters['b'] /= np.prod(self.parameters['b'].shape)

        self.gradients['w'] = np.zeros_like(self.parameters['w'])
        self.gradients['b'] = np.zeros_like(self.parameters['b'])

        self._input = None
        self._output = None
        self._activation = activation
        self._prev_derivative = None

        if self._activation not in ["sigmoid", "relu"]:
            raise ValueError("Available activation functions: 'sigmoid', 'relu'.")

    def forward(self, X):
        # w.shape = (o, i) output_dim = 1 here
        # b.shape = (o, 1)
        # self._input.shape = (1, h)
        self._input = X
        # self._output.shape = (1, o)
        self._output = np.dot(self.parameters['w'], np.transpose(self._input)) + self.parameters['b']
        if self._activation == "sigmoid":
            self._output = np.transpose(sigmoid(self._output))
        if self._activation == "relu":
            self._output = np.transpose(relu(self._output_temp))

        return self._output


    def backward(self, post_derivative):
        # post_d.shape = (1, o)
        # d_f.shape = (1, o)
        if self._activation == "sigmoid":
            d_f = d_sigmoid(self._output)
        if self._activation == "relu":
            d_f = d_relu(self._output)

        d_f = self._output * (1-self._output)
        self.gradients['w'] = np.dot(np.transpose(d_f*post_derivative), self._input)
        self.gradients['b'] = np.transpose(d_f*post_derivative)
        # prev_d.shape = (1, h)
        self._prev_derivative = np.dot(d_f*post_derivative, self.parameters['w'])

        return self._prev_derivative 




class BaseModel(object):
    def __init__(self) -> None:
        self._output = None
        self.layers = []

    def forward(self, X):
        raise NotImplementedError()

    def backward(self, derivative):
        raise NotImplementedError()




class OneLayerMLQPwithLinearOut(BaseModel):
    def __init__(self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        activation: str="sigmoid"
        ) -> None:

        super().__init__()
        self.hidden_layer = MLQPLayer(input_size=input_size, output_size=hidden_size, activation=activation)
        self.output_layer = LinearLayer(input_size=hidden_size, output_size=output_size, activation="sigmoid")
        self.layers = [self.output_layer, self.hidden_layer]
        

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._output = self.hidden_layer.forward(X)
        self._output = self.output_layer.forward(self._output)
        return self._output

    def backward(self, derivative: np.ndarray) -> np.ndarray:
        derivative = self.output_layer.backward(post_derivative=derivative)
        derivative = self.hidden_layer.backward(post_derivative=derivative)
        return 

class TwoLayerMLQP(BaseModel):
    def __init__(self, 
        input_size: int, 
        hidden_size: int,
        output_size: int, 
        activation: str="sigmoid") -> None:

        super().__init__()
        self.mlqp_layer1 = MLQPLayer(input_size=input_size, output_size=hidden_size, activation=activation)
        self.mlqp_layer2 = MLQPLayer(input_size=hidden_size, output_size=output_size, activation=activation)
        self.layers = [self.mlqp_layer1, self.mlqp_layer2]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self._output = self.mlqp_layer1.forward(X)
        self._output = self.mlqp_layer2.forward(self._output)
        return self._output

    def backward(self, derivative: np.ndarray):
        derivative = self.mlqp_layer2.backward(derivative)
        derivative = self.mlqp_layer1.backward(derivative)
        return

class ThreeLayerMLQP(BaseModel):
    def __init__(self,
        input_size: int,
        h1: int, # size of hidden layer 1
        h2: int, # size of hidden layer 2  
        output_size: int,
        activation: str="sigmoid",
    ) -> None:
        super().__init__()
        self.mlqp_layer1 = MLQPLayer(input_size=input_size, output_size=h1, activation=activation)
        self.mlqp_layer2 = MLQPLayer(input_size=h1, output_size=h2, activation=activation)
        self.mlqp_layer3 = MLQPLayer(input_size=h2, output_size=output_size, activation="sigmoid")
        self.layers = [self.mlqp_layer1, self.mlqp_layer2, self.mlqp_layer3]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self._output = self.mlqp_layer1.forward(X)
        self._output = self.mlqp_layer2.forward(self._output)
        self._output = self.mlqp_layer3.forward(self._output)
        return self._output

    def backward(self, derivative: np.ndarray):
        derivative = self.mlqp_layer3.backward(derivative)
        derivative = self.mlqp_layer2.backward(derivative)
        derivative = self.mlqp_layer1.backward(derivative)
        return 


class TwoLayerMLQPwithLinearOut(BaseModel):
    def __init__(self,
        input_size: int,
        h1: int, # size of hidden layer 1
        h2: int, # size of hidden layer 2  
        output_size: int,
        activation: str="sigmoid",
    ) -> None:
        super().__init__()
        self.mlqp_layer1 = MLQPLayer(input_size=input_size, output_size=h1, activation=activation)
        self.mlqp_layer2 = MLQPLayer(input_size=h1, output_size=h2, activation=activation)
        self.linear_layer = LinearLayer(input_size=h2, output_size=output_size, activation="sigmoid")
        self.layers = [self.mlqp_layer1, self.mlqp_layer2, self.linear_layer]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self._output = self.mlqp_layer1.forward(X)
        self._output = self.mlqp_layer2.forward(self._output)
        self._output = self.linear_layer.forward(self._output)
        return self._output

    def backward(self, derivative: np.ndarray):
        derivative = self.linear_layer.backward(derivative)
        derivative = self.mlqp_layer2.backward(derivative)
        derivative = self.mlqp_layer1.backward(derivative)
        return 











        
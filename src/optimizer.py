from modeling import BaseModel

class Optimizer(object):
    def __init__(self, lr=1e-4):
        self.lr = lr
    
    def set_lr(self, lr):
        # Reset learning rate
        self.lr = lr
        
    def optimize(self, model: BaseModel):
        for layer in model.layers:
            for key, _ in layer.params.items():
                layer.params[key] -= (self.lr * layer.grads[key])




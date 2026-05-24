from torch import device

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, device: device = device("cpu")):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.device = device
    
    def __call__(self, best_loss: float, model):
        if self.best_loss is None:
            self.best_loss = best_loss
            self.best_model_state = model.state_dict()
        elif best_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = best_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

    def get_best_model(self, model):
        model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        return model
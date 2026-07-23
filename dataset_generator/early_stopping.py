from torch import device

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, device: device = device("cpu")):
        self.patience = patience
        self.delta = delta
        self.best_f2_score = None
        self.best_tiebreak_metric = None
        self.early_stop = False
        self.counter = 0
        self.device = device
        self.best_model_state = None

    def __call__(self, f2_score: float, tiebreak_metric: float, model):
        if self.best_f2_score is None:
            self.best_f2_score = f2_score
            self.best_tiebreak_metric = tiebreak_metric
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return

        if f2_score > self.best_f2_score + self.delta:
            self.best_f2_score = f2_score
            self.best_tiebreak_metric = tiebreak_metric
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        elif f2_score >= self.best_f2_score - self.delta and tiebreak_metric > self.best_tiebreak_metric + self.delta:
            # PR-AUC tied the current best (within delta): prefer the checkpoint
            # with the higher tiebreak metric among equally-good-by-PR-AUC epochs.
            self.best_tiebreak_metric = tiebreak_metric
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter += 1  # Don't reset counter here, as this should only be a tiebreaker, not a new best epoch.
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def get_best_model(self, model):
        model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        return model
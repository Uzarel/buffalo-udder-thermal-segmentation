class EarlyStopper:
    """
    Checks if for a number of epochs the validation loss does not get better.

    Args:
        patience (int): number of epochs for which the validation loss does not get better
    """
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

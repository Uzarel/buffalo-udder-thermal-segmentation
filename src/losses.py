from torch.nn.modules.loss import _Loss


class WeightedLoss(_Loss):
    """
    Weight loss function by a fixed factor.

    Args:
        loss (callable): function to apply weighting
        weight (float): fixed factor for weighting loss
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Join multiple loss functions into one by means of fixed factors.

    Args:
        losses (list): list of loss functions to join
        weights (list): list of fixed factors for weighting losses
    """

    def __init__(self, losses, weights=None):
        super().__init__()
        self.__name__ = "joint_loss"

        if weights is None:
            weights = [1.0] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError(
                    "Number of weights must be equal to the number of losses"
                )
        self.losses = [
            WeightedLoss(loss, weight) for loss, weight in zip(losses, weights)
        ]

    def forward(self, *input):
        joint_loss = 0.0
        for weighted_loss in self.losses:
            joint_loss += weighted_loss(*input)
        return joint_loss

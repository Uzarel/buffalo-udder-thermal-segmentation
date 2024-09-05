from typing import Optional, List
from functools import partial

from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as MetricF
from segmentation_models_pytorch.base.modules import Activation

from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from segmentation_models_pytorch.losses._functional import focal_loss_with_logits, soft_dice_score, to_tensor

from torch import Tensor, long, log
import torch.nn.functional as F

class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return MetricF.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(base.Metric):
    def __init__(
        self,
        beta=1,
        eps=1e-7,
        threshold=0.5,
        activation=None,
        ignore_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return MetricF.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):
    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return MetricF.accuracy(
            y_pr, y_gt, threshold=self.threshold, ignore_channels=self.ignore_channels
        )


class Recall(base.Metric):
    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return MetricF.recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):
    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return MetricF.precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class SoftBCELossMetric(base.Metric):
    __name__ = "soft_bce_loss"
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[Tensor] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            y_pred: Tensor of shape (N, C, H, W)
            y_true: Tensor of shape (N, H, W) or (N, 1, H, W)

        Returns:
            metric: Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (
                1 - self.smooth_factor
            )
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        # Return the mean or sum of loss as the metric
        if self.reduction == "mean":
            metric = loss.mean()
        elif self.reduction == "sum":
            metric = loss.sum()
        else:
            metric = loss

        return metric


class FocalLossMetric(base.Metric):
    __name__ = "focal_loss"
    
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        **kwargs
    ):
        """Compute Focal loss as a metric

        Args:
            mode: Loss mode 'binary', 'multiclass', or 'multilabel'
            alpha: Prior probability of having positive value in the target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
            normalized: Compute normalized focal loss.
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode, you
                should use `reduction="sum"`.
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(**kwargs)

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculate the focal loss as a metric.

        Args:
            y_pred: Tensor of shape (N, C, H, W)
            y_true: Tensor of shape (N, H, W) or (N, C, H, W)

        Returns:
            metric: The focal loss metric
        """
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            metric = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            metric = 0

            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                metric += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return metric


class DiceLossMetric(base.Metric):
    __name__ = "dice_loss"
    
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        **kwargs
    ):
        """Dice metric for image segmentation tasks.

        Args:
            mode: Metric mode 'binary', 'multiclass', or 'multilabel'
            classes:  List of classes that contribute in the metric computation.
            log_loss: If True, metric computed as `- log(dice_coeff)`, otherwise `dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            ignore_index: Label that indicates ignored pixels (does not contribute to metric)
            eps: Small epsilon for numerical stability to avoid zero division error
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(**kwargs)

        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=long)

        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculate the Dice score as a metric.

        Args:
            y_pred: Tensor of shape (N, C, H, W)
            y_true: Tensor of shape (N, H, W) or (N, C, H, W)

        Returns:
            metric: The Dice score metric
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(long), num_classes
                ).permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            metric = -log(scores.clamp_min(self.eps))
        else:
            metric = 1. - scores

        mask = y_true.sum(dims) > 0
        metric *= mask.to(metric.dtype)

        if self.classes is not None:
            metric = metric[self.classes]

        return self.aggregate_metric(metric)

    def aggregate_metric(self, metric):
        return metric.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

from typing import Any, Optional
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn.pool import knn
from torch_geometric.data import Batch
from torchmetrics import MaxMetric
from lidar_multiclass.models.modules.randla_net import RandLANet
from lidar_multiclass.models.modules.point_net import PointNet
from lidar_multiclass.utils import utils

log = utils.get_logger(__name__)


class Model(LightningModule):
    """
    A LightningModule organizesm your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        neural_net_class = self.get_neural_net_class(self.hparams.neural_net_class_name)
        self.model = neural_net_class(self.hparams.neural_net_hparams)

        self.softmax = nn.Softmax(dim=1)

    def setup(self, stage: Optional[str]):
        """
        Setup metrics as needed.
        Nota : stage "validate" should be included in "fit" stage for our usage.
        Setup criterion (loss) except in predict mode.
        """
        if stage == "fit":
            self.train_iou = self.hparams.iou()
            self.val_iou = self.hparams.iou()
            self.val_iou_best = MaxMetric()
        if stage == "test":
            self.test_iou = self.hparams.iou()
        if stage != "predict":
            self.criterion = self.hparams.criterion

    def forward(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch)
        return logits

    def step(self, batch: Any):
        logits = self.forward(batch)
        targets = batch.y
        loss = self.criterion(logits, targets)
        with torch.no_grad():
            proba = self.softmax(logits)
            preds = torch.argmax(logits, dim=1)
        return loss, logits, proba, preds, targets

    def on_fit_start(self) -> None:
        self.experiment = self.logger.experiment[0]

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.train_iou(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True
        )
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
        }

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        if self.hparams.classification_smoothing_k_nn:
            preds = self.smooth_preds(batch, preds)
        self.val_iou(preds, targets)
        self.log("val/iou", self.val_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def validation_epoch_end(self, outputs):
        iou = self.val_iou.compute()
        self.val_iou_best.update(iou)
        self.log(
            "val/iou_best", self.val_iou_best.compute(), on_epoch=True, prog_bar=True
        )
        # self.val_iou.reset()  # in case of `num_sanity_val_steps` in trainer

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        if self.hparams.classification_smoothing_k_nn:
            preds = self.smooth_preds(batch, preds)
        self.test_iou(preds, targets)
        self.log("test/iou", self.test_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def predict_step(self, batch: Any):
        logits = self.forward(batch)
        proba = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)
        if self.hparams.classification_smoothing_k_nn:
            preds = self.smooth_preds(batch, preds)
        return {"batch": batch, "proba": proba, "preds": preds}

    def smooth_preds(self, batch, preds):
        """KNN consensus smoothing for preds, except for class "unclassified"."""
        k = self.hparams.classification_smoothing_k_nn

        preds = preds.clone()
        assign_idx = knn(
            batch.pos, batch.pos, k, batch_x=batch.batch_x, batch_y=batch.batch_x
        )
        knn_preds = preds[assign_idx[1]].view(-1, k)
        modes, _ = knn_preds.mode(dim=1)
        mode_freq = (knn_preds == modes.unsqueeze(1)).sum(dim=1) / k
        majority_mask = mode_freq >= 0.5
        classified_points_mask = preds > 1
        mask = classified_points_mask * majority_mask
        preds[mask] = modes[mask]
        return preds

    def get_neural_net_class(self, class_name):
        """Access class of neural net based on class name."""
        for neural_net_class in [PointNet, RandLANet]:
            if class_name in neural_net_class.__name__:
                return neural_net_class
        raise KeyError(f"Unknown class name {class_name}")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.lr)
        try:
            lr_scheduler = self.hparams.lr_scheduler(optimizer)
        except:
            # OneCycleLR needs optimizer and max_lr
            lr_scheduler = self.hparams.lr_scheduler(optimizer, self.lr)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": self.hparams.monitor,
        }

        return config

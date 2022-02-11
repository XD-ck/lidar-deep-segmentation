from typing import Any, Optional
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.data import Batch
from torchmetrics import MaxMetric
from tqdm import tqdm
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
        self.val_iou(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/iou", self.val_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
            "entropy": None,
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
        self.test_iou(preds, targets)
        self.log("test/iou", self.test_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
            "entropy": None,
        }

    def predict_step(self, batch: Any):
        logits = self.forward(batch)
        proba = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)
        entropy = Categorical(probs=proba).entropy()
        return {"batch": batch, "proba": proba, "preds": preds, "entropy": entropy}

    def stochastic_forward_fc_end(self):
        self.model.fc_end[-2] = self.model.fc_end[-2].train()
        logits = self.model.fc_end(self.model.pre_fc_end_x)
        # TODO: abstract this reshape of scores
        logits = logits.squeeze(-1)  # B, C, N
        logits = torch.cat(
            [score_cloud.permute(1, 0) for score_cloud in logits]
        )  # B*N, C
        return logits

    def monte_carlo_predict_step(
        self, batch: torch_geometric.data.Data, n_stochastic_pass: int = 25
    ):
        outputs = self.predict_step(batch)

        logits = []
        for _ in tqdm(range(n_stochastic_pass)):
            logits += [self.stochastic_forward_fc_end()]
        concats = torch.cat([s.unsqueeze(0) for s in logits])
        logits_std = torch.std(concats, dim=0)
        logits_sum = torch.sum(concats, dim=0)
        probas = self.softmax(logits_sum)
        preds = torch.argmax(logits_sum, dim=1)
        logits_std = logits_std[range(preds.shape[0]), preds]
        # TODO: Other uncertainty indicators could be used like max relative value of logits / proba ?
        # TODO: Using average logits should not change entropy here, but test it !
        entropy = Categorical(logits=logits_sum).entropy()

        outputs.update(
            {
                "proba": probas,
                "preds": preds,
                "entropy": entropy,
                "mc_dropout": logits_std,
            }
        )
        return outputs

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
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

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

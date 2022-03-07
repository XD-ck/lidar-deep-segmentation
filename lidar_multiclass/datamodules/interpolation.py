import os
from typing import Dict, List

import laspy
import numpy as np
import torch
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from torch.distributions import Categorical

from lidar_multiclass.utils import utils
from lidar_multiclass.utils import utils

from lidar_multiclass.datamodules.transforms import ChannelNames

log = utils.get_logger(__name__)


class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        output_dir: str,
        classification_dict: Dict[int, str],
        names_of_probas_to_save: List[str] = [],
        test_time_augmentation: bool = False,
        PredictedClassification="PredictedClassification",
        ProbasEntropy="entropy",
    ):

        os.makedirs(output_dir, exist_ok=True)
        self.preds_dirpath = output_dir
        self.current_las_filepath = ""
        self.classification_dict = classification_dict
        self.names_of_probas_to_save = names_of_probas_to_save
        self.test_time_augmentation = test_time_augmentation
        self.PredictedClassification = PredictedClassification
        self.ProbasEntropy = ProbasEntropy

        self.softmax = torch.nn.Softmax(dim=1)

        self.reverse_classification_mapper = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }

        self.index_of_probas_to_save = [
            list(classification_dict.values()).index(name)
            for name in names_of_probas_to_save
        ]

    @torch.no_grad()
    def update_with_inference_outputs(self, outputs: dict):
        """
        Save the predicted classes in las format with position.
        Handle las loading when necessary.

        :param outputs: outputs of a step.
        """
        batch = outputs["batch"].detach()
        batch_logits = outputs["logits"][:, self.index_of_probas_to_save].detach()
        for batch_idx, las_filepath in enumerate(batch.las_filepath):
            is_a_new_tile = las_filepath != self.current_las_filepath
            if is_a_new_tile:
                close_previous_las_first = self.current_las_filepath != ""
                if close_previous_las_first:
                    self.interpolate_and_save()
                self._load_las_for_classification_update(las_filepath)
            idx_x = batch.batch_x == batch_idx
            self.logits_u_sub.append(batch_logits[idx_x])
            self.pos_u_sub.append(batch.pos_copy_subsampled[idx_x])
            idx_y = batch.batch_y == batch_idx
            self.pos_u.append(batch.pos_copy[idx_y])

    def _load_las_for_classification_update(self, filepath):
        """Load a LAS and add necessary extradim."""

        self.las = laspy.read(filepath)
        self.current_las_filepath = filepath

        coln = self.PredictedClassification
        param = laspy.ExtraBytesParams(name=coln, type=int)
        self.las.add_extra_dim(param)
        self.las[coln][:] = 0

        param = laspy.ExtraBytesParams(name=self.ProbasEntropy, type=float)
        self.las.add_extra_dim(param)
        self.las[self.ProbasEntropy][:] = 0.0

        for class_name in self.names_of_probas_to_save:
            param = laspy.ExtraBytesParams(name=class_name, type=float)
            self.las.add_extra_dim(param)
            self.las[class_name][:] = 0.0

        self.pos = torch.from_numpy(
            np.asarray(
                [
                    self.las.x,
                    self.las.y,
                    self.las.z,
                ],
                dtype=np.float32,
            ).transpose()
        )
        self.logits_u_sub = []
        self.pos_u_sub = []
        self.pos_u = []

    @torch.no_grad()
    def interpolate_and_save(self):
        """
        Interpolate all predicted probabilites to their original points in LAS file, and save.
        Returns the path of the updated, saved LAS file.
        """

        basename = os.path.basename(self.current_las_filepath)

        os.makedirs(self.preds_dirpath, exist_ok=True)
        self.output_path = os.path.join(self.preds_dirpath, basename)
        log.info(f"Updated LAS will be saved to {self.output_path}")

        # Cat
        self.pos_u = torch.cat(self.pos_u).cpu()
        self.pos_u_sub = torch.cat(self.pos_u_sub).cpu()
        self.logits_u_sub = torch.cat(self.logits_u_sub).cpu()

        if self.test_time_augmentation:
            # TODO: get a unique value by position for self.update_logits

            pass
        self.logits_u = knn_interpolate(
            self.logits_u_sub,
            self.pos_u_sub,
            self.pos,
            batch_x=None,
            batch_y=None,
            k=10,
            num_workers=4,
        )

        # Here create missing elements
        self.preds_u = torch.argmax(self.logits_u, dim=1)
        self.probas_u = self.softmax(self.logits_u)
        self.entropy_u = Categorical(probs=self.probas_u).entropy()
        # Remap predictions to good classification codes
        self.preds_u = np.vectorize(self.reverse_classification_mapper.get)(
            self.preds_u
        )
        self.preds_u = torch.from_numpy(self.preds_u)

        # ASSIGN
        for class_idx_in_tensor, class_name in enumerate(self.names_of_probas_to_save):
            self.las[class_name] = self.probas_u[:, class_idx_in_tensor]
        self.las[self.PredictedClassification] = self.preds_u
        if len(self.entropy_u):
            self.las[self.ProbasEntropy] = self.entropy_u

        log.info(f"Saving...")
        self.las.write(self.output_path)
        log.info(f"Saved.")

        # Clean-up - get rid of current data to go easy on memory
        self.current_las_filepath = ""
        del self.las
        del self.pos
        del self.pos_u_sub
        del self.pos_u
        del self.preds_u
        del self.entropy_u
        del self.probas_u
        del self.logits_u_sub

        return self.output_path

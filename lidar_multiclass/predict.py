import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm

from lidar_multiclass.utils import utils
from lidar_multiclass.datamodules.interpolation import Interpolator


log = utils.get_logger(__name__)


@utils.eval_time
def predict(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Those are the 3 needed inputs
    assert os.path.exists(config.predict.resume_from_checkpoint)
    assert os.path.exists(config.predict.src_las)

    torch.set_grad_enabled(False)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_all_transforms()
    datamodule._set_predict_data([config.predict.src_las])

    # TODO: pass as a predict.data_handler config for full parameterization outside this repo scope.
    data_handler = Interpolator(
        config.predict.output_dir,
        datamodule.dataset_description.get("classification_dict"),
        names_of_probas_to_save=config.predict.names_of_probas_to_save,
    )

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.predict.resume_from_checkpoint)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    for index, batch in tqdm(
        enumerate(datamodule.predict_dataloader()), desc="Infering probabilities..."
    ):
        batch.to(device)
        outputs = model.monte_carlo_predict_step(batch)
        data_handler.update_with_inference_outputs(outputs)

        if index >= 1:
            break  ###### TODO - this is for debugging purposes ###################

    updated_las_path = data_handler.interpolate_and_save("predict")
    log.info(f"Updated LAS saved to : {updated_las_path}")


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_multiclass.utils import utils
    from lidar_multiclass.train import train
    from lidar_multiclass.predict import predict

    # A couple of optional utilities:
    # - disabling python warnings
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    return predict(config)


if __name__ == "__main__":
    # cf. https://github.com/facebookresearch/hydra/issues/1283
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()

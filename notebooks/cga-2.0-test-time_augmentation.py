# get the config


import os
import os.path as osp
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    from lidar_multiclass.datamodules.interpolation import Interpolator
    from lidar_multiclass.utils import utils

    # Those are the 2 needed inputs, in addition to the hydra config.
    assert os.path.exists(config.predict.resume_from_checkpoint)
    assert os.path.exists(config.predict.src_las)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data([config.predict.src_las])

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.predict.resume_from_checkpoint)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    data_handler = Interpolator(
        config.predict.output_dir,
        datamodule.dataset_description.get("classification_dict"),
        names_of_probas_to_save=config.predict.names_of_probas_to_save,
    )

    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)
        outputs = model.predict_step(batch)
        data_handler.update_with_inference_outputs(outputs)

    updated_las_path = data_handler.interpolate_and_save()
    return updated_las_path


if __name__ == "__main__":
    # cf. https://github.com/facebookresearch/hydra/issues/1283
    sys.path.append(osp.dirname(osp.dirname(__file__)))
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()

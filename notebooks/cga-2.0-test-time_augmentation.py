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

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.predict.resume_from_checkpoint)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # METHOD 1
    config.predict.names_of_probas_to_save = [
        a + "_A" for a in config.predict.names_of_probas_to_save
    ]
    datamodule.dataset_description["classification_dict"] = {
        k: v + "_A"
        for k, v in datamodule.dataset_description.get("classification_dict").items()
    }
    datamodule._set_predict_data([config.predict.src_las])
    data_handler = Interpolator(
        config.predict.output_dir,
        datamodule.dataset_description.get("classification_dict"),
        names_of_probas_to_save=config.predict.names_of_probas_to_save,
        test_time_augmentation=False,
        PredictedClassification="preds_A",
        ProbasEntropy="entropy_A",
    )
    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)
        outputs = model.predict_step(batch)
        data_handler.update_with_inference_outputs(outputs)

    updated_las_path = data_handler.interpolate_and_save()

    # METHOD 2 - test_time_augmentation=True
    config.predict.names_of_probas_to_save = [
        a.replace("_A", "_B") for a in config.predict.names_of_probas_to_save
    ]
    datamodule.dataset_description["classification_dict"] = {
        k: v.replace("_A", "_B")
        for k, v in datamodule.dataset_description.get("classification_dict").items()
    }

    datamodule._set_predict_data(
        [updated_las_path], subtiles_overlap=config.subtiles_overlap
    )
    data_handler = Interpolator(
        config.predict.output_dir,
        datamodule.dataset_description.get("classification_dict"),
        names_of_probas_to_save=config.predict.names_of_probas_to_save,
        test_time_augmentation=True,
        PredictedClassification="preds_B",
        ProbasEntropy="entropy_B",
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

<div align="center">

# Semantic Segmentation - Inspection Module

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)
</div>
<br><br>

## Description
### Context
A fast and sensitive semantic segmentation of High Density Lidar data was performed with geometric rule-based algorithm to identify buildings. It yielded a high number of false positive. Around 160km² of Lidar data was thoroughly inspected to identify false positive and false negative. At larger scale, this kind of human inspection would be intractable.

### Objective
We train a semantic segmentation neural network to confirm or refute automatically the majority of "candidate buildings points" obtained from the rule-based algorithm, while also identifying cases of uncertainty for later human inspection. This results in an output point cloud in which only a fraction of the candidate building points remain to be inspected.

### Content

1) Training and evaluation of the model.
2) Prediction of point-level probabilities.
3) Validation module decision process:
    1) Clustering of candidate buildings points into candidate building groups
    2) Superposition of external vector database of Buildings.
    3) Decisions at the point-level based on probabilities : _confirmed_ or _refuted_ (as a building).
    3) Decision at the group-level:
        1) Confirmation: if proportion of _confirmed_ >= threshold OR if proportion of _overlayed_ >= threshold.
        2) Refutation: if proportion of _refuted_  >= threshold AND proportion of _overlayed_ < threshold
        3) Uncertainty: elsewise.
    4) Update of the point cloud based on those decisions.

4) Multiobjective hyperparameter optimization of the decision process (point-level and group-level thresholds) to maximize automation, precision, and recall.


## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/CharlesGaydon/Segmentation-Validation-Model
cd Segmentation-Validation-Model

# [OPTIONAL] If you want to use a gpu make sure cuda toolkit is installed
sudo apt install nvidia-cuda-toolkit

# install conda
https://www.anaconda.com/products/individual


# create conda environment (you may need to run lines manually as conda may not activate properly from bash script)
bash bash/setup_environment/setup_env.sh

# activate using
conda activate validation_module
```

Rename `.env_example` to `.env` and fill out `LOG PATH`, `DATAMODULE`, and `LOGGER` sections.

Train model with a specific experiment from [configs/experiment/](configs/experiment/)
```yaml
# default
python run.py experiment=PN_debug
```

Evaluate the model and get inference results on the validation dataset
```yaml
# to evaluate and infer at the same time
python run.py experiment=PN_validate trainer.resume_from_checkpoint=/path/to/checkpoints.ckpt fit_the_model=false test_the_model=true
# to log IoU without saving predictions to new LAS files 
python run.py experiment=PN_validate callbacks.save_preds.save_predictions=false trainer.resume_from_checkpoint=/path/to/checkpoints.ckpt fit_the_model=false test_the_model=true
```
To evaluate on test data instead of val data, replace `experiment=PN_validate` by `experiment=PN_test`.


Run a multi-objectives hyperparameters optimization of the decision thresholds, to maximize recall and precision directly while also maximizing automation.

```yaml
python run.py -m task=optimize optimize.todo='prepare+optimize+evaluate+update' optimize.predicted_las_dirpath="/path/to/val/las/folder/" optimize.results_output_dir="/path/to/save/updated/val/las/"  optimize.best_trial_pickle_path="/path/to/best_trial.pkl"
```

To evaluate best solution on test set, simply change the input las folder and the results output folder, and remove the optimization from the todo. The path to the best trial stays the same.

```yaml
python run.py task=optimize optimize.todo='prepare+evaluate+update' print_config=false optimize.predicted_las_dirpath="/path/to/test/las/folder/" optimize.results_output_dir="/path/to/save/updated/test/las/" optimize.best_trial_pickle_path="/path/to/best_trial.pkl"
```

Additionally, if you want to update the las classification based on those decisions, add an `optimize.update_las=true` argument.


Finally, to apply the module on unseen data, you will need 1. a checkpointed Model, 2. An hydra directory with the configurations used to establish the checkpointed Model, 3. a pickled Optuna "Best trial" with decision thresholds, and 4. a source LAS file. Then, run :

```yaml
python run.py --config-path /path/to/.hydra --config-name config.yaml task=predict hydra.run.dir=path/to/Segmentation-Validation-Model +prediction.resume_from_checkpoint=/path/to/checkpoints.ckpt +prediction.src_las=/path/to/input.las +prediction.best_trial_pickle_path=/path/to/best_trial.pkl prediction.output_dir=/path/to/save/updated/test/las/ datamodule.batch_size=16
```

Please note that "hydra.run.dir" is the directory of the project, it's not a mistake (loading a different config from .hydra with "--config-path" may change that path, we currently need that step to put everything back).

### If you're only interested in doing inference on a pre-existing model
If you're only interested in doing inference on a pre-existing model, you just need to:
  install the dependencies
  fill out "LOGS_DIR", "DATASET_DIR" and "INPUT_BD_TOPO_SHP_PATH" from the .env file.
  Prepare and run the command line for unseen data.
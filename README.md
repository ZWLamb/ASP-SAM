Authorized deployment of ASP-SAM
### Setup Enviorment
For our experiments, we utilized the Anaconda package manager and included an `environment.yml` to replicate our setup. To establish the environment, please execute the following command:
```console
conda env create -f env.yml
conda activate asp_sam
```

### Training
Before beginning model training, review config.py to configure the essential hyperparameters. To start the training, execute the command below:
```console
python train.py
```
### Model Evaluation
For evaluation, we are utilizing the following metrics: `Dice Score`, `Aggregated Jaccard Index (AJI)`, and `Panoptic Quality (PQ)`. To perform the model evaluation, use the command provided below:
```console
python verify.py
```
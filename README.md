### Setup Enviorment
For our experiments, we utilized the Anaconda package manager and included an `environment.yml` to replicate our setup. To establish the environment, please execute the following command:
```console
conda env create -f env.yml
conda activate asp_sam
```
### Prepare weight files
Download the weight file from [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), 
rename it to `sam_vit_h.pt`, and place it in the root directory.
```console
wget https://github.com/whai362/PVT/releases/download/v1.0/pvt_v2_b2.pth
```
put the file 'pvt_v2_b2.pth' in 'models/pretrained_pth/pvt/'

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
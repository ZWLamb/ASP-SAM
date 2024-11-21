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

### Some Result in paper
    Table 1:
    DSB     DICE: 0.93091324829536935,AJI: 0.756618987885321375,PQ:0.74384320482745632
    MoNuSeg DICE: 0.85703477632958764,AJI: 0.508903847576684113,PQ:0.50659374751029468
    GlaS    DICE: 0.91927013723857629,AJI: 0.736702857672964612,PQ:0.72918243766949212

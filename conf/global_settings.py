import os
from datetime import datetime

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total Training epoches
EPOCH = 200
step_size = 10
i = 1
MILESTONES = []
while i * 5 <= EPOCH:
    MILESTONES.append(i* step_size)
    i += 1

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 5









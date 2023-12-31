import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


#=======================================================================================#
###===================================================================================###

DATASET_DOWNLOAD_LINK = 'https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset'

###===================================================================================###
#=======================================================================================#

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = ''
VAL_DIR = ''
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
EPOCH = 101
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS = 3
L1_LAMBDA = 100
LOAD_MODEL = False
SAVE_MODEL = True
SAVE_PATH = 'trained_models'


transform_both = A.Compose(
    [
    A.Resize(width = IMAGE_SIZE , height = IMAGE_SIZE),
    
    ],
    additional_targets={"image0" : "image"}
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(0.5),
        A.ColorJitter( p = 0.1),
        A.Normalize(mean = [0.5]*CHANNELS , std = [0.5]*CHANNELS , max_pixex_values = 255.0),
        ToTensorV2(),

    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean = [0.5]*CHANNELS , std = [0.5]*CHANNELS , max_pixex_values = 255.0),
        ToTensorV2(),
    ]
)
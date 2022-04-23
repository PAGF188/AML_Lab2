import os
from random import randint
import torch

APP_NAME = '%s-%d' % ('lab2', randint(0, 100))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_DIR = ROOT_DIR + 'data/'
MODEL_SAVE_DIR = ROOT_DIR + 'save/'
LOG_DIR = ROOT_DIR + 'log/'
MODELS_DIR = ROOT_DIR + 'models/'
LOG_PATH = LOG_DIR + APP_NAME + '.log'
LOG_FORMAT = '%(asctime)-15s %(filename)s:%(funcName)s:[%(levelname)s] %(message)s'

MODEL1_PRETRAINED_AUGMENTATION = "pretrained_augmentation.pt"
MODEL2_PRETRAINED_NOT_AUGMENTATION = "pretrained_not_augmentation.pt"
MODEL3_SCRATCH_AUGMENTATION = "scratch_augmentation.pt"
MODEL4_SCRATCH_NOT_AUGMENTATION = "scratch_not_augmentation.pt"

MODEL_UNET = "unet.pt"

# MODELS CONFIGURATION
LR_DENSENET = 1e-4
DENSE_NET_INPUT_SIZE = (224, 224)
UNET_NET_INPUT_SIZE = (572, 572)

N_CLASES = 10
N_CLASES_UNET = N_CLASES + 1
BACTH_SIZE = 32
BACTH_SIZE2 = 1  # I run the code in my own GPU
NUM_EPOCHS = 3
CRITERION = torch.nn.CrossEntropyLoss()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

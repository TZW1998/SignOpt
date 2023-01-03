import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tensorboardX import SummaryWriter
from torch import autocast 
from torch.cuda.amp import GradScaler 

import numpy as np
import random
import copy

import argparse
import os, shutil
import time

from optimizers import *
from models import *


init_model = torch.load("init_model.pt")

init_param = dict(init_model.named_parameters())

after_model = torch.load("after_model.pt")

after_param = dict(after_model.named_parameters())

check_param_diff = {n:np.unique((( after_param[n].cpu().detach().numpy()) // 0.001).astype(int)) for n,p in init_param.items()}
print(np.unique(np.concatenate(list(check_param_diff.values()))).shape)
pass
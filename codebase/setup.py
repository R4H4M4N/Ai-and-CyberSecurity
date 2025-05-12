import argparse
import torch
from pathlib import Path
import numpy as np


# ------------------------------------------------------
# CIFAR10

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])       # means values of CIFAR10 training set
CIFAR10_STD = np.array([0.2471, 0.2435, 0.2616])        # std values of CIFAR10 training set

# class names corresponding to each label
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


CIFAR100_MEAN = np.array([0.5070758,  0.4865503,  0.44091913])
CIFAR100_STD = np.array([0.26733097, 0.25643396, 0.27614763])


GTSRB_MEAN = np.array([0.34169674, 0.31255573, 0.3215503])
GTSRB_STD = np.array([0.28010625, 0.26819786, 0.2746381])


def init_config():
    parser = argparse.ArgumentParser(description='CSIT375_975 experimental setup.')

    # Default paths need to be changed for colab.
    parser.add_argument('--out_dir', default="/home/wzong/My Passport/projects/CSIT375_975_labs_assignments", type=str, help='')
    parser.add_argument('--data_dir', default="/home/wzong/My Passport/projects/data", type=str, help='')

    cfg = parser.parse_args()

    # create Path objects from strings for easy use.
    cfg.out_dir = Path(cfg.out_dir)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.device = torch.device('cuda:0')  # use the first GPU

    return cfg









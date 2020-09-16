import torch
import numpy as np
import os, time
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer


class MotionPredictDataset(Dataset):
    def __init__(self,
                cfg,
                str_loader='train_data_loader',
                fn_rasterizer=build_rasterizer,
                test_mask_path=None):

        self.cfg = cfg
        self.str_loader = str_loader
        self.fn_rasterizer = fn_rasterizer
        self.test_mask_path = test_mask_path

        self.setup()

    def setup(self):

        self.dm = LocalDataManager(None)
        self.rasterizer = self.fn_rasterizer(self.cfg, self.dm)
        self.data_zarr = ChunkedDataset(self.dm.require(self.cfg[self.str_loader]["key"])).open(cached=False)

        if self.str_loader == 'test_data_loader':
            test_mask = np.load(self.test_mask_path)["arr_0"]
            self.ds = AgentDataset(self.cfg, self.data_zarr, self.rasterizer, agents_mask=test_mask)
        else:
            self.ds = AgentDataset(self.cfg, self.data_zarr, self.rasterizer)

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):

        # return 1000
        return len(self.ds)
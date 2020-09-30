import numpy as np
import os, time
# os.environ['KMP_DUPLICATE_LIB_OK']='False'

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18

from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from src.dataset import *
from src.model import *

## lightning
from src.effnet import EfficientNet

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.cuda.amp import GradScaler, autocast 

from tqdm import tqdm

import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
from l5kit.evaluation import write_pred_csv

from matplotlib import animation, rc 
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)

DIR_INPUT = "../../input/lyft-motion-prediction-autonomous-vehicles"
SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

DEBUG = False

cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    
    'raster_params': {
        'raster_size': [160, 160],  ## 300 300
        'pixel_size': [0.32, 0.32], ## 0.5 0.5
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 4,
    },

    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers':2,
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 500000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

test_dataset = MotionPredictDataset(cfg, str_loader="test_data_loader", test_mask_path = os.path.join(DIR_INPUT, 'scenes/mask.npz'))
print("test_dataset ok")
test_loader = DataLoader(test_dataset, batch_size=256, 
                            num_workers=8, shuffle=False)

# Referred https://www.kaggle.com/pestipeti/pytorch-baseline-inference
def run_prediction(predictor, data_loader):
    predictor.eval()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:

            ## batch
            inputs = data["image"].unsqueeze(2).to(device)
            target_availabilities = data["target_availabilities"].to(device)
            
            matrix = data["world_to_image"].to(device)
            centroid = data["centroid"].to(device)[:,None,:].to(torch.float)
            # target_availabilities = data["target_availabilities"].to(device)
            # targets = data["target_positions"].to(device)
            pred, confidences = predictor(inputs, [0],[0])
            
            ##
            bs, tl = inputs.shape[0], 50
            bias = torch.tensor([40.0, 80.0])[None,None,:].to(device)
            ##
            
            matrix_inv = torch.inverse(matrix)
            pred = pred + bias[:,None,:,:]
            pred = torch.cat([pred,torch.ones((bs,3,tl,1)).to(device)], dim=3)
            pred = torch.stack([torch.matmul(matrix_inv.to(torch.float), pred[:,i].transpose(1,2)) 
                                for i in range(3)], dim=1)
            pred = pred.transpose(2,3)[:,:,:,:2]
            pred = pred - centroid[:,None,:,:]

            pred_coords_list.append(pred.cpu().numpy().copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
            
            
            
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    return timestamps, track_ids, coords, confs


## load latest model
import torchvision

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


import torchvision
import torch_optimizer as optim
import torchvision
#import torch_optimizer as optim
from adamp import AdamP


class LyftModel(pl.LightningModule):
    def __init__(self, cfg: Dict, learning_rate = 5e-5, num_modes=3):
        super().__init__()

        #self.learning_rate = learning_rate
        ## c
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len


        # avg pool
        self.backbone = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        self.backbone.stem[0] = nn.Conv3d(
            25,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False)


        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(512, out_features=self.num_preds + num_modes)

    def init_weight():
        init_layer(self.logit)

    def forward(self, images, matrix, centroid, infer=False):

        x = self.backbone.stem(images)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.logit(x)

        # pred (bs)x(modes)x(time)x(2D coords)
        # confidences (bs)x(modes)
        bs, _ = x.shape
        tl = self.future_len

        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)

        if infer:
            matrix_inv = torch.inverse(matrix)
            pred = pred + bias[:,None,:,:]
            pred = torch.cat([pred,torch.ones((bs,3,tl,1)).to(device)], dim=3)
            pred = torch.stack([torch.matmul(matrix_inv.to(torch.float), pred[:,i].transpose(1,2)) 
                                for i in range(3)], dim=1)
            pred = pred.transpose(2,3)[:,:,:,:2]
            pred = pred - centroid[:,None,:,:]

        return pred, confidences

    def training_step(self, batch, batch_idx):
        data = batch

        ## batch
        inputs = data["image"].unsqueeze(2).to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)
        matrix = data["world_to_image"].to(device)
        centroid = data["centroid"].to(device)[:,None,:].to(torch.float)

        ## forward
        pred, confidences = self(inputs, matrix, centroid)


        ## fix
        bs,tl,_ = targets.shape
        assert tl == cfg["model_params"]["future_num_frames"]
        targets = targets + centroid
        targets = torch.cat([targets,torch.ones((bs,tl,1)).to(device)], dim=2)
        targets = torch.matmul(matrix.to(torch.float), targets.transpose(1,2))
        targets = targets.transpose(1,2)[:,:,:2]
        bias = torch.tensor([40.0, 80.0])[None,None,:].to(device)
        targets = targets - bias

        ## back
        loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result


    def configure_optimizers(self):
        ## lr find
        #optimizer = torch.optim.AdamW(self.parameters(), lr=(5e-5 or self.learning_rate))

        #optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = AdamP(self.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-2, nesterov=False)
        #optimizer = optim.Lamb(self.parameters(), lr= 1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, )
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=int(22000000/256), epochs=1)
        return [optimizer]#, [scheduler]


if __name__ == '__main__':

    device = torch.device("cuda:0")

    ckpt = torch.load("lightning_logs/version_4/checkpoints/N-Step-Checkpoint_3_57500.ckpt")
    model = LyftModel(cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)

    timestamps, track_ids, coords, confs = run_prediction(model, test_loader)

    csv_path = "sub3.csv"
    write_pred_csv(
        csv_path,
        timestamps=timestamps,
        track_ids=track_ids,
        coords=coords,
        confs=confs)
    print(f"Saved to {csv_path}")
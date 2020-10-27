##### import numpy as np
import os, sys, time
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer


sys.path.append('src')
from src.dataset import *

## lightning
# from effnet import EfficientNet

import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt
## START FOR VIS
import l5kit
from l5kit.visualization import *
from l5kit.geometry import *

# ===== GENERATE AND LOAD CHOPPED DATASET
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS


import imageio
from IPython.display import Video
from PIL import Image, ImageDraw

from pathlib import Path

DIR_INPUT = "../lyft-motion-prediction-autonomous-vehicles"
SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

DEBUG = False

cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 30,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [300, 300],  ## [384, 128]
        'pixel_size': [0.4, 0.4], ## 0.5 0.5
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 1,
    },
    
        'test_data_loader': {
    'key': 'scenes/test.zarr',
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 4,
},

    'val_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 8,
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 500000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}

# set env variable for data
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT


###
import torchvision
# from adamp import AdamP
from src.loss import *
# from linformer_pytorch import Linformer
# from axial_attention import AxialImageTransformer, AxialAttention

def init_layer(layer):
    nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

from efficientnet_pytorch import EfficientNet
class LyftModel(pl.LightningModule):
    def __init__(self, cfg: Dict, model_name="efficientnet-b1", num_modes=3):
        super().__init__()

        ## c
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        
        self.effnet0 = EfficientNet.from_pretrained(f'{model_name}', in_channels=65, num_classes=self.num_preds + num_modes,)

    def forward(self, images):
        ## road
        x = self.effnet0(images)
        # pred (bs)x(modes)x(time)x(2D coords)
        # confidences (bs)x(modes)
        bs, _ = x.shape
        tl = self.future_len

        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    def training_step(self, batch, batch_idx):
        data = batch
        inputs = batch["image"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)
        pred, confidences = self(inputs)
        ## back
        ## TODO: log learning rate
        loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def configure_optimizers(self):
        ## lr find
        #optimizer = torch.optim.AdamW(self.parameters(), lr=(5e-5 or self.learning_rate))

        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0003, weight_decay=0.)
        #optimizer = AdamP(self.parameters(), lr=0.00003, betas=(0.9, 0.999), weight_decay=0., nesterov=True)
        #optimizer = optim.Lamb(self.parameters(), lr= 1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, )
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=int(22000000/64), epochs=10)
        return [optimizer]#, [scheduler]


class ensemble(pl.LightningModule):
    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()
        ## c
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        
        m1 = LyftModel(cfg, model_name="efficientnet-b1")
        w1 = torch.load("ens_weights/600px_b1_2_47000.ckpt", map_location="cpu")
        m1.load_state_dict(w1['state_dict'])

        m2 = LyftModel(cfg, model_name="efficientnet-b7")
        w2 = torch.load("ens_weights/b7_2_42000.ckpt", map_location="cpu")
        m2.load_state_dict(w2['state_dict'])
        
        self.m1 = m1.effnet0
        self.m2 = m2.effnet0
        
        #self.head = nn.Sequential(nn.Linear(3840, 3840), nn.ReLU(), nn.Dropout(0.2), nn.Linear(3840, self.num_preds + num_modes))
        self.head = nn.Linear(606, self.num_preds + num_modes)
    def forward(self, images):
        with torch.no_grad():
            x1 = self.m1(images)
            x2 = self.m2(images)

        x = torch.cat([x1, x2], dim=1)
        x = self.head(x)
        bs, _ = x.shape
        tl = self.future_len

        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    def training_step(self, batch, batch_idx):
        data = batch
        inputs = batch["image"]#.to(device)
        target_availabilities = data["target_availabilities"]#.to(device)
        targets = data["target_positions"]#.to(device)
        pred, confidences = self(inputs)
        ## back
        ## TODO: log learning rate
        loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities)
        #result = pl.TrainResult(loss)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.)
        return [optimizer]#, [schedulers]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ===== INIT DATASET AND LOAD MASK

if __name__ == '__main__':
    # ==== EVAL LOOP
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', 
                    type=str, 
                    help='path to ckpt')

    parser.add_argument('--mode',

                        type=str,
                        help='if true, creates a submission')


    args = parser.parse_args()

    dm = LocalDataManager(None)
    num_frames_to_chop = 100
        
    val_mode = "sample"
    eval_cfg = cfg["val_data_loader"]
    eval_cfg['key'] = f'scenes/{val_mode}.zarr'

#     eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
#                                   num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)

    eval_base_path = f"../lyft-motion-prediction-autonomous-vehicles/scenes/{val_mode}_chopped_100"

    
    
    rasterizer = build_rasterizer(cfg, dm)

    eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    print(eval_mask_path)
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]

    if args.mode == "eval":
        print('here')
        eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
        data_loader = DataLoader(eval_dataset, batch_size=eval_cfg["batch_size"], 
                                    num_workers=eval_cfg["num_workers"], pin_memory=False)
        print(eval_dataset)
    elif args.mode == "sub":
        test_dataset = MotionPredictDataset(cfg, str_loader="test_data_loader", test_mask_path = os.path.join(DIR_INPUT, 'scenes/mask.npz'))
        print("test_dataset ok")
        data_loader = DataLoader(test_dataset, batch_size=eval_cfg["batch_size"], 
                                    num_workers=eval_cfg["num_workers"])
                                
    

    model = LyftModel(cfg).to(device)
    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['state_dict'])
    #model = nn.DataParallel(model)
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []

    progress_bar = tqdm(data_loader)
    pred_coords_list = []
    timestamps = []
    agent_ids = []
    confidences_list = []
    with torch.no_grad():
        for i, data in enumerate(progress_bar):
            inputs = data["image"].to(device)
            targets = data["target_positions"]
            
            pred, confidences = model(inputs)
            pred = pred.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []
            # convert into world coordinates and compute offsets
            for idx in range(len(pred)):
                for mode in range(3):
                    pred[idx, mode, :, :] = transform_points(pred[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]

            confidences_list.append(confidences.cpu().numpy().copy())
            pred_coords_list.append(pred.copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())
            
    timestamps = np.concatenate(timestamps)
    track_ids = np.concatenate(agent_ids)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    from tempfile import gettempdir


    pred_path = f"{args.ckpt}.csv"
    

    write_pred_csv(pred_path,
            timestamps=timestamps,
            track_ids=track_ids,
            coords=coords,
            confs=confs)

    if args.mode == "eval":
        from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
        metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
        for metric_name, metric_mean in metrics.items():
            print(metric_name, metric_mean)

        print(metrics['neg_multi_log_likelihood'])
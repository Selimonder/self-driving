import numpy as np
import os, sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import *

from src.dataset import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
import bz2, pickle

import torchvision
from src.loss import *

from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from efficientnet_pytorch import EfficientNet

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR_INPUT = "lyft-motion-prediction-autonomous-vehicles"
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
        'key': 'scenes/train.zarr',
        'batch_size': 36,
        'shuffle': True,
        'num_workers': 8,
    },
    
     'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 14,
        'shuffle': True,
        'num_workers': 8,
    },
       
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0,
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 500000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="600px_b1",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


###

class LyftModel(pl.LightningModule):
    def __init__(self, cfg: Dict, learning_rate = 0.0003, num_modes=3):
        super().__init__()

        self.learning_rate = learning_rate
        ## c
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        
        self.effnet0 = EfficientNet.from_pretrained('efficientnet-b1', in_channels=65, num_classes=self.num_preds + num_modes,)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=10e-5
        return [optimizer]

if __name__ == '__main__':

    ## MAIN TRAINING_SCRIPT ##
    

    callbacks = [LearningRateMonitor(logging_interval='step'), 
    CheckpointEveryNSteps(save_step_frequency=1000)]

    train_dataset = MotionPredictDataset(cfg)
    train_cfg = cfg['train_data_loader']
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], 
                             num_workers=train_cfg['num_workers'], shuffle=train_cfg['shuffle'], pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LyftModel(cfg)#.to(device)

    ckpt_path = "65frame/lightning_logs/version_10/checkpoints/600px_b1_5_121000.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['state_dict'])
    
    # model.effnet0._conv_stem = Conv2dStaticSamePadding(65, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=(300, 300))
    # print(model.effnet0)
    # print(f"From {ckpt_path}")
    # 
    trainer = Trainer(resume_from_checkpoint=ckpt_path, default_root_dir='./65frame', gpus=1, precision=16,max_epochs=100, callbacks=callbacks)#  gradient_clip_val=0.5  # precision=16,, limit_train_batches=0.8) 
    trainer.fit(model, train_loader)
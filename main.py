import numpy as np
import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK']='False'

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
from dataset import *
from model import *
from effnet import EfficientNet
from loss import *
from linformer_pytorch import Linformer

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda.amp import GradScaler, autocast 

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
        'raster_size': [128, 128],  ## 300 300
        'pixel_size': [0.33, 0.33], ## 0.5 0.5
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 8,
    },

    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 8,
        'shuffle': False,
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
        prefix="N-Step-Checkpoint",
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


if __name__ == '__main__':
    print("starting")
    callbacks = [
        pl.callbacks.LearningRateLogger('step'),
        CheckpointEveryNSteps(save_step_frequency=2500)
    ]

    train_dataset = MotionPredictDataset(cfg)
    print("train_dataset ok")
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_data_loader']['batch_size'], 
                             num_workers=cfg['train_data_loader']['num_workers'], shuffle=True)

    
    ## test_mask_path = os.path.join(DIR_INPUT, 'scenes/mask.npz') for test
    

    t_start = time.time()
    for i, batch in enumerate(train_loader):
        if i == 16:
            break
        print(i, batch['image'].shape)

    t_end = time.time()
    t_total = t_end - t_start
    print(t_total)

    # ==== INIT MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)


    class LyftModel(pl.LightningModule):
        def __init__(self, cfg: Dict, num_modes=3):
            super().__init__()

            ## c
            num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
            num_in_channels = 3 + num_history_channels

            # X, Y coords for the future positions (output shape: Bx50x2)
            self.future_len = cfg["model_params"]["future_num_frames"]
            num_targets = 2 * self.future_len

            # TODO: support other than resnet18?
            self.backbone = EfficientNet.from_pretrained('efficientnet-b1', in_channels=num_in_channels)
            backbone_out_features = self.backbone._fc.weight.shape[1]

            self.linformer = Linformer(
            input_size=2, # Dimension 1 of the input
            channels=1280, # Dimension 2 of the input
            dim_d=64, # Overwrites the inner dim of the attention heads. If None, sticks with the recommended channels // nhead, as in the "Attention is all you need" paper
            dim_k=128, # The second dimension of the P_bar matrix from the paper
            dim_ff=128, # Dimension in the feed forward network
            dropout_ff=0.15, # Dropout for feed forward network
            nhead=4, # Number of attention heads
            depth=2, # How many times to run the model
            dropout=0.1, # How much dropout to apply to P_bar after softmax
            activation="gelu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
            checkpoint_level="C0", # What checkpoint level to use. For more information, see below.
            parameter_sharing="layerwise", # What level of parameter sharing to use. For more information, see below.
            k_reduce_by_layer=0, # Going down `depth`, how much to reduce `dim_k` by, for the `E` and `F` matrices. Will have a minimum value of 1.
            full_attention=False, # Use full attention instead, for O(n^2) time and space complexity. Included here just for comparison
            include_ff=True, # Whether or not to include the Feed Forward layer
            w_o_intermediate_dim=None, # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
            )

            # You can add more layers here.
            self.head = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(in_features=2560, out_features=2560),
            )

            self.num_preds = num_targets * num_modes
            self.num_modes = num_modes

            self.logit = nn.Linear(2560, out_features=self.num_preds + num_modes)

        def init_weight():
            init_layer(self.logit)

        def forward(self, x):
            x = self.backbone(x)
            x = F.avg_pool2d(x, kernel_size=(2, 2))
            x = F.dropout(x, p=0.2, training=self.training)        
            x = torch.mean(x, dim=3)
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2

            #x = F.dropout(x, p=0.3, training=self.training)
            x = x.transpose(1, 2)
            x = self.linformer(x)
            x = x.transpose(1, 2)

            x = torch.flatten(x, 1)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.logit(x)
            
            # pred (bs)x(modes)x(time)x(2D coords)
            # confidences (bs)x(modes)
            bs, _ = x.shape
            pred, confidences = torch.split(x, self.num_preds, dim=1)
            pred = pred.view(bs, self.num_modes, self.future_len, 2)
            assert confidences.shape == (bs, self.num_modes)
            confidences = torch.softmax(confidences, dim=1)
            return pred, confidences

        def training_step(self, batch, batch_idx):
            data = batch
            inputs = data["image"].to(device)
            target_availabilities = data["target_availabilities"].to(device)
            targets = data["target_positions"].to(device)

            pred, confidences = self(inputs)
            loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities)

            result = pl.TrainResult(loss)
            result.log('train_loss', loss, on_epoch=True)
            return result

        # def validation_step(self, batch, batch_idx):
        #     data = batch
        #     inputs = data["image"].to(device)
        #     target_availabilities = data["target_availabilities"].to(device)
        #     targets = data["target_positions"].to(device)

        #     pred, confidences = self(inputs)
        #     loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities)

        #     result = pl.EvalResult(checkpoint_on=loss)
        #     result.log('val_loss', loss)
        #     return result

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
            #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #                            self.optimizer, max_lr=3e-3,
            #                            steps_per_epoch=int(22000000/128),
            #                            epochs=1)
            return optimizer#, [self.scheduler]

    # default used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )



    model = LyftModel(cfg)

    resume_ckpt = "lightning_logs/version_21/checkpoints/N-Step-Checkpoint_2_90000.ckpt"
    trainer = Trainer(resume_from_checkpoint=resume_ckpt, gpus=1, max_epochs=10, precision=16, gradient_clip_val=0.5, callbacks=callbacks)
    trainer.fit(model, train_loader)
import numpy as np
import os, sys, time
# os.environ['KMP_DUPLICATE_LIB_OK']='False'

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

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

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
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 8,
    },

    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 2,
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 500000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}

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
        CheckpointEveryNSteps(save_step_frequency=2500)
    ]

    train_dataset = MotionPredictDataset(cfg)
    train_cfg = cfg['train_data_loader']
    print("train_dataset ok")
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], 
                             num_workers=train_cfg['num_workers'], shuffle=train_cfg['shuffle'], pin_memory=False)


    # val_dataset = MotionPredictDataset(cfg, str_loader="val_data_loader")
    # print("val_data_ok ok")
    # val_loader = DataLoader(val_dataset, batch_size=cfg['val_data_loader']['batch_size'], 
    #                          num_workers=cfg['val_data_loader']['num_workers'], shuffle=False)

    
    # ## test_mask_path = os.path.join(DIR_INPUT, 'scenes/mask.npz') for test
    

    # t_start = time.time()
    # for i, batch in enumerate(train_loader):
    #     if i == 16:
    #         break
    #     print(i, batch['image'].shape)

    # t_end = time.time()
    # t_total = t_end - t_start
    # print(t_total)

    # ==== INIT MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)


    from src.loss import *
    from linformer_pytorch import Linformer
    from effnet import *

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


    resume_ckpt = "lightning_logs/version_4/checkpoints/N-Step-Checkpoint_3_57500.ckpt"
    model.load_state_dict(torch.load(resume_ckpt)['state_dict'])
    ## gonna check the .15pct of the val_set 
    trainer = Trainer(gpus=1, max_epochs=100, 
                     precision=16, callbacks=callbacks, limit_train_batches=0.1, gradient_clip_val=1.0)
                     ## excluding | gradient_clip_val=0.5, | val_check_interval=0.50, test_percent_check=0.1 , auto_lr_find=True

    trainer.fit(model, train_loader)
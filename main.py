import numpy as np
import os, time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from src.dataset import *
from src.model import *

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
        'raster_size': [300, 300],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0,
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 6000000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

if __name__ == '__main__':

    print("starting")

    train_dataset = MotionPredictDataset(cfg)
    print("train_dataset ok")
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_data_loader']['batch_size'], 
                                pin_memory=True, num_workers=4)
    
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LyftModel(cfg)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    # model = nn.DataParallel(model)
    # Later we have to filter the invalid steps.
    criterion = nn.MSELoss(reduction="none")

    # ==== TRAIN LOOP

    progress_bar = tqdm(total =cfg["train_params"]["max_num_steps"])
    losses_train = []

    global_step = 0
    epochs = 20
    for epoch in range(epochs):
        for data in train_loader:

            model.train()
            torch.set_grad_enabled(True)

            # Forward pass
            inputs = data["image"].to(device)
            target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
            targets = data["target_positions"].to(device)

            outputs = model(inputs).reshape(targets.shape)
            loss = criterion(outputs, targets)

            # not all the output steps are valid, but we can filter them out from the loss using availabilities
            loss = loss * target_availabilities
            loss = loss.mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            
            if (global_step+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:
                torch.save(model.state_dict(), f'model_state_{itr}.pth')

            progress_bar.update()
            progress_bar.set_description(f"loss: {loss.item():.6f} loss(avg): {np.mean(losses_train[-100:]):.6f}")
            
            global_step += 1
        
        progress_bar.reset()

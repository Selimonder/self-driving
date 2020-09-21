import os, bz2, pickle, random
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from tqdm import tqdm
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

import l5kit
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from multiprocessing import Pool

if __name__ == '__main__':
    
    DIR_INPUT = "../input/lyft-motion-prediction-autonomous-vehicles"

    SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
    MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

    DEBUG = False

    print(f"l5kit version: {l5kit.__version__}")
    print(f"pytorch version: {torch.__version__}")
    
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
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 16,
        },

        'train_params': {
            'max_num_steps': 1000 if DEBUG else 22496709,
            'checkpoint_every_n_steps': 5000,

            # 'eval_every_n_steps': -1
        }
    }
    
   
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    def save_sample(i):
        #idx = random.randint(0, len(dataset))

        # 300px, 0.5 raster size, 10 historical frames
        obj_save(dataset[i], f'sample_{i}', './cache/pre_300px__0_5__10')

    def obj_save(obj, name, dir_cache):
        with bz2.BZ2File(f'{dir_cache}/{name}.pbz', 'wb') as f:
            pickle.dump(obj, f)

    dm = LocalDataManager()
    dataset_path = dm.require(cfg["train_data_loader"]["key"])
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    rast = build_rasterizer(cfg, dm)
    dataset = AgentDataset(cfg, zarr_dataset, rast)
    
    max_ = len(dataset)
    print(f"max: {len(dataset)}")
    print(f"cfg : {cfg['train_params']['max_num_steps']}")
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(save_sample, range(max_)), total=max_))
import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from config import get_config_regression, get_config_tune
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import assign_gpu, count_parameters, setup_seed
from typing import Union

# 将MMSA_test函数的代码复制到这里

def MMSA_test(
    config: Union[dict, str],
    weights_path: str,
    feature_path: str, 
    # seeds: list = [], 
    gpu_id: int = 0, 
):
    """Test MSA models on a single sample.

    Load weights and configs of a saved model, input pre-extracted
    features of a video, then get sentiment prediction results.

    Args:
        model_name: Name of MSA model.
        config: Config dict or path to config file. 
        weights_path: Pkl file path of saved model weights.
        feature_path: Pkl file path of pre-extracted features.
        gpu_id: Specify which gpu to use. Use cpu if value < 0.
    """
    if type(config) == str or type(config) == Path:
        config = Path(config)
        with open(config, 'r') as f:
            config_data  = json.load(f)
            args = config_data['afn_ag_msa']['commonParams']
            args['model_name'] = 'afn_ag_msa'
            args.update(config_data['afn_ag_msa']['datasetParams']['mosi'])
            # args.need_data_aligned = False
    elif type(config) == dict or type(config) == edict:
        args = config
        args['model_name']='afn_ag_msa'
    else:
        raise ValueError(f"'config' should be string or dict, not {type(config)}")
    args['train_mode'] = 'regression' # backward compatibility.

    if gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    args['device'] = device
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1]]
    args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0]]
    model = AMIO(args)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.get('use_bert', None):
            if type(text := feature['text_bert']) == np.ndarray:
                text = torch.from_numpy(text).float()
        else:
            if type(text := feature['text']) == np.ndarray:
                text = torch.from_numpy(text).float()
        if type(audio := feature['audio']) == np.ndarray:
            audio = torch.from_numpy(audio).float()
        if type(vision := feature['vision']) == np.ndarray:
            vision = torch.from_numpy(vision).float()
        text = text.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        vision = vision.unsqueeze(0).to(device)
        if args.get('need_normalized', None):
            audio = torch.mean(audio, dim=1, keepdims=True)
            vision = torch.mean(vision, dim=1, keepdims=True)
        # TODO: write a do_single_test function for each model in trains
        if args['model_name'] == 'afn_ag_msa' or args['model_name'] == 'afn_ag_msa' :
            output = model(text, (audio, torch.tensor(audio.shape[1]).unsqueeze(0)), (vision, torch.tensor(vision.shape[1]).unsqueeze(0)))
        elif args['model_name'] == 'tfr_net':
            input_mask = torch.tensor(feature['text_bert'][1]).unsqueeze(0).to(device)
            output, _ = model((text, text, None), (audio, audio, input_mask, None), (vision, vision, input_mask, None))
        else:
            output = model(text, audio, vision)
        if type(output) == dict:
            output = output['M']
    return output.cpu().detach().numpy()[0][0]
   
# 设置路径和GPU ID
config_path = '/home/liuweilong/MMSA_AFN_AG_MSA/src/MMSA/config/afn_ag_msa.json'
weights_path = '/home/liuweilong/MMSA_AFN_AG_MSA/saved_models/afn_ag_msa-mosi.pth'
feature_path = '/home/liuweilong/caogao/benke/test_features/test_happy_en.pkl'
gpu_id = 0  # 使用GPU 0，如果要在CPU上运行，设置为-1

# 调用MMSA_test函数
output = MMSA_test(config_path, weights_path, feature_path, gpu_id)

# 打印输出结果
print(output)

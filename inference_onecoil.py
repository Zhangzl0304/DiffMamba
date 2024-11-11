import os

################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
result_dir = "/home/zhenlin/DiffMamba/Result_Multi/v5_c4/"
model_path = "/home/zhenlin/DiffMamba/log/v5_cmask4_Tssm_conv_lnorm/model_597000.pt"
pic_save_path = '/home/zhenlin/DiffMamba/Pic_Result_Multi/v5_c4/'
npz_save_path = "/bigdata/zhenlin/DiffMamba/MultiCoil/v5_c4/"
################################################


from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_full_size_model_and_diffusion,
)
from improved_diffusion.train_util import TrainLoop
import warnings
warnings.filterwarnings('ignore')
from data_loader_Cmask import TestCmaskNormMultiOneCoil
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from torch import optim
from tqdm import tqdm
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#inference check
from improved_diffusion.sampling_util import CMR_sampling_Multi_onecoil
# dist_util.setup_dist()
# import wandb
# wandb.init(project="DiffCMR_Multich_inf_hpc", name = 'Model_State 112236 try')
# wandb.config = {"step": 1000-45, "round": 1,"model":'112236','sampler':'ddim'}

logger.configure(dir=result_dir)
arg_dict = model_and_diffusion_defaults()

arg_dict["image_size"]=128

print(arg_dict)
model, diffusion = create_full_size_model_and_diffusion(**arg_dict)
model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )

logger.log("creating model and diffusion...")
logger.log(f"Model parameters: {count_parameters(model)}")
logger.log(f"Using model State {model_path}")
logger.log(f"using GPU{os.environ['CUDA_VISIBLE_DEVICES']}")
logger.log(f"Model info:{arg_dict}")
model.to(dist_util.dev())
model.eval()


Testdata = TestCmaskNormMultiOneCoil(4, 'sax', norm = '01')
torch.set_num_threads(8)

CMR_sampling_Multi_onecoil(diffusion, model, result_dir, Testdata, logger, True, pic_save_path=pic_save_path,npz_save_path=npz_save_path)

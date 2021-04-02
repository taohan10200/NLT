import os
from easydict import EasyDict as edict
import time
import torch

__C = edict()
cfg = __C

__C.seed = 3035  # random seed, for reproduction

# ---------------the parameters of training----------------
__C.sou_dataset = 'GCC'  #
__C.model_type = 'vgg16' # choices=['ResNet50','vgg16']
__C.phase ='DA_train' # choices=['DA_train','pre_train' , 'fine_tune'])
__C.gpu_id = "0,1"     # single gpu:"0"..; multi gpus:"2,3,
__C.target_dataset ='SHHB' # dataset choices =  ['SHHB',  'UCF50',  'QNRF', 'MALL', 'UCSD', 'SHHA']


__C.init_weights=None





# ======== nlt learning rate setting==============
__C.step_size=10  # The number of epochs to reduce the meta learning rates
__C.gamma=0.98  # Gamma for the meta-train learning rate decay
__C.nlt_lr = 1e-5               # MCNN：1e-4 Res50:1e-5
__C.nlt_lr_decay = 1e-4         # learning rate decay rate
__C.DA_stop_epoch = 200



# ======Parameters for pretain phase===========
__C.pre_lr=1e-5  # Learning rate for pre-train phase
__C.pre_weight_decay=1e-4  # Weight decay for the optimizer during pre-train
__C.pre_gamma=0.98 # Gamma for the pre-train learning rate decay
__C.pre_max_epoch=100 # Epoch number for pre-train phase
__C.pre_step_size = 2
__C.pre_batch_size = 8

# ======Parameters for fine_tune phase===========
__C.GCC_pre_train_model = './exp/SHHB/pre/04-01_07-17_SHHB_vgg16__lr1e-05pre_train/all_ep_5_mae_87.7_mse_198.7.pth'
__C.fine_lr = 1e-6
__C.fine_weight_decay=1e-4  # Weight decay for the optimizer during fine_tune
__C.fine_step_size = 2
__C.fine_gamma=0.98 # Gamma for the pre-train learning rate decay
# ==============common parameters==================
__C.print_freq = 50  # print frequency
__C.max_epoch=250
__C.VAL_DENSE_START = 150
__C.val_freq = 1           # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now + "_" + __C.target_dataset + "_" + __C.model_type + "_" +\
             '_lr' + str(__C.pre_lr) + '_'+\
            __C.phase

if __C.target_dataset == "SHHA":
    __C.EXP_PATH = "./exp/SHHA"
if __C.target_dataset == "QNRF":
    __C.EXP_PATH = "./exp/QNRF"
if __C.target_dataset == "SHHB":
    __C.EXP_PATH = "./exp/SHHB"
if __C.target_dataset == "MALL":
    __C.EXP_PATH = "./exp/MALL"
if __C.target_dataset == "UCSD":
    __C.EXP_PATH = "./exp/UCSD"
if __C.target_dataset == "UCF50":
    __C.EXP_PATH = "./exp/UCF50"
if __C.target_dataset == "GCC":
    __C.EXP_PATH = "./exp"
if __C.target_dataset == "WE":
    __C.EXP_PATH = "./exp/WE"


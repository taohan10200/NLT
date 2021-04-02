from easydict import EasyDict as edict
from config import cfg
#init
__C_GCC = edict()
cfg_data = __C_GCC

__C_GCC.STD_SIZE = (544,960)   #original image size
__C_GCC.TRAIN_SIZE = (480,480)     # (480,848) #Compress image size to reduce training time

__C_GCC.DATA_PATH = '../ProcessedData'
__C_GCC.GCC_scene_dir = './data_split/GCC_scenes'

__C_GCC.sou_batch_size = 4
__C_GCC.target_shot_size = 4  # The number of training samples for each class in a task



__C_GCC.DATA_GT = 'k15_s4'
__C_GCC.MEAN_STD =([0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.184846073389])
# ([0.34221865981849997, 0.36924636363950003, 0.3505347073075],[0.230081893504,0.217988729477,0.203343153])
__C_GCC.MALL_DATA_PATH = '../ProcessedData/MALL'
__C_GCC.MALL_scene_dir = './data_split/MALL_scenes'
__C_GCC.MALL_TRAIN_SIZE = (360,480)     # (480,848) #Compress image size to reduce training time
__C_GCC.MALL_STD_SIZE = (480,640)   #original image size

__C_GCC.UCSD_DATA_PATH = '../ProcessedData/UCSD'
__C_GCC.UCSD_scene_dir = './data_split/UCSD_scenes'
__C_GCC.UCSD_TRAIN_SIZE = (360,544)     # (480,848) #Compress image size to reduce training time
__C_GCC.UCSD_STD_SIZE = (480,720)   #original image size


__C_GCC.SHHB_DATA_PATH = '../ProcessedData/SHHB'
__C_GCC.SHHB_scene_dir = './data_split/SHHB_scenes'
__C_GCC.SHHB_TRAIN_SIZE = (576,768)
__C_GCC.SHHB_STD_SIZE = (768,1024)   #original image size

__C_GCC.QNRF_DATA_PATH = '../ProcessedData/QNRF'
__C_GCC.QNRF_scene_dir = './data_split/QNRF_scenes'
__C_GCC.QNRF_TRAIN_SIZE = (480,480)
__C_GCC.QNRF_STD_SIZE = (768,1024)   #original image size

__C_GCC.SHHA_DATA_PATH = '../ProcessedData/SHHA'
__C_GCC.SHHA_scene_dir = './data_split/SHHA_scenes'
__C_GCC.SHHA_TRAIN_SIZE = (480,480)
__C_GCC.SHHA_STD_SIZE = (768,1024)   #original image size

__C_GCC.WE_DATA_PATH = '../ProcessedData/WE'
__C_GCC.WE_scene_dir = './data_split/WE_scenes'
__C_GCC.WE_TRAIN_SIZE = (432,544)
__C_GCC.WE_STD_SIZE = (576,720)   #original image size
__C_GCC.WE_test_list =['104207','200608','200702','202201','500717']


__C_GCC.LABEL_FACTOR = 1
__C_GCC.LOG_PARA  = 100.
__C_GCC.num_batch = 100 # The number for different tasks used for meta-train
__C_GCC.val_batch_size = 1


__C_GCC.MAX_LIST = [10.0,25.0,50.0,100.0,300.0,600.0,1000.0,2000.0,4000.0]
if cfg.target_dataset == 'MALL':
    __C_GCC.FILTER_RULE = {'level': [1,2,3,4],
                                'time_duration': [8, 18],
                                'weather': [0,1,5,6],
                                'cnt_range': [0,200],
                                'min_ratio':0.0} # Mall
elif cfg.target_dataset == 'SHHB':
    __C_GCC.FILTER_RULE = {'level': [1,2,3,4,5],
                           'time_duration': [6, 20],
                           'weather': [0,1,5,6],
                           'cnt_range': [10,600],
                           'min_ratio':0.3} # B
elif cfg.target_dataset == 'SHHA':
   # __C_GCC.FILTER_RULE = {'level': [4,5,6,7,8],
   #                        'time_duration': [6, 20],
   #                        'weather': [0,1,3,5,6],
   #                        'cnt_range': [33,4000],
   #                        'min_ratio':0.5} # A
    __C_GCC.FILTER_RULE = {'level': [4,5,6,7,8], 'time_duration': [5, 21], 'weather': [0,1,5,6], 'cnt_range': [400,4000], 'min_ratio':0.6} # UCF QNRF

   # __C_GCC.FILTER_RULE = {'level': [1,2,3,4,5], 'time_duration': [6, 20], 'weather': [0,1,5,6], 'cnt_range': [10,600], 'min_ratio':0.3} # B
# __C_GCC.FILTER_RULE = {'level': [5,6,7,8], 'time_duration': [8, 18], 'weather': [0,1,5,6], 'cnt_range': [400,4000], 'min_ratio':0.6} # UCF 50
elif cfg.target_dataset == 'QNRF':
    __C_GCC.FILTER_RULE = {'level': [4,5,6,7,8], 'time_duration': [5, 21], 'weather': [0,1,5,6], 'cnt_range': [400,4000], 'min_ratio':0.6} # UCF QNRF
elif cfg.target_dataset == 'WE':
    __C_GCC.FILTER_RULE = {'level': [2,3,4,5,6], 'time_duration': [6, 19], 'weather': [0,1,5,6], 'cnt_range': [0,1000], 'min_ratio':0.0} # WE
# __C_GCC.FILTER_RULE = {'level': [1,2,3,4], 'time_duration': [8, 18], 'weather': [0,1,5,6], 'cnt_range': [0,200], 'min_ratio':0.0} # Mall
#
elif cfg.target_dataset=='UCSD':
    __C_GCC.FILTER_RULE = {'level': [1, 2, 3, 4],
                           'time_duration': [8, 18],
                           'weather': [0, 1, 5, 6],
                           'cnt_range': [0, 200],
                           'min_ratio': 0.0}  # UCSD

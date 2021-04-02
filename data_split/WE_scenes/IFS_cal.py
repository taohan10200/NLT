import os
import pandas as pd
from glob import glob
import numpy as np
import operator
data_path = '/media/D/ht/ProcessedData/QNRF/train/den'
data_path='/media/D/ht/C-3-Framework-trans/trans-display/GCC2MALL/GCC_s2t'
paths =  glob(os.path.join(data_path, '*s2t'))
print(paths)
img=[]
a = {}

for path in paths:
    src=path
    dst=path+'.png'
    os.rename(src,dst)
    print(dst)
#     den = pd.read_csv(path, sep=',', header=None).values
#     den = den.astype(np.float32, copy=False)
#
#     img.append({'name':path.split('/')[-1],'count':np.sum(den) })
#     # print(img)
# img.sort(key=lambda obj:(obj.get('count')), reverse=False)
# print( img)
#
# with open('train/train_.txt') as f:
#     lines = f.readlines()
#
# data_files = []
# for line in lines:
#     line = line.strip('\n')
#     data_files.append(line)
# print(data_files)
#
# a = {'0-50':[],'50-100':[],'100-150':[],'150-200':[],'250-300':[],'300-350':[]}
# target = []
# flag = 0
# with open('IFStrain.txt','a') as f:
#
#         for name in paths:
#             f.write(name+'\n')
#     f.close()
# print(len(target))
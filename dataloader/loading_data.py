import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader.GCC import  GCC
from dataloader.MALL import  MALL
from dataloader.SHHB import SHHB
from dataloader.UCSD import UCSD
from dataloader.WE import WE
from dataloader.SHHA import SHHA
from dataloader.QNRF import QNRF
from dataloader.setting import cfg_data
import torchvision
import torch
import random


def get_min_size(batch):
    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]

    for i_sample in batch:

        _, ht, wd = i_sample.shape
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd


def random_crop(img, den, dst_size):
    # dst_size: ht, wd

    _, ts_hd, ts_wd = img.shape

    # print img.shape
    # print den.shape
    dst_size[0] -= dst_size[0] % 16
    dst_size[1] -= dst_size[1] % 16
    x1 = int(random.randint(0, ts_wd - dst_size[1]) / cfg_data.LABEL_FACTOR * cfg_data.LABEL_FACTOR)
    y1 = int(random.randint(0, ts_hd - dst_size[0]) / cfg_data.LABEL_FACTOR * cfg_data.LABEL_FACTOR)
    x2 = int(x1 + dst_size[1])
    y2 = int(y1 + dst_size[0])

    label_x1 = int(x1 / cfg_data.LABEL_FACTOR)
    label_y1 = int(y1 / cfg_data.LABEL_FACTOR)
    label_x2 = int(x2 / cfg_data.LABEL_FACTOR)
    label_y2 = int(y2 / cfg_data.LABEL_FACTOR)

    return img[:, y1:y2, x1:x2], den[label_y1:label_y2, label_x1:label_x2]


def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out


def SHHA_collate(batch):
    # @GJY
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch))  # imgs and dens
    imgs, dens = [transposed[0], transposed[1]]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        min_ht, min_wd = get_min_size(imgs)

        # pdb.set_trace()
        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample], dens[i_sample], [min_ht, min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)

        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs, cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))

def loading_data(args):
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA

    sou_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
        own_transforms.RandomHorizontallyFlip(),
        # Rand_Augment()
    ])

    # converts a PIL Image(H*W*C) in the range[0,255]
    # to a torch.FloatTensor of shape (C*H*W) in the range[0.0, 1.0]
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    if args.phase == 'DA_train' or  args.phase== 'fine_tune':
        # Load meta-train set
        IFS_path = '/media/D/ht/C-3-Framework-trans/trans-display/GCC2SHHB/s2t'
        IFS_path = '/media/D/ht/C-3-Framework-trans/trans-display/GCC2QNRF/s2t'
        IFS_path = '/media/D/ht/C-3-Framework-trans/trans-display/GCC2WE/s2t'
        trainset = GCC('train',main_transform=sou_main_transform, img_transform=img_transform, gt_transform=gt_transform,filter_rule=cfg_data.FILTER_RULE,IFS_path=None)
        sou_loader=DataLoader(trainset, batch_size=cfg_data.sou_batch_size, shuffle=True, num_workers=12, drop_last=True, pin_memory=True)

        if args.target_dataset == 'QNRF':
            tar_main_transform = own_transforms.Compose([
                own_transforms.RandomHorizontallyFlip()
            ])
            trainset = QNRF('train', main_transform=tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_shot_loader = DataLoader(trainset, batch_size=cfg_data.target_shot_size, shuffle=True, num_workers=12, collate_fn=SHHA_collate, drop_last=True)

            valset = QNRF('val',img_transform=img_transform, gt_transform=gt_transform)
            tar_val_loader = DataLoader(valset, batch_size=1, num_workers=8, pin_memory=True)

            testset = QNRF('test',img_transform=img_transform, gt_transform=gt_transform)
            tar_test_loader = DataLoader(testset, batch_size=1, num_workers=8, pin_memory=True)
        elif args.target_dataset == 'SHHA':
            tar_main_transform = own_transforms.Compose([
                own_transforms.RandomHorizontallyFlip()
            ])
            trainset = SHHA('train', main_transform=tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_shot_loader = DataLoader(trainset, batch_size=cfg_data.target_shot_size, shuffle=True, num_workers=12, collate_fn=SHHA_collate, drop_last=True)

            valset = SHHA('val',img_transform=img_transform, gt_transform=gt_transform)
            tar_val_loader = DataLoader(valset, batch_size=1, num_workers=8, pin_memory=True)

            testset = SHHA('test',img_transform=img_transform, gt_transform=gt_transform)
            tar_test_loader = DataLoader(testset, batch_size=1, num_workers=8, pin_memory=True)

        elif args.target_dataset == 'MALL':
            tar_main_transform = own_transforms.Compose([
                own_transforms.RandomCrop(cfg_data.MALL_TRAIN_SIZE),
                own_transforms.RandomHorizontallyFlip()
            ])
            trainset = MALL('train', main_transform=tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_shot_loader = DataLoader(trainset, batch_size=cfg_data.target_shot_size, shuffle=True, num_workers=12, drop_last=True, pin_memory=True)

            valset = MALL('val',img_transform=img_transform, gt_transform=gt_transform)
            tar_val_loader = DataLoader(valset, batch_size=8, num_workers=8, pin_memory=True)

            testset = MALL('test',img_transform=img_transform, gt_transform=gt_transform)
            tar_test_loader = DataLoader(testset, batch_size=12, num_workers=8, pin_memory=True)

        elif args.target_dataset == 'UCSD':
            tar_main_transform = own_transforms.Compose([
                own_transforms.RandomCrop(cfg_data.UCSD_TRAIN_SIZE),
                own_transforms.RandomHorizontallyFlip(),
            ])
            trainset = UCSD('train', main_transform=tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_shot_loader = DataLoader(trainset, batch_size=cfg_data.target_shot_size, shuffle=True, num_workers=12, drop_last=True, pin_memory=True)

            valset = UCSD('val', img_transform=img_transform, gt_transform=gt_transform)
            tar_val_loader = DataLoader(valset, batch_size=8, num_workers=8, pin_memory=True)

            testset = UCSD('test',img_transform=img_transform, gt_transform=gt_transform)
            tar_test_loader = DataLoader(testset, batch_size=12, num_workers=8, pin_memory=True)
        elif args.target_dataset == 'SHHB':
            tar_main_transform = own_transforms.Compose([
                own_transforms.RandomCrop(cfg_data.SHHB_TRAIN_SIZE),
                own_transforms.RandomHorizontallyFlip(),
                # Rand_Augment()
            ])

            trainset = SHHB('train',  main_transform= tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_shot_loader = DataLoader(trainset, batch_size=cfg_data.target_shot_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

            valset = SHHB('val',img_transform=img_transform, gt_transform=gt_transform)
            tar_val_loader = DataLoader(valset, batch_size=8, num_workers=8, pin_memory=True)

            testset = SHHB('test',img_transform=img_transform, gt_transform=gt_transform)
            tar_test_loader = DataLoader(testset, batch_size=8, num_workers=8, pin_memory=True)

        elif args.target_dataset == 'WE':
            tar_test_loader = []
            tar_main_transform = own_transforms.Compose([
                own_transforms.RandomCrop(cfg_data.WE_TRAIN_SIZE),
                own_transforms.RandomHorizontallyFlip(),
                # Rand_Augment()
            ])
            trainset = WE(None,'train', main_transform= tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_shot_loader = DataLoader(trainset, batch_size=cfg_data.target_shot_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
            valset = WE(None, 'val', main_transform=tar_main_transform, img_transform=img_transform, gt_transform=gt_transform)
            tar_val_loader = DataLoader(valset, batch_size=12, shuffle=False, num_workers=8, drop_last=False,
                                         pin_memory=True)

            for subname in cfg_data.WE_test_list:
                sub_set = WE(subname, 'test',  img_transform=img_transform, gt_transform=gt_transform)
                tar_test_loader.append(
                    DataLoader(sub_set,batch_size=12,num_workers=8, pin_memory=True)
                )
        else:
            print("Please set the target dataset as one of them：SHHB,  UCF50,  QNRF, MALL, UCSD, SHHA")

        return sou_loader, tar_shot_loader, tar_val_loader, tar_test_loader,restore_transform

    if args.phase == 'pre_train':
        trainset = GCC('train', main_transform=sou_main_transform, img_transform=img_transform, gt_transform=gt_transform)
        train_loader = DataLoader(trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, drop_last=True,pin_memory=True)

        valset = GCC('val',img_transform=img_transform, gt_transform=gt_transform)
        val_loader = DataLoader(valset, batch_size = 12, num_workers=8, pin_memory=True)


        return train_loader, val_loader,restore_transform

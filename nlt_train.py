import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from models.nlt_counter import NLT_Counter
from misc.utils import *
from config import cfg
from dataloader.loading_data import loading_data
import torchvision.utils as vutils
from collections import OrderedDict
import pdb
from  misc.quality import get_psnr,get_ssim

class NLT_Trainer(object):
    def __init__(self, cfg_data, pwd):
        self.cfg_data = cfg_data
        self.pwd = pwd
        self.exp_path = cfg.EXP_PATH
        self.exp_name = cfg.EXP_NAME
        if not osp.exists(self.exp_path):
            os.makedirs(self.exp_path)

        self.sou_loader, self.tar_shot_loader, self.tar_val_loader, self.tar_test_loader, self.restore_transform = loading_data(cfg)

        self.sou_model = NLT_Counter( backbone=cfg.model_type)
        self.tar_model = NLT_Counter( mode='nlt', backbone=cfg.model_type)

        self.sou_optimizer = torch.optim.Adam(self.sou_model.parameters(), lr = cfg.nlt_lr, weight_decay=cfg.nlt_lr_decay)

        self.tar_optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.tar_model.encoder.parameters()), 'lr': cfg.nlt_lr}, \
            {'params': filter(lambda p: p.requires_grad, self.tar_model.decoder.parameters()), 'lr':cfg.nlt_lr}])

        self.sou_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.sou_optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        self.tar_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.tar_optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        #
        if cfg.init_weights is not None:
            self.pretrained_dict = torch.load(cfg.init_weights)  # ['params']
            self.sou_model.load_state_dict(self.pretrained_dict,strict=False)
            self.tar_model.load_state_dict(self.pretrained_dict,strict=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
        self.sou_model = torch.nn.DataParallel(self.sou_model).cuda()
        self.tar_model = torch.nn.DataParallel(self.tar_model).cuda()

        self.tar_model_record = {"best_mae": 1e20, "best_mse": 1e20, "best_model_name": "", "update_flag": 0,
                                 "temp_test_mae": 1e20, "temp_test_mse": 1e20}

        self.sou_model_record = {"best_mae": 1e20, "best_mse": 1e20, "best_model_name": "", "update_flag": 0,
                                 "temp_test_mae": 1e20, "temp_test_mse": 1e20 }
        self.epoch = 0
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ["exp"])

    def forward(self):
        timer = Timer()
        self.global_count = 0
        for epoch in range(1, cfg.max_epoch + 1):
            self.epoch = epoch
            self.train()
            self.fine_tune()
            if self.epoch % cfg.val_freq == 0:
                if cfg.target_dataset is "WE":
                    self.tar_model_V2(self.tar_val_loader, "val")
                if cfg.target_dataset in ["VENICE", "QNRF", "SHHA", "SHHB",  "MALL", "UCSD"]:
                    self.tar_model_V1(self.tar_val_loader, "val")

                print('=' * 50)
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(self.epoch / cfg.max_epoch)))
            self.sou_lr_scheduler.step()
            self.tar_lr_scheduler.step()
        self.writer.close()

    def train(self):
        self.sou_model.train()
        self.tar_model.train()
        train_loss = AverageMeter()
        train_mae = AverageMeter()
        train_mse = AverageMeter()

        shot_loss = AverageMeter()
        shot_mae = AverageMeter()
        shot_mse = AverageMeter()

        for i, (a, b) in enumerate(zip(self.sou_loader, self.tar_shot_loader), 1):
            self.global_count = self.global_count + 1
            sou_img,sou_label = a[0].cuda(),a[1].cuda()
            shot_img, shot_label = b[0].cuda(), b[1].cuda()
            if self.epoch <cfg.DA_stop_epoch:

                # ==================change sou_model parameters===============
                sou_pred = self.sou_model(sou_img)
                loss = F.mse_loss(sou_pred.squeeze(), sou_label.squeeze())

                self.sou_optimizer.zero_grad()
                loss.backward()
                self.sou_optimizer.step()
                self.writer.add_scalar('data/sou_loss', loss.item(), self.global_count)
                train_loss.update(loss.item())
                sou_pred_cnt, sou_label_cnt = self.mae_mse_update(sou_pred, sou_label, train_mae, train_mse)
                self.tar_model.load_state_dict(self.sou_model.state_dict(), strict=False)
                # ================================================================
            else:
                sou_label_cnt=0
                sou_pred_cnt=0
                #=====================change tar_model parameters================

            shot_pred = self.tar_model(shot_img)
            loss_mse = F.mse_loss(shot_pred.squeeze(), shot_label.squeeze())
            loss = self.weight_decay_loss(self.tar_model, 1e-4) + loss_mse
            self.tar_optimizer.zero_grad()
            loss.backward()
            self.tar_optimizer.step()
            self.writer.add_scalar('data/shot_loss', loss.item(), self.global_count)
            shot_loss.update(loss.item())
            pred_cnt, label_cnt = self.mae_mse_update(shot_pred, shot_label, shot_mae, shot_mse)

            # ===============================================================
            if i % cfg.print_freq == 0:
                print('Epoch {}, Loss={:.4f} s_gt={:.1f} s_pre={:.1f},t_gt={:.1f} t_pre={:.1f} lr={:.4f}'.format(
                    self.epoch, loss.item(), sou_label_cnt,sou_pred_cnt,label_cnt, pred_cnt, self.sou_optimizer.param_groups[0]['lr']*10000))

        self.writer.add_scalar('data/train_loss_tar', float(shot_loss.avg), self.epoch)
        self.writer.add_scalar('data/train_mae_tar', float(shot_mae.avg), self.epoch)
        self.writer.add_scalar('data/train_mse_tar', float( np.sqrt(shot_mse.avg)), self.epoch)

        self.writer.add_scalar('data/train_loss_sou', float(train_loss.avg), self.epoch)
        self.writer.add_scalar('data/train_mae_sou', float(train_mae.avg), self.epoch)
        self.writer.add_scalar('data/train_mse_sou', float(np.sqrt(train_mse.avg)), self.epoch)

        # Start validation for this epoch, set model to eval mode
    def fine_tune(self):
        for i, (shot_img, shot_label) in enumerate(self.tar_shot_loader, 1):
            if i <= 50:
                shot_img = shot_img.cuda()
                shot_label = shot_label.cuda()
                shot_pred = self.tar_model(shot_img)

                loss_mse = F.mse_loss(shot_pred.squeeze(), shot_label.squeeze())
                loss = self.weight_decay_loss(self.tar_model, 1e-4) + loss_mse
                self.tar_optimizer.zero_grad()
                loss.backward()
                self.tar_optimizer.step()
            else:
                break

    def tar_model_V2(self, dataset, mode=None):# Run meta-validatio
        self.tar_model.eval()
        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)
        val_losses = AverageMeter()
        val_maes = AverageMeter()
        if mode =='val' :
            for i, batch in enumerate(dataset, 1):
                with torch.no_grad():
                    img = batch[0].cuda()
                    label = batch[1].cuda()
                    pred = self.tar_model(img)
                    self.mae_mse_update(pred, label, val_maes, losses=val_losses)
            mae = np.average(val_maes.avg)
            loss = np.average(val_losses.avg)

            self.writer.add_scalar('data/val_mae', mae, self.epoch)
            self.writer.add_scalar('data/val_loss',loss, self.epoch)
            self.tar_model_record = update_model(
                self.tar_model.module, self.epoch, self.exp_path, self.exp_name, [mae, 0, loss], self.tar_model_record,
                self.log_txt)
            print_summary(self.exp_name, [mae, 0, loss], self.tar_model_record)

        else:
            for i_sub, i_loader in enumerate(dataset, 0):
                for i, batch in enumerate(i_loader, 1):
                    with torch.no_grad():
                        img = batch[0].cuda()
                        label = batch[1].cuda()
                        pred = self.tar_model(img)
                        self.mae_mse_update(pred,label,maes=maes,losses=losses,cls_id=i_sub)
                        if i == 1 and self.epoch%10==0:
                            vis_results(self.epoch, self.writer, self.restore_transform,
                                        img, pred.data.cpu().numpy(), label.data.cpu().numpy(), self.exp_name)

            mae = np.average(maes.avg)
            loss = np.average(losses.avg)

            self.writer.add_scalar("data/mae_s1", maes.avg[0], self.epoch)
            self.writer.add_scalar("data/mae_s2", maes.avg[1], self.epoch)
            self.writer.add_scalar("data/mae_s3", maes.avg[2], self.epoch)
            self.writer.add_scalar("data/mae_s4", maes.avg[3], self.epoch)
            self.writer.add_scalar("data/mae_s5", maes.avg[4], self.epoch)

            self.writer.add_scalar("data/test_mae", float(mae), self.epoch)
            self.writer.add_scalar('data/test_loss', float(loss), self.epoch)
            logger_txt(self.log_txt, self.epoch, [mae, 0, loss])
            self.tar_model_record['temp_test_mae'] = mae
            self.tar_model_record['temp_test_mse'] = 0
        # Print loss and maeuracy for this epoch


    def tar_model_V1(self, dataset, mode=None):
        self.tar_model.eval()
        losses = AverageMeter()
        maes  = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()

        # tqdm_gen = tqdm.tqdm(dataset)
        for i, batch in enumerate(dataset, 1):
            with torch.no_grad():
                img = batch[0].cuda()
                label = batch[1].cuda()
                pred = self.tar_model(img)
                if mode == 'test':
                    self.mae_mse_update(pred, label, maes, mses, ssims,psnrs,losses)
                else:
                    self.mae_mse_update(pred, label, maes, mses, losses=losses)
                if i == 1 and self.epoch%10==0:
                    vis_results(self.epoch, self.writer, self.restore_transform,
                                img, pred.data.cpu().numpy(), label.cpu().detach().numpy(), self.exp_name)
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        if mode == "val":
            self.writer.add_scalar('data/val_mae', mae, self.epoch)
            self.writer.add_scalar('data/val_mse', mse, self.epoch)
            self.writer.add_scalar('data/val_loss',loss, self.epoch)
            self.tar_model_record = update_model(
                self.tar_model.module, self.epoch, self.exp_path, self.exp_name, [mae, mse, loss], self.tar_model_record,
                self.log_txt)
            print_summary(self.exp_name, [mae, mse, loss], self.tar_model_record)

        elif mode == "test":
            self.writer.add_scalar('data/test_mae', mae, self.epoch)
            self.writer.add_scalar('data/test_mse', mse, self.epoch)
            self.writer.add_scalar('data/test_loss',loss, self.epoch)
            self.writer.add_scalar("data/test_ssim", ssims.avg, self.epoch)
            self.writer.add_scalar("data/test_psnr", psnrs.avg, self.epoch)

            self.tar_model_record['temp_test_mae'] = mae
            self.tar_model_record['temp_test_mse'] = mse
            logger_txt(self.log_txt, self.epoch, [mae, mse, loss])



    def weight_decay_loss(self,model, lamda):
        loss_weight = 0
        loss_bias = 0
        for name, param in model.named_parameters():
            if  'nlt_weight' in name:
                loss_weight += 0.5 * torch.sum(torch.pow(param - 1, 2))
            elif  'nlt_bias' in name:
                loss_bias   += 0.5 * torch.sum(torch.pow(param, 2))
            return lamda*loss_weight + lamda*10*loss_bias

    def mae_mse_update(self,pred,label,maes,mses=None,ssims=None,psnrs=None,losses=None,cls_id=None):
        for num in range(pred.size()[0]):
            sub_pred = pred[num].data.cpu().squeeze().numpy()/ self.cfg_data.LOG_PARA
            sub_label = label[num].data.cpu().squeeze().numpy() / self.cfg_data.LOG_PARA
            pred_cnt = np.sum(sub_pred)
            gt_cnt =   np.sum(sub_label)
            mae = abs(pred_cnt - gt_cnt)
            mse = (pred_cnt - gt_cnt)*(pred_cnt - gt_cnt)

            if ssims and psnrs is not None:
                ssims.update(get_ssim(sub_label,sub_pred))
                psnrs.update(get_psnr(sub_label,sub_pred))

            if cls_id is not None:
                maes.update(mae,cls_id)
                if losses is not None:
                    loss = F.mse_loss(pred.detach().squeeze(), label.detach().squeeze())
                    losses.update(loss.item(),cls_id)
                if mses is not None:
                    mses.update(mse,cls_id)
            else:
                maes.update(mae)
                if losses is not None:
                    loss = F.mse_loss(pred.detach().squeeze(), label.detach().squeeze())
                    losses.update(loss.item())
                if mses is not None:
                    mses.update(mse)

        return pred_cnt,gt_cnt

def vis_results(epoch, writer, restore, img, pred_map, gt_map, exp_name=None):
    pil_to_tensor = standard_transforms.ToTensor()
    x = []
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
        if idx > 1:  # show only one group
            break
        pil_input = restore(tensor[0])
        pil_label = torch.from_numpy(tensor[2] / (tensor[2].max() + 1e-10)).repeat(3, 1, 1)
        pil_output = torch.from_numpy(tensor[1] / (tensor[2].max() + 1e-10)).repeat(3, 1, 1)
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)
    if 'temp' in exp_name:
        writer.add_image(exp_name, x)
    else:
        writer.add_image(exp_name + '_epoch_' + str(epoch), x)


if __name__ == '__main__':
    import torch

    seed = cfg.seed
    if cfg.seed == 0:
        print('Using random seed.')
        # torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', seed)
        np.random.seed(seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pwd = os.path.split(os.path.realpath(__file__))[0]
    print(pwd)
    if cfg.phase == 'DA_train':
        from dataloader.setting import cfg_data

        trainer = NLT_Trainer(cfg_data, pwd)
        trainer.forward()
    else:
        raise ValueError('Please set correct phase.')
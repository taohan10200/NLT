
""" fine_tune the GCC pre_trained model on target dataset. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.nlt_counter import NLT_Counter
from misc.utils import *
from config import cfg
from dataloader.loading_data import loading_data
import torchvision.utils as vutils
from misc.quality import get_ssim,get_psnr
import pdb

class Fine_tune_Trainer(object):
    def __init__(self, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.pwd = pwd
        self.exp_path = cfg.EXP_PATH
        self.exp_name = cfg.EXP_NAME
        self.exp_path = osp.join(self.exp_path, 'fine_tune')
        if not osp.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.sou_query_loader, self.tar_shot_loader, self.tar_val_loader, self.tar_test_loader,self.restore_transform = loading_data(cfg)

        self.sou_model = NLT_Counter( mode='fine_tune', backbone=cfg.model_type)

        self.sou_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.sou_model.parameters()), lr = cfg.fine_lr, weight_decay=cfg.fine_weight_decay)

        self.sou_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.sou_optimizer, step_size=cfg.fine_step_size, gamma=cfg.fine_gamma)

        if cfg.GCC_pre_train_model is not None:
            print('load GCC pre_trained model')
            self.pretrained_dict = torch.load(cfg.GCC_pre_train_model)
            self.sou_model.load_state_dict(self.pretrained_dict)

        self.sou_model = torch.nn.DataParallel(self.sou_model).cuda()

        self.sou_model_record = {"best_mae": 1e20, "best_mse": 1e20, "best_model_name": "", "update_flag": 0,
                                 "temp_test_mae": 1e20, "temp_test_mse": 1e20}
        self.epoch = 0
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ["exp"])

    def forward(self):
        timer = Timer()
        self.global_count = 0
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            self.epoch = epoch
            if self.epoch % cfg.val_freq == 0:
                self.sou_model_V1(self.tar_val_loader, "val")
                print('=' * 50)
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(self.epoch / cfg.max_epoch)))
            self.sou_lr_scheduler.step()
        self.writer.close()
    def train(self):
        self.sou_model.train()

        train_loss = AverageMeter()
        train_mae = AverageMeter()
        train_mse = AverageMeter()

        for i, (img,gt_map) in enumerate( self.tar_shot_loader, 1):
            self.global_count = self.global_count + 1
            shot_img, shot_label  = img.cuda(), gt_map.cuda()

            # ==================change sou_model parameters===============
            shot_pred = self.sou_model(shot_img)
            loss = F.mse_loss(shot_pred.squeeze(), shot_label.squeeze())
            self.sou_optimizer.zero_grad()
            loss.backward()
            self.sou_optimizer.step()
            train_loss.update(loss.item())
            self.writer.add_scalar('data/fine_tune_loss', float(loss), self.global_count)
            sou_pred_cnt, sou_label_cnt = self.mae_mse_update(shot_pred, shot_label, train_mae, train_mse)

            # ===============================================================
            if i % 50 == 0:
                print('Epoch {}, Loss={:.4f} s_gt={:.1f} s_pre={:.1f}'.format(
                    self.epoch, loss.item(), sou_label_cnt,sou_pred_cnt))

        self.writer.add_scalar('data/train_loss_tar', float(train_loss.avg), self.epoch)
        self.writer.add_scalar('data/train_mae_tar', float(train_mae.avg), self.epoch)
        self.writer.add_scalar('data/train_mse_tar', float(np.sqrt(train_mse.avg)), self.epoch)


        # Start validation for this epoch, set model to eval mode

    def validation(self):# Run meta-validation
        self.sou_model.eval()
        if cfg.target_dataset in ["WE", "SHFD"]:
            val_loss =AverageCategoryMeter(5)
            val_mae = AverageCategoryMeter(5)
            # self.tar_model.eval()
            for i_sub, i_loader in enumerate(self.tar_val_loader, 0):
                tqdm_gen = tqdm.tqdm(i_loader)
                for i, batch in enumerate(tqdm_gen, 1):
                    img = batch[0].cuda()
                    gt_map = batch[1].cuda()
                    with torch.no_grad():
                        pred = self.sou_model(inp=img)
                        self.mae_mse_update(pred, gt_map, val_mae,losses=val_loss,cls_id=i_sub)
                        if i == 1 :
                            vis_results(self.epoch, self.writer, self.restore_transform,
                                        img, pred.data.cpu().numpy(), gt_map.data.cpu().numpy(),  'temp_val/sou')

            mae = np.average(val_mae.avg)
            loss = np.average(val_loss.avg)

            self.writer.add_scalar("data/mae_s1", val_mae.avg[0], self.epoch)
            self.writer.add_scalar("data/mae_s2", val_mae.avg[1], self.epoch)
            self.writer.add_scalar("data/mae_s3", val_mae.avg[2], self.epoch)
            self.writer.add_scalar("data/mae_s4", val_mae.avg[3], self.epoch)
            self.writer.add_scalar("data/mae_s5", val_mae.avg[4], self.epoch)

            self.writer.add_scalar("data/tar_val_mae", float(mae), self.epoch)
            self.writer.add_scalar('data/tar_val_loss', float(loss), self.epoch)


            # Print loss and maeuracy for this epoch
            self.record = update_model(
                self.sou_model.module, self.epoch, self.exp_path, self.exp_name, [mae, 0, loss], self.record,
                self.log_txt)
            print('Epoch {}, Val, mae={:.2f} mse={:.2f}'.format(self.epoch, mae, 0))
            self.record['val_loss'].append(loss)
            self.record['val_mae'].append(mae)

    def sou_model_V1(self, dataset, mode=None):
        self.sou_model.eval()
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
                pred = self.sou_model(img)
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
                self.sou_model.module, self.epoch, self.exp_path, self.exp_name, [mae, mse, loss], self.sou_model_record,
                self.log_txt)
            print_summary(self.exp_name, [mae, mse, loss], self.sou_model_record)

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
            if 'mtl_weight' in name:
                loss_weight += 0.5 * torch.sum(torch.pow(param - 1, 2))
            elif 'mtl_bias' in name:
                loss_bias   += 0.5 * torch.sum(torch.pow(param,2))
            return lamda*loss_weight + lamda*loss_bias

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
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    if cfg.seed == 0:
        print('Using random seed.')
    else:
        print('Using manual seed:', seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    pwd = os.path.split(os.path.realpath(__file__))[0]
    print(pwd)
    if cfg.phase == 'fine_tune':
        from dataloader.setting import cfg_data

        trainer = Fine_tune_Trainer(cfg_data, pwd)
        trainer.forward()
    else:
        raise ValueError('Please set correct phase.')
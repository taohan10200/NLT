
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch.nn.functional as F
from models.nlt_counter import NLT_Counter
from misc.utils import *
from tensorboardX import SummaryWriter
from dataloader.loading_data import loading_data
from dataloader.setting import cfg_data
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
from config import cfg
class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, cfg,pwd):
        # Set the folder to save the records and checkpoints
        # Set cfg to be shareable in the class
        self.cfg_data = cfg_data
        self.pwd = pwd
        self.exp_path = cfg.EXP_PATH
        self.exp_name = cfg.EXP_NAME
        self.exp_path = osp.join(self.exp_path, 'pre')
        self.train_loader, self.val_loader,self.restore_transform = loading_data(cfg)

        self.model = NLT_Counter( mode='pre', backbone=cfg.model_type)
        if cfg.init_weights is not None:
            self.pretrained_dict = torch.load(cfg.init_weights)  # ['params']
            self.model.load_state_dict(self.pretrained_dict)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = cfg.pre_lr,weight_decay=cfg.pre_weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.pre_step_size,gamma=cfg.pre_gamma)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.record = {}
        self.record['train_loss'] = []
        self.record['train_mae'] = []
        self.record['train_mse'] = []

        self.record['val_loss'] = []
        self.record['val_mae'] = []
        self.record['val_mse'] = []
        

        self.record['best_mae'] = 1e10
        self.record['best_mse'] = 1e10
        self.record['best_model_name'] =''

        self.record['update_flag'] = 0

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ["exp"])
    def save_model(self, name):
        torch.save(dict(params=self.model.module.state_dict()), osp.join(self.exp_path,self.exp_name,name + '.pth'))

    def train(self):
        """The function for the pre_train on GCC dataset."""
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        for epoch in range(1, cfg.pre_max_epoch + 1):
            self.model.train()
            train_loss_avg = Averager()
            train_mae_avg = Averager()
            train_mse_avg = Averager()
                
            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                global_count = global_count + 1

                img   = batch[0].cuda()
                label = batch[1].cuda()

                pred = self.model(img)
                loss = F.mse_loss(pred.squeeze(), label)

                # Print loss and maeuracy for this step
                label_cnt = label.sum().data / self.cfg_data.LOG_PARA
                pred_cnt =  pred.sum().data / self.cfg_data.LOG_PARA
                mae = torch.abs(label_cnt-pred_cnt).item()
                mse = (label_cnt - pred_cnt).pow(2).item()

                tqdm_gen.set_description('Epoch {}, Loss={:.4f} gt={:.1f} pred={:.1f} lr={:.4f}'.format(epoch, loss.item(),label_cnt ,pred_cnt, self.optimizer.param_groups[0]['lr']*10000))
            #     # Add loss and maeuracy for the averagers
                train_loss_avg.add(loss.item())
                train_mae_avg.add(mae)
                train_mse_avg.add(mse)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_avg = train_loss_avg.item()
            train_mae_avg = train_mae_avg.item()
            train_mse_avg = np.sqrt(train_mse_avg.item())

            self.writer.add_scalar('data/loss',train_loss_avg, global_count)
            self.writer.add_scalar('data/mae', train_mae_avg, global_count)
            self.writer.add_scalar('data/mse', train_mse_avg, global_count)
            # Start validation for this epoch, set model to eval mode

            self.model.eval()
            val_loss_avg = Averager()
            val_mae_avg = Averager()
            val_mse_avg = Averager()

            # Print previous information  
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val mae={:.2f} mae={:.2f}'.format(self.record['best_model_name'], self.record['best_mae'],self.record['best_mse']))
            # Run validation
            for i, batch in enumerate(self.val_loader, 1):
                # print(i)
                with torch.no_grad():
                    data = batch[0].cuda()
                    label = batch[1].cuda()
                    pred = self.model(inp=data)
                    loss = F.mse_loss(pred.squeeze(), label)
                    val_loss_avg.add(loss.item())

                    for img in range(pred.size()[0]):
                        pred_cnt = (pred[img] / self.cfg_data.LOG_PARA).sum().data
                        gt_cnt   = (label[img] / self.cfg_data.LOG_PARA).sum().data
                        mae = torch.abs(pred_cnt - gt_cnt).item()
                        mse = (pred_cnt - gt_cnt).pow(2).item()
                        val_mae_avg.add(mae)
                        val_mse_avg.add(mse)

            # Update validation averagers
            val_loss_avg = val_loss_avg.item()
            val_mae_avg = val_mae_avg.item()
            val_mse_avg = np.sqrt(val_mse_avg.item())
            
            self.writer.add_scalar('data/val_loss', float(val_loss_avg), epoch)
            self.writer.add_scalar('data/val_mae',  float(val_mae_avg), epoch)
            self.writer.add_scalar('data/val_mse',  float(val_mse_avg), epoch)
            # Print loss and maeuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} mae={:.4f}  mse={:.4f}'.format(epoch, val_loss_avg, val_mae_avg,val_mse_avg))


            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch)+'_'+str(val_mae_avg))

            # Update the logs
            self.record['train_loss'].append(train_loss_avg)
            self.record['train_mae'].append(train_mae_avg)
            self.record['train_mse'].append(train_mse_avg)

            self.record['val_loss'].append(val_loss_avg)
            self.record['val_mae'].append(val_mae_avg)

            self.record = update_model(
                self.model.module, epoch, self.exp_path, self.exp_name, [val_mae_avg, val_mse_avg, val_loss_avg], self.record,
                self.log_txt)

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / cfg.max_epoch)))
        self.lr_scheduler.step()
        self.writer.close()

def vis_results(writer, restore, img, pred_map, gt_map):

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
    writer.add_image('temp', x)


if __name__ == '__main__':
    import torch
    # Set manual seed for PyTorch
    if cfg.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pwd = os.path.split(os.path.realpath(__file__))[0]
    if cfg.phase == 'pre_train':
        trainer = PreTrainer(cfg,pwd)
        trainer.train()
    else:
        raise ValueError('Please set correct phase.')

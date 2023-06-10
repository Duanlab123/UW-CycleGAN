import os
import os.path
from os import path
import matplotlib
matplotlib.use('Agg')
import random
import numpy as np
import copy
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from PIL import Image
from utils.networks3 import Discriminator, BetaNet, TransNet, HazeProduceNet, DepthEstimationNet, Lightnet
#from utils.networks5 import BetaNet, Transnet,Lightnet,Depthnet, Cleannet,HazeProduceNet,BasicBlock, BottleNeck

from utils.dataload2 import Dataset
from utils.loss import AdversarialLoss
import scipy.io  
from matplotlib import pyplot as plt
# import pdb
from utils.general import (LOGGER, increment_path, check_suffix)
from depth_estimate.model.pytorch_DIW_scratch import pytorch_DIW_scratch
import pathlib
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class cyclegan:
    def __init__(self, opt):
        self.opt = opt
        self.train_result = Path(self.opt.save_dir) / 'weights'  # weights dir save_dir = ROOT/runs/train/exp/weights
        pathlib.Path(self.train_result).mkdir(parents=True, exist_ok=True)

        self.device = opt.DEVICE

        self.loss = []
        self.loss_path = self.train_result / 'loss_uw.txt'
        self.epoch_path = self.train_result / 'epoch_loss.txt'

    def train_model(self):

        # check_suffix(opt.weights_A, '.pt')  # check weights
        # pretrained = str(opt.weights_A).endswith('.pt')
        pretrained = path.exists(opt.weights_A)
        if pretrained:
            self.model_t = torch.load(opt.weights_t, map_location=self.device)
            self.model_b = torch.load(opt.weights_b, map_location=self.device)
            self.model_A = torch.load(opt.weights_A, map_location=self.device)
            self.model_d = torch.load(opt.weights_d, map_location=self.device)

        else:
            self.model_t = TransNet(32, init_weights=False, use_pretrained=False).to(self.device)
            self.model_b = BetaNet(32, init_weights=False, use_pretrained=False).to(self.device)
            self.model_A = Lightnet(32, init_weights=False, use_pretrained=False).to(self.device)
            self.model_d = DepthEstimationNet(32, init_weights=False, use_pretrained=False).to(self.device)
        
        self.model_c2u = HazeProduceNet(base_channel_nums=48).to(self.device)
        self.discriminator_u2c = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
        self.discriminator_c2u = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
        
        self.depth = pytorch_DIW_scratch
        self.depth = torch.nn.parallel.DataParallel(self.depth, device_ids=[0])
        model_parameters = torch.load('./depth_estimate/checkpoints/test_local/best_generalization_net_G.pth')
        self.depth.load_state_dict(model_parameters)
        self.depth.eval()
        
        self.model_c2u.cuda()
        self.model_t.cuda()
        self.model_b.cuda()
        self.model_A.cuda()
        self.model_d.cuda()
        self.depth.cuda()
        self.discriminator_u2c.cuda()
        self.discriminator_c2u.cuda()
        
        self.optimizer_t = optim.Adam(self.model_t.parameters(), lr=float(opt.LR_J), betas=(0.9, 0.999))
        self.optimizer_b = optim.Adam(self.model_b.parameters(), lr=float(opt.LR_J), betas=(0.9, 0.999))
        self.optimizer_A = optim.Adam(self.model_A.parameters(), lr=float(opt.LR_J), betas=(0.9, 0.999))
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=float(opt.LR_J), betas=(0.9, 0.999))
        self.optimizer_dis_u2c = optim.Adam(self.discriminator_u2c.parameters(), lr=float(0.1*opt.LR_J), betas=(0.9, 0.999))
        self.optimizer_dis_c2u = optim.Adam(self.discriminator_c2u.parameters(), lr=float(0.1*opt.LR_J), betas=(0.9, 0.999))

        logs = [("epoch"), ("beta_loss"), ("trans_loss"), ("light_loss"), ("depth_loss"), ("total"), ("psnr"), ("mse")]
        with open(self.loss_path, 'a') as f:
            f.write('%s\n' % ' '.join([str(item) for item in logs]))
            f.write('\r\n')

        logs = [("epoch"), ("beta_loss"), ("trans_loss"), ("light_loss"), ("depth_loss"), ("cycle_loss"), ("total"), ("psnr"), ("mse")]
        with open(self.epoch_path, 'a') as f:
            f.write('%s\n' % ' '.join([str(item) for item in logs]))
            f.write('\r\n')

        loss_epoch = []
        for epoch in range(opt.strat_epoch, opt.epochs):
            epoch_stage = 50
            if epoch % epoch_stage == 0:
                self.optimizer_t, _ = exp_lr_scheduler(self.optimizer_t, epoch, lr_decay_epoch=epoch_stage)
                self.optimizer_b, _ = exp_lr_scheduler(self.optimizer_b, epoch, lr_decay_epoch=epoch_stage)
                self.optimizer_A, _ = exp_lr_scheduler(self.optimizer_A, epoch, lr_decay_epoch=epoch_stage)
                self.optimizer_d, lr = exp_lr_scheduler(self.optimizer_d, epoch, lr_decay_epoch=epoch_stage)
                self.optimizer_dis_u2c, lr = exp_lr_scheduler(self.optimizer_dis_u2c, epoch, lr_decay_epoch=epoch_stage)
                self.optimizer_dis_c2u, lr = exp_lr_scheduler(self.optimizer_dis_c2u, epoch, lr_decay_epoch=epoch_stage)
                print(f'epoch {epoch}, lr {lr}')

            self.train_dataset = Dataset(crop_size=opt.img_size, clean_file=opt.clean_dir, depth_file=opt.depth_dir, underwater_file=opt.underwater_dir, reference_file=opt.underwater_dir_refer, device=self.device, split='unpair')
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.opt.batch_size, num_workers=0, drop_last=True, shuffle=False, collate_fn=Dataset.collate_fn)

            for clean, depth, under, under_refer in self.train_loader:

                clean = clean.to(self.device, non_blocking=True).float()/255
                depth = depth.to(self.device, non_blocking=True).float()
                under = under.to(self.device, non_blocking=True).float()/255
                under_refer = under_refer.to(self.device, non_blocking=True).float()/255
                clean_images, hazy_c, clean_cp, depth_gt, depth_cm, depth_cp, beta_c, beta_gt, t_gt, t_c, \
                B_gt, A_c, hazy_images, clean_hp, hazy_h, depth_hm, depth_hp, t_h, A_h, loss, logs = self.train_process(epoch, clean, depth, under, under_refer)

                psnr = self.psnr(self.postprocess(under_refer), self.postprocess(clean_hp))
                psnr = round( psnr.item(), 4)
                mse = torch.mean((under_refer.float()-clean_hp.float())**2)
                mse = round( mse.item(), 4)
                logs.append( ('psnr', psnr) )
                logs.append( ('mse', mse) )
                with open(self.loss_path, 'a') as f:
                    f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
                loss_epoch.append(loss)

            if epoch % 1 == 0:

                self.save_result('c2u2c', epoch, 'clean--->underwater--->clean_p', clean_images, hazy_c, clean_cp)
                self.save_depth('c2u_p', epoch, 'd-refer--->d-phy--->d-cyc', depth_gt, depth_cp, depth_cm)
                self.save_result('c2u2c-t', epoch, f't_gt--->t_c--->underwater-->beta{beta_c.data}-->betagt{beta_gt.data}', t_gt, t_c, hazy_c)
                self.save_result('c2u2c-A', epoch, 'B_gt--->A_c--->clean', B_gt, A_c, hazy_c)

                self.save_result('u2c2u', epoch, 'underwater--->clean_p--->underwater', hazy_images, clean_hp, hazy_h)
                self.save_depth('u2c_pm', epoch, 'd-refer--->d-phy--->d-cyc', hazy_images, depth_hp, depth_hm)
                self.save_result('u2c2u-t', epoch, 't_h--->A_h--->clean_m', t_h, A_h, hazy_images)

                hazy_c = self.postprocess(hazy_c)[0]
                self.imsave(hazy_c, os.path.join(self.train_result, f'{epoch}_fakeuw.png'))

            if epoch % 2 == 0:

                epoch_loss = sum(loss_epoch) / len(loss_epoch)
                logs.append( ('mean loss', round(epoch_loss.item(), 4) ) )  
                with open(self.epoch_path, 'a') as f:
                    f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
                    f.write('\r\n')
                print('epoch:', epoch, 'current loss =', epoch_loss.item())   # self. epoch_loss[1]


            if epoch % 100 == 0:
                save_model = self.train_result / f'model_t {epoch}.pt'
                epoch_model = copy.deepcopy(self.model_t)
                torch.save(epoch_model.cuda(), save_model)

                save_model = self.train_result / f'model_b {epoch}.pt'
                epoch_model = copy.deepcopy(self.model_b)
                torch.save(epoch_model.cuda(), save_model)
                
                save_model = self.train_result / f'model_A {epoch}.pt'
                epoch_model = copy.deepcopy(self.model_A)
                torch.save(epoch_model.cuda(), save_model)

                save_model = self.train_result / f'model_d {epoch}.pt'
                epoch_model = copy.deepcopy(self.model_d)
                torch.save(epoch_model.cuda(), save_model)


    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


    def train_process(self, epoch, clean_images, depth_input, hazy_images, refer):
        self.model_t.train()
        self.model_b.train()
        self.model_A.train()
        self.model_d.train()
        self.discriminator_c2u.train()
        self.discriminator_u2c.train()
        # ----------------------------
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type='lsgan').to(self.device)
        self.epoch = epoch
        # --------------------------clean-->underwater-->clean--------------------------------------
        hazy_c, beta_gt, B_gt, t_gt, depth_gt = self.model_c2u.forward_random_parameters(clean_images, depth_input)  #
        A_c = self.model_A.forward(hazy_c)
        # t_c, beta_c, depth_cp = self.model_tb.forward(hazy_c)  # I, eta, B,t
        t_c = self.model_t.forward(hazy_c)  # I, eta, B,t
        beta_c, depth_cp = self.model_b.forward(hazy_c, t_c)  # I, eta, B,t
        clean_cp = ((hazy_c-A_c)/t_c + A_c).clamp(0, 1)
        #----------------------------------------------------
        """
        self.optimizer_dis_c2u.zero_grad()
        dis_real_haze, _ = self.discriminator_c2u((hazy_images))  # underwater--> clean
        dis_fake_haze, _ =  self.discriminator_c2u(hazy_c.detach())
        dis_haze_fake_loss =  self.adversarial_loss((dis_fake_haze), is_real=False, is_disc=True)
        dis_haze_real_loss = self.adversarial_loss((dis_real_haze), is_real=True, is_disc=True)
        dis_haze_loss = (dis_haze_real_loss + dis_haze_fake_loss) / 2
        dis_haze_loss.backward()
        self.optimizer_dis_c2u.step()
        """
        #----------------------------------------------------
        self.optimizer_t.zero_grad()
        self.optimizer_b.zero_grad()
        self.optimizer_A.zero_grad()
        cycle_loss_c2u2c = self.l1_loss(clean_images, clean_cp)
        light_loss = self.l1_loss(B_gt, A_c)
        tran_loss = self.l1_loss(t_gt, t_c)
        beta_loss = self.l1_loss(beta_c, beta_gt)
        phy = 0.01*self.l1_loss(depth_cp, depth_gt)
        loss = cycle_loss_c2u2c + light_loss + tran_loss + beta_loss + phy
        loss.backward()
        self.optimizer_t.step()
        self.optimizer_b.step()
        self.optimizer_A.step()

        self.optimizer_d.zero_grad()
        depth_cm = self.model_d.forward(clean_cp.detach())
        depth_loss_c2u2c = self.l1_loss(depth_cm, depth_gt)
        depth_loss_c2u2c.backward()
        self.optimizer_d.step()

        # ------------------------------------------underwater-->clean-->underwater----------------------------------------------
        # ----(1) [clean,depth,beta]=u2c(haze)-->(2)depth2=depth(clean)-->(3)fake_u=c2u(clean,depth2,beta)----------
        A_h = self.model_A.forward(hazy_images)
        # t_h, beta_h, depth_hp = self.model_tb.forward(hazy_images)
        t_h = self.model_t.forward(hazy_images)
        beta_h, depth_hp = self.model_b.forward(hazy_images, t_h)
        clean_hp = ((hazy_images - A_h) / t_h + A_h).clamp(0, 1)
        depth_hm = self.model_d.forward(clean_hp)
        with torch.no_grad():
            depth_hrefer = self.depth_estimate(clean_hp.detach())
        t = torch.exp(-depth_hm*beta_h)
        hazy_h = (clean_hp * t + A_h * (1 - t)).clamp(-1, 1)
        # ===============================================================================

        self.optimizer_dis_u2c.zero_grad()
        dis_real_clean, _ = self.discriminator_u2c(clean_images)  # clean-->underwater
        dis_fake_clean, _ = self.discriminator_u2c(clean_hp.detach())
        dis_clean_real_loss = self.adversarial_loss((dis_real_clean), is_real=True, is_disc=True)
        dis_clean_fake_loss = self.adversarial_loss((dis_fake_clean), is_real=False, is_disc=True)
        dis_clean_loss = (dis_clean_real_loss + dis_clean_fake_loss) / 2
        dis_clean_loss.backward()
        self.optimizer_dis_u2c.step()
        # ------------total loss------------------------
        self.optimizer_t.zero_grad()
        self.optimizer_b.zero_grad()
        self.optimizer_A.zero_grad()
        gen_fake_clean, _ = self.discriminator_u2c(clean_hp)
        gen_fake_clean_ganloss = 0.5*self.adversarial_loss((gen_fake_clean), is_real=True, is_disc=False)
        cycle_loss_u2c2u = self.l1_loss(hazy_images, hazy_h)
        clean_loss = self.l1_loss(refer, clean_hp)
        loss_un = 0.5*cycle_loss_u2c2u + 0.2*gen_fake_clean_ganloss + 0.5*clean_loss
        loss_un.backward()
        self.optimizer_t.step()
        self.optimizer_b.step()
        self.optimizer_A.step()

        self.optimizer_d.zero_grad()
        depth_hm = self.model_d.forward(clean_hp.detach())
        depth_lossu = self.l1_loss(depth_hp.detach(), depth_hm) + self.l1_loss(depth_hrefer, depth_hm)
        depth_lossu.backward()
        self.optimizer_d.step()
        """
        self.optimizer_tb.zero_grad()
        self.optimizer_A.zero_grad()
        cycle_loss_c2u2c = self.l1_loss(clean_images, clean_cp)
        cycle_loss_u2c2u = self.l1_loss(hazy_images, hazy_h)
        cycle_loss = cycle_loss_c2u2c + cycle_loss_u2c2u
        light_loss = self.l1_loss(B_gt, A_c)
        tran_loss = self.l1_loss(t_gt, t_c)
        beta_loss = self.l1_loss(beta_c, beta_gt)
        phy = 0.01*self.l1_loss(depth_cp, depth_gt)
        loss = cycle_loss + light_loss + tran_loss + beta_loss + phy
        loss.backward()
        self.optimizer_tb.step()
        self.optimizer_A.step()

        self.optimizer_d.zero_grad()
        depth_cm = self.model_d.forward(clean_cp.detach())
        depth_loss_c2u2c = self.l1_loss(depth_cm, depth_gt)
        depth_loss_c2u2c.backward()
        self.optimizer_d.step()
        """
        # ------------clean--->underwater----->clean------------------------
        logs = [
            ("iter", epoch),
            ("beta_loss", round(beta_loss.item(), 4)),
            ("trans_loss", round(tran_loss.item(), 4)),
            ("light_loss", round(light_loss.item(), 4)),
            ("depth_loss", round(depth_loss_c2u2c.item(), 4)),
            ("cycle_loss", round(cycle_loss_u2c2u.item(), 4)),
            ("total", round(loss.item(), 4)),
        ]

        return clean_images, hazy_c, clean_cp, depth_gt, depth_cm, depth_cp, beta_c, beta_gt, t_gt, t_c, \
                B_gt, A_c, hazy_images, clean_hp, hazy_h, depth_hm, depth_hp, t_h, A_h, loss, logs

    def depth_estimate(self, clean1):
        self.depth.eval()
        depth_hm = self.depth.forward(clean1)
        depth_hm = torch.exp(depth_hm)
 
        depth_hm = torch.div(1, depth_hm)
        return depth_hm

    def postprocess(self, img, size=None):
        # [0, 1] => [0, 255]
        if size is not None:
            img = torch.nn.functional.interpolate(img, size, mode='bicubic')
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def imsave(self, img, path):
        im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
        im.save(path)

    def psnr(self, a, b):
        mse = torch.mean((a.float()-b.float())**2)
        if mse == 0:
            return torch.tensor(0)
        psnr = 10*torch.log10(255*255 / mse)
        return psnr

    def save_result(self, mod, epoch, title, outputs, x_qry, y_qry):
        # -------------------------------end finetunning and save some results----------------------------------------

        temp_out = outputs
        temp_input = x_qry
        temp_label = y_qry

        temp_out = temp_out.detach().cpu().numpy()
        temp_input = temp_input.detach().cpu().numpy()
        temp_label = temp_label.detach().cpu().numpy()
        # psnr1 = self.calculate_psnr(temp_out[0], temp_label[0])
        num_img = 5 if len(temp_out) > 6 else len(temp_out)
        fig, ax = plt.subplots(num_img, 3, figsize=(6, 6))
        if len(temp_out) == 1:
            [ax[0].imshow(temp_out[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_out[0: num_img])]
            [ax[1].imshow(temp_input[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_input[0: num_img])]
            [ax[2].imshow(temp_label[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_label[0: num_img])]
        else:
            for i, _ in enumerate(temp_out[0: num_img]):
                ax[i][0].imshow(temp_out[i].transpose((1, 2, 0)))
                ax[i][0].axis('off')

            for i, _ in enumerate(temp_input[0: num_img]):
                ax[i][1].imshow(temp_input[i].transpose((1, 2, 0)))
                ax[i][1].axis('off')

            for i, _ in enumerate(temp_label[0: num_img]):
                ax[i][2].imshow(temp_label[i].transpose((1, 2, 0)))
                ax[i][2].axis('off')

        f = self.train_result / f'{mod}_batch{epoch}_labels.png'
        plt.title(title, x=-1.4, y=-0.6)
        plt.savefig(f)
        plt.close()
        
    def save_depth(self, mod, epoch, title, outputs, x_qry, y_qry):
        # -------------------------------end finetunning and save some results----------------------------------------
        temp_out = outputs
        temp_input = x_qry
        temp_label = y_qry

        temp_out = temp_out.detach().cpu().numpy()
        temp_label = temp_label.detach().cpu().numpy()
        temp_input = temp_input.detach().cpu().numpy()
        num_img = 5 if len(temp_out) > 6 else len(temp_out)
        fig, ax = plt.subplots(num_img, 3, figsize=(6, 6))
        if len(temp_out) == 1:
            [ax[0].imshow(temp_out[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_out[0: num_img])]
            [ax[1].imshow(temp_input[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_input[0: num_img])]
            [ax[2].imshow(temp_label[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_label[0: num_img])]
        else:
            for i, _ in enumerate(temp_out[0: num_img]):
                ax[i][0].imshow(temp_out[i].transpose((1, 2, 0)))
                ax[i][0].axis('off')

            for i, _ in enumerate(temp_input[0: num_img]):
                ax[i][1].imshow(temp_input[i].transpose((1, 2, 0)))
                ax[i][1].axis('off')

            for i, _ in enumerate(temp_label[0: num_img]):
                ax[i][2].imshow(temp_label[i].transpose((1, 2, 0)))
                ax[i][2].axis('off')

        f = self.train_result / f'{mod}_batch{epoch}_depth.png'
        plt.title(title)
        plt.savefig(f)
        plt.close()


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=260):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.9 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print(f'learning rate', param_group['lr'])
    return optimizer, param_group['lr']


def parse_opt(known=False):
    # opt.save_dir , opt.epochs, opt.batch_size, opt.weights, opt.task_num, opt.noise_num1, opt.path, opt.evolve
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2001)
    parser.add_argument('--strat_epoch', type=int, default=0)
    parser.add_argument('--clean-dir', default='./dataset/air_image', help='dir of dataset')
    parser.add_argument('--depth-dir', default='./dataset/air_depth', help='dir of dataset')
    parser.add_argument('--underwater-dir', default='./uw-cyclegan/dataset/underwater_img', help='dir of dataset')
    parser.add_argument('--underwater-dir-refer', default='./uw-cyclegan/dataset/underwater_refer', help='dir of dataset')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs, -1 for auto-batch')
    parser.add_argument('--weights_c2u', type=str, default=ROOT / 'pretrained_model', help='initial weights path')
    
    parser.add_argument('--weights_A', type=str, default=None, help='initial weights path')
    parser.add_argument('--weights_d', type=str, default=None, help='initial weights path')
    parser.add_argument('--weights_t', type=str, default=None, help='initial weights path')
    parser.add_argument('--weights_b', type=str, default=None, help='initial weights path')
    
    parser.add_argument('--PSNR', default='RGB', help='psnr')
    parser.add_argument('--LR-J', default=1e-4, help='learning rate')
    parser.add_argument('--LR-D', default=1e-4, help='learning rate')
    parser.add_argument('--LR-R', default=1e-4, help='learning rate')
    parser.add_argument('--D2G_LR', default='0.1', help='discriminator/generator learning rate ratio')
    parser.add_argument('--img_size', '--img', '--img-size', type=int, default=256, help='size (pixels) of train, val image')
    parser.add_argument('--device', default='gpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--GPU', type=list, default=[1], help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp1', help='save to project/name')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(Path(opt.project) / opt.name)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in opt.GPU)

    # init device
    if torch.cuda.is_available():
        opt.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        opt.DEVICE = torch.device("cpu")

    cycle = cyclegan(opt)
    cycle.train_model()


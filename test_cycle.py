import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from utils.networks2 import Discriminator, BetaTrans, HazeProduceNet, DepthEstimationNet, Lightnet
# from utils.networks5 import BetaNet, Transnet,Lightnet,Depthnet, Cleannet,HazeProduceNet,BasicBlock, BottleNeck
from utils.dataload2_test import Dataset
import pathlib
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class cyclegan:
    def __init__(self, opt):
        self.opt = opt
        print(self.opt.save_dir)
        self.test_result = Path(self.opt.save_dir) / 'weights'  # weights dir save_dir = ROOT/runs/train/exp/weights
        pathlib.Path(self.test_result).mkdir(parents=True, exist_ok=True)

        self.device = opt.DEVICE

        self.loss_path = self.test_result / 'metrics.txt'

    def test_model(self):

        self.model_c2u = HazeProduceNet(base_channel_nums=48).to(self.device)
        self.model_t = torch.load(opt.weights_t)
        self.model_b = torch.load(opt.weights_b)
        self.model_A = torch.load(opt.weights_A)
        self.model_d = torch.load(opt.weights_d)
        #self.model_tb =BetaTrans(32,init_weights=False, use_pretrained=False).to(self.device)
        #self.model_A = Lightnet(32, init_weights=False, use_pretrained=False).to(self.device)
        #self.model_d = DepthEstimationNet(32, init_weights=False, use_pretrained=False).to(self.device)
        self.model_c2u.cuda()
        self.model_t.cuda()
        self.model_b.cuda()
        self.model_A.cuda()
        self.model_d.cuda()

        self.test_dataset = Dataset(crop_size=opt.img_size, clean_file=opt.clean_dir, depth_file=opt.depth_dir, underwater_file=opt.underwater_dir, reference_file=opt.underwater_dir_refer, device=self.device, split='unpair')
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.batch_size, num_workers=0, drop_last=True, shuffle=False, collate_fn=Dataset.collate_fn)

        epoch = 0

        for clean, depth, under, under_refer in self.test_loader:

            name = self.test_dataset.load_name(epoch)[:-4]
            # wholename = name + '.png'

            clean = clean.to(self.device, non_blocking=True).float()/255
            depth = depth.to(self.device, non_blocking=True).float()
            under = under.to(self.device, non_blocking=True).float()/255
            under_refer = under_refer.to(self.device, non_blocking=True).float()/255

            hazy_images, clean_hp, hazy_h, depth_hm, depth_hp, t_h, A_h = self.test_process(under)

            under_p = self.postprocess(under)[0]
            refer_p = self.postprocess(under_refer)[0]
            result_p = self.postprocess(clean_hp)[0]

            path = self.test_result
            # save_name = os.path.join(path, wholename)
            # print('predicted_results shape:', result_p.shape, type(result_p))
            self.imsave( result_p, os.path.join(path, name+'_ours.png') )
            self.imsave( under_p, os.path.join(path, name+'_under.png') )
            self.imsave( refer_p, os.path.join(path, name+'_gt.png') )
            t_pred = self.postprocess(t_h)[0]
            A_pred = self.postprocess(A_h)[0]
            self.imsave(t_pred, os.path.join(path, name+'_t.png'))
            self.imsave(A_pred, os.path.join(path, name+'_A.png'))

            epoch += 1

            self.save_result(epoch, name, hazy_images, clean_hp, under_refer)
            self.save_result(epoch, name+'_tA', t_h, A_h, hazy_images)

    def test_process(self, hazy_images):
        self.model_t.eval()
        self.model_b.eval()
        self.model_A.eval()
        self.model_d.eval()

        A_h = self.model_A.forward(hazy_images)
        t_h = self.model_t.forward(hazy_images)
        beta_h, depth_hp = self.model_b.forward(hazy_images, t_h)
        clean_hp = ((hazy_images - A_h) / t_h + A_h).clamp(0, 1)

        depth_hm = self.model_d.forward(clean_hp)
        t = torch.exp(-depth_hm*beta_h)
        hazy_h = (clean_hp * t + A_h * (1 - t)).clamp(-1, 1)

        return hazy_images, clean_hp, hazy_h, depth_hm, depth_hp, t_h, A_h

    def postprocess(self, img, size=None):
        # [0, 1] => [0, 255]
        if size is not None:
            img = torch.nn.functional.interpolate(img, size, mode='bicubic')
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def psnr(self, a, b):
        mse = torch.mean((a.float()-b.float())**2)
        if mse == 0:
            return torch.tensor(0)
        psnr = 10*torch.log10(255*255 / mse)
        return psnr, mse

    def imsave(self, img, path):
        im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
        im.save(path)

    def save_result(self, epoch, name, outputs, x_qry, y_qry):
        # -------------------------------end finetunning and save some results----------------------------------------

        temp_out = outputs
        temp_input = x_qry
        temp_label = y_qry

        temp_out = temp_out.detach().cpu().numpy()
        temp_label = temp_label.detach().cpu().numpy()
        temp_input = temp_input.detach().cpu().numpy()
        # psnr = self.calculate_psnr(temp_out[0], temp_label[0])

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

        f = self.test_result / f'{name}.png'
        plt.title(name, x=-1.4, y=-0.6)
        plt.savefig(f)
        plt.close()

    def save_depth(self, epoch, name, outputs, x_qry, y_qry):
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

        f = self.test_result / f'{name}_depth.png'
        plt.title(name)
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

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--weights-t', default='./model/model_t.pt', help='dir of dataset')
    parser.add_argument('--weights-b', default='./model/model_b.pt', help='dir of dataset')
    parser.add_argument('--weights-A', default='./model/model_A.pt', help='dir of dataset')
    parser.add_argument('--weights-d', default='./model/model_d.pt', help='dir of dataset')
    parser.add_argument('--clean-dir', default='uw-cyclegan/test_img/data1/img', help='dir of dataset')
    parser.add_argument('--depth-dir', default='uw-cyclegan/test_img/data1/img', help='dir of dataset')
    parser.add_argument('--underwater-dir', default='uw-cyclegan/test_img/data1/img', help='dir of dataset')
    parser.add_argument('--underwater-dir-refer', default='uw-cyclegan/test_img/data1/refer', help='dir of dataset')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for auto-batch')
    parser.add_argument('--PSNR', default='RGB', help='psnr')
    parser.add_argument('--img_size', '--img', '--img-size', type=int, default=256,
                        help='size (pixels) of train, val image')
    parser.add_argument('--device', default='gpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--GPU', type=list, default=[1], help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--project', default=ROOT / 'test_result', help='save to project/name')
    parser.add_argument('--name', default='test01', help='save to project/name')
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
    cycle.test_model()
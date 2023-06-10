import os
from scipy.io import loadmat
import cv2
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from matplotlib import pyplot as plt
from pathlib import Path
from depth_estimate.model.pytorch_DIW_scratch import pytorch_DIW_scratch
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(1, os.cpu_count())  # number of multiprocessing threads
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, crop_size, clean_file, depth_file, underwater_file, reference_file, device, split='unpair'):

        self.clean_data = self.load_flist(clean_file)
        self.under_data = self.load_flist(underwater_file)
        random.shuffle(self.under_data)
        random.shuffle(self.clean_data)
        
        self.depth_data = self.img2mat(self.clean_data, depth_file)
        self.under_refer = self.img2img(self.under_data, reference_file)

        self.input_size = crop_size
        self.split = split
        self.device = device
        print(self.under_data[0])

        self.depth = pytorch_DIW_scratch
        self.depth = torch.nn.parallel.DataParallel(self.depth, device_ids=[0])
        model_parameters = torch.load('./depth_estimate/checkpoints/test_local/best_generalization_net_G.pth')
        self.depth.load_state_dict(model_parameters)


    def __len__(self):
        return len(self.clean_data) #if len(self.clean_data)>len(self.under_data) else len(self.under_data)


    def __getitem__(self, index):
        clean, depth, under, under_refer = self.load_item(index)
        return clean, depth, under, under_refer


    def load_item(self, index):
        #while(True):
        #   clean_index = int(np.random.random() * len(self.clean_data))
        clean = cv2.imread(self.clean_data[index])
            #if min(clean.shape[0:2])>=self.input_size:
            #    break
        while(True):
            under_index = int(np.random.random() * len(self.under_data))
            under = cv2.imread(self.under_data[under_index])
            if min(under.shape[0:2]) >= self.input_size:
                break

        if os.path.exists(self.under_refer[under_index]):
            under_refer = cv2.imread(self.under_refer[under_index])
        else:
            under_refer = under

        clean, xc, yc = self.get_square_img(clean)
        under, xb, yb = self.get_square_img(under)
        under_refer = self.get_square_refer(under_refer, xb, yb)

        clean = cv2.resize(clean, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        under = cv2.resize(under, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        under_refer = cv2.resize(under_refer, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)

        if os.path.exists(self.depth_data[index]):
            depth = np.array(loadmat(self.depth_data[index])['dph']).astype(np.float32)
            depth = self.get_square_refer(depth, xc, yc)
            depth = cv2.resize(depth, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        else:
            depth = self.depth_estimate(clean)

        # fig, ax = plt.subplots(1, 4, figsize=(6, 6))
        # ax[0].imshow(clean[:, :, ::-1])
        # ax[1].imshow(under[:, :, ::-1])
        # ax[2].imshow(under_refer[:, :, ::-1])
        # ax[3].imshow(depth)
        # plt.show()

        clean = np.ascontiguousarray(clean[:, :, ::-1].transpose((2, 0, 1)))
        depth = np.ascontiguousarray( np.expand_dims(depth, axis= 0) )
        under = np.ascontiguousarray( under[:, :, ::-1].transpose((2, 0, 1)))
        under_refer = np.ascontiguousarray( under_refer[:, :, ::-1].transpose((2, 0, 1)))

        return torch.from_numpy(clean), torch.from_numpy(depth), torch.from_numpy(under), torch.from_numpy(under_refer)

    @staticmethod
    def collate_fn(batch):
        clean, depth, under, under_refer = zip(*batch)  # transposed
        return torch.stack(clean, 0), torch.stack(depth, 0), torch.stack(under, 0), torch.stack(under_refer, 0)

    def depth_estimate(self, clean):
        clean1 = np.ascontiguousarray(clean[:, :, ::-1].transpose((2, 0, 1)))
        clean1 = torch.from_numpy(clean1).to(self.device, non_blocking=True).float() / 255
        clean1 = clean1.unsqueeze(0)
        self.depth.eval()
        with torch.no_grad():
            depth_hm = self.depth.forward(clean1)
            depth_hm = torch.exp(depth_hm)
            depth_hm = 1 / depth_hm
            #depth_hm1 = depth_hm1.data.cpu().numpy()
            depth_hm = depth_hm.data.cpu().numpy()
            depth_hm = np.squeeze(depth_hm)#+depth_hm1.max()

        return depth_hm

    def get_square_img(self, img):
        h, w = img.shape[0:2]
        if h < w:
            x_b = random.randint(0, w- h)
            img = img[0:h, x_b:x_b+h, :]
            y_b = 0
        elif h >= w:
            y_b = random.randint(0, h-w)
            img = img[y_b:y_b+h, 0:h, :]
            x_b = 0
        return img, x_b, y_b

    def get_square_refer(self, img, x_b, y_b):
        h, w = img.shape[0:2]
        if h < w:
            img = img[0:h, x_b:x_b+h]
        elif h >= w:
            img = img[y_b:y_b+h, 0:h]
        return img

    def img2img(self, list, path):
        f = []
        f += [path + os.sep + str(Path(x).name) for x in list]
        return f

    def img2mat(self, list, path):
        f = []
        f += [path + os.sep + str(Path(x).stem) + '.mat' for x in list]
        return f

    def load_flist(self, flist):
        f = []
        path = flist
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            else:
                raise Exception(f'{p} does not exist')

        f2 = []
        f2 += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
        return f2


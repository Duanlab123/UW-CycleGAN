import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import  Image
import scipy.io as sio
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def get_art_light(x1, y1,eta,depth, sigma,x_c,y_c,L_art=0.8,r_l=0.3,Z_l=0.3):

    v = L_art * np.exp(-1.0 / (2 * sigma ** 2) * ( (x1-x_c) ** 2 + (y1-y_c) ** 2))
    v=np.expand_dims([v], axis=3)
    D= Z_l**2 +(0**2) * ((x1-x_c)**2+(y1-y_c)**2)
    D=np.expand_dims([np.sqrt(D)], axis=3)
    D_decay=np.exp(np.multiply(-1, np.multiply(D+depth,eta)))
    art_light = np.multiply(D_decay, v)
    return art_light,v

def wc_generator(image,depth,A=0.7,water_type=1,water_depth=0,water_illum=[1,0,0]):

    eta_r = np.array([0.30420412, 0.30474395, 0.35592191, 0.32493874, 0.55091001,0.42493874, 0.55874165, 0.13039252 , 0.10760831, 0.15963731])
    eta_g = np.array([0.11727661, 0.05999663, 0.11227639, 0.15305673, 0.14385827, 0.12305673, 0.0518615, 0.18667714, 0.1567016, 0.205724217])
    eta_b = np.array([0.1488851, 0.30099538, 0.38412464, 0.25060999, 0.01387215, 0.055799195, 0.0591001, 0.5539252 , 0.60103   , 0.733602  ])

    eta_rI,eta_gI,eta_bI=torch.tensor([[[eta_r[water_type]]]]), torch.tensor([[[eta_g[water_type]]]]), torch.tensor([[[eta_b[water_type]]]])
    eta1 = torch.stack([eta_rI, eta_gI, eta_bI],axis=3)
    water_type2= 8  if water_type+1==10 else  water_type+1
    eta_rI,eta_gI,eta_bI=torch.tensor([[[eta_r[water_type2]]]]), torch.tensor([[[eta_g[water_type2]]]]), torch.tensor([[[eta_b[water_type2]]]])
    eta2 = torch.stack([eta_rI, eta_gI, eta_bI],axis=3)
    weight =torch.from_numpy( np.random.uniform(0, 1, 1) )
    eta = weight*eta1 + (1-weight)*eta2
    print(f'attentuation: {eta}')

    depth =depth+water_depth
    eta =eta.detach().numpy()
    t = np.exp(np.multiply(-1, np.multiply(depth,eta)))
    w_a, w_b, w_c = water_illum[0], water_illum[1], water_illum[2],
    L_a, L_b, L_c = 0.9,1,1
    Z_b= np.random.uniform(0,2)
    L_t1 = w_a*L_a
    L_t2 = w_b*L_b*np.exp(np.multiply(-1, np.multiply(depth+Z_b,eta)))
    x,y=np.meshgrid(np.linspace(0,depth.shape[2]-1,depth.shape[2]),np.linspace(0,depth.shape[1]-1,depth.shape[1]))
    sigma,z_l,r_l=np.random.uniform(0.2,1.1)*t.shape[2], np.random.uniform(-1,1),np.random.uniform(0.1,1)
    L_t3, v= get_art_light(x, y,eta, depth, sigma,x_c=np.random.randint(0,x.shape[1]),y_c=np.random.randint(0,x.shape[0]),L_art=0.75,Z_l= z_l,r_l=r_l)
    L_t=L_t1+L_t2+w_c*L_t3 #L_t3
#        L_tt=L_t1+L_t2+w_c*v
    direct =np.multiply( np.multiply(image,L_t),t )
#        direct1 =np.multiply( np.multiply(image,L_tt),t )
    eta_rI1,eta_gI1,eta_bI1=torch.tensor([[[eta_r[water_type]]]]), torch.tensor([[[eta_g[water_type]]]]), torch.tensor([[[eta_b[water_type]]]])
    eta_haze = torch.stack([eta_rI1,eta_gI1,eta_bI1],axis=3)
    eta_haze =eta_haze.detach().numpy()
    t_haze = np.exp(np.multiply(-1, np.multiply(depth,eta_haze)))
    image_haze = np.multiply( np.multiply(A*255, np.subtract(1.0, t_haze)), t)
    B = np.multiply( A*255, t)
    I = direct +  image_haze

    """
    if self.batch_size == 1:
        fig, ax = plt.subplots(1, 8, figsize=(3, 6))
        ax[0].imshow(np.squeeze(depth[0]))
        ax[1].imshow(np.squeeze(L_t2[0]))
        ax[2].imshow(np.squeeze(L_t3[0]))
        ax[3].imshow(np.squeeze(v[0]))
        ax[4].imshow(np.squeeze(image[0]) / 255)
        ax[5].imshow(np.squeeze(image_haze[0]) / 255)
        ax[6].imshow(np.squeeze(direct[0] + 0) / 255)
        ax[7].imshow(np.squeeze(I[0] + 0) / 255)
    else:
        fig, ax = plt.subplots(4, image.shape[0], figsize=(3, 3 * (image.shape[0])))
        [ax[0][i].imshow(np.squeeze(image_haze[i])/255) for i in range(image.shape[0])]
        [ax[1][i].imshow(np.squeeze(direct[i])/255) for i in range(image.shape[0])]
        [ax[2][i].imshow(np.squeeze(image[i])/255) for i in range(image.shape[0])]
        [ax[3][i].imshow(np.squeeze(I[i])/255) for i in range(image.shape[0])]
    """
    return I, B, t













from PIL import Image
from torchvision.transforms import functional as F
import os
import glob
import numpy as np


def resize(origin_dir, result_dir):
    rlist = []
    if isinstance(origin_dir, str):
        if os.path.exists(origin_dir):
            plist = list(glob.glob(origin_dir + '/*.jpg')) + list(glob.glob(origin_dir + '/*.png')) + list(
                glob.glob(origin_dir + '/*.jpeg'))
            plist.sort()

            for i in range(len(plist)):
                img1 = Image.open(plist[i])
                ext = os.path.splitext(plist[i])[-1]
                # print(ext)
                h, w, _ = np.array(img1).shape
                img2 = img1

                if h < 256 and w >= 256:
                    img2 = F.resize(img1, [256, w])
                if w < 256 and h >= 256:
                    img2 = F.resize(img1, [h, 256])
                if h < 256 and w < 256:
                    img2 = F.resize(img1, [256, 256]) 

                rlist.append(result_dir + plist[i])

                filename = plist[i].split('/')[-1]
                print('filename: ', filename)
                savepath = result_dir + '/' + filename[:-4] + '.png'
                print('savepath: ', savepath)
                print(savepath)

                img2.save(savepath)

    return rlist


if __name__ == '__main__':
    od = './dataset/underwater_reference_1'
    rd = './dataset/underwater_reference'
    if not os.path.exists(rd):
        os.makedirs(rd)
    resize(od, rd)
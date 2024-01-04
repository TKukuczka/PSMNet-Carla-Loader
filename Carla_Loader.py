import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess
from . import listflowfile as lt
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path_left, path_right):                        
    left_img = Image.open(path_left).convert("RGB")
    right_img = Image.open(path_right).convert("RGB")
    return left_img, right_img

def depthmap_loader(path):
    depth=Image.open(path).convert("RGB")
    array = np.array(depth, dtype=np.float32)
    normalized_depth = np.dot(array[:, :, :], [1.0, 256.0, 65536.0])
    normalized_depth = normalized_depth / ((256.0 * 256.0 * 256.0) - 1)
    normalized_depth = normalized_depth * 1000
    
    return normalized_depth

def disparity_loader(path, fov=60,baseline=0.4, width=1937):
    normalized_depth=depthmap_loader(path)
    focal = width / (2.0 * np.tan((fov * np.pi) / 360.0))
    ref_disp = (focal * baseline) / normalized_depth
    idx = ref_disp < 0.5
    ref_disp[idx] = 0
    idx=ref_disp > 191
    ref_disp[idx] = 191
    return ref_disp, 1

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_depth, training,fov,baseline, loader=default_loader, dploader=disparity_loader, depth_loader=depthmap_loader):
        self.left = left
        self.right = right
        self.depth_L = left_depth
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.depth_loader=depthmap_loader
        self.fov=fov
        self.baseline=baseline

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        depth_L = self.depth_L[index]

        left_img, right_img = self.loader(left, right)
        #right_img = self.loader(right)
        dataL, scaleL = self.dploader(depth_L,fov=self.fov,baseline=self.baseline)
        depth=self.depth_loader(depth_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            
            
            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]
            depth = depth[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            
            return left_img, right_img, dataL, depth
        else:
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL, left, right, depth_L

    def __len__(self):
        return len(self.left)
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as trfs
from PIL import Image, ImageOps
import os
import utils

class TrainSet(Dataset):
    """
    parameter:
        img_ls: image list
        img_dir: input image path
        gt_dir: ground truth path
        patch_size: patch size
        patch_num: the number of extracted patches
    """
    def __init__(self, img_ls, img_dir, gt_dir, patch_size = 256, patch_num = 4):
        self.img_ls = img_ls
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.patch_num = patch_num
    
    def preprocess(self, pil_img, patch_size, patch_coords, flip_op):
        if flip_op is 1:
            pil_img = ImageOps.mirror(pil_img)
        elif flip_op is 2:
            pil_img = ImageOps.flip(pil_img)

        # patch extraction
        img_nd = np.array(pil_img, dtype=np.float32)
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index):
        img_name = self.img_ls[index] # input image name
        gt_name = utils.get_gt_name(img_name) # ground truth image name
        # read data
        img_data = Image.open(os.path.join(self.img_dir, img_name)) # read input image
        gt_data = Image.open(os.path.join(self.gt_dir, gt_name)) # read ground truth
        w, h = img_data.size
        flip_op = np.random.randint(3) # 0：no change；1：flip horizontal；2：flip upside down；
        # patch coordinates
        patch_x = np.random.randint(0, high= w - self.patch_size)
        patch_y = np.random.randint(0, high= h - self.patch_size)
        img_patches = self.preprocess(img_data, self.patch_size, (patch_x, patch_y), flip_op)
        gt_patches = self.preprocess(gt_data, self.patch_size, (patch_x, patch_y), flip_op)
        # preprocess an image patch and append it into tensor
        for _ in range(self.patch_num-1):
            flip_op = np.random.randint(3)
            patch_x = np.random.randint(0, high= w - self.patch_size)
            patch_y = np.random.randint(0, high= h - self.patch_size)
            temp = self.preprocess(img_data, self.patch_size, (patch_x, patch_y), flip_op)
            img_patches = np.append(img_patches, temp, axis=0)
            temp =  self.preprocess(gt_data, self.patch_size, (patch_x, patch_y), flip_op)
            gt_patches = np.append(gt_patches, temp, axis=0)
        return {'input': torch.from_numpy(img_patches), 'gt': torch.from_numpy(gt_patches)}
    
    def __len__(self):
        return len(self.img_ls)

class ValiSet(Dataset):
    """
    parameter:
        img_ls: validation image list
        img_dir: validation image directory
        gt_dir: ground truth image directory
        patch_size: patch size
    """
    def __init__(self, img_ls, img_dir, gt_dir, patch_size = 256):
        self.img_ls = img_ls
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size

        self.trans = trfs.Compose([
            trfs.Resize(patch_size),
            trfs.CenterCrop(patch_size),
            trfs.ToTensor()
        ])

    def __len__(self):
        return len(self.img_ls)

    def __getitem__(self, index):
        img_name = self.img_ls[index] # image name
        gt_name = utils.get_gt_name(img_name) # ground truth name
        # read data
        img_data = Image.open(os.path.join(self.img_dir, img_name)) # input image data
        gt_data = Image.open(os.path.join(self.gt_dir, gt_name)) # ground truth image data
        # image transform
        img_trans = self.trans(img_data)
        gt_trans = self.trans(gt_data)
        return {'input': img_trans, 'gt': gt_trans}
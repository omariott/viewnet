import os
import time
import csv

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
from skimage import io, transform
import matplotlib.image
import matplotlib.pyplot as plt


def bbcrop(image):
    segmask = np.asarray(image)[:,:,3]
    rows = np.any(segmask, axis=1)
    cols = np.any(segmask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    height, width = bottom-top, right-left
    hcenter = (top + bottom) // 2
    wcenter = (left + right) // 2
    maxdim = max(height, width)
    topcorner = hcenter - maxdim // 2
    leftcorner = wcenter - maxdim // 2
    cropped = TF.crop(image, topcorner, leftcorner, maxdim, maxdim)
    return cropped
    
    


class ShapeNet_paired_dataset(Dataset):
    """ShapeNet dataset."""
    """
        planes : 02691156, start 21feae1212b07575f23c3116d040903f
        bus start ee8d5ded331cfb76c0100ca2e1097f90
        cars : 02958343, start e619129ae2e1d09884ccdf529614144
        chairs : 03001627, start 6af8d7bfa508b8d23759750e8db40476
        
    """

    def __init__(self, split, category, src_dir, bg, bg_path,
                transform=torchvision.transforms.ToTensor(),):

        category_dict = {'aeroplane': '02691156',
                         'bicycle': '02834778',
                         'boat': '04530566',
                         'bottle': '02876657',
                         'bus': '02924116',
                         'car': '02958343',
                         'chair': '03001627',
                         'motorbike': '03790512',
                         'sofa': '04256520',
                         'diningtable': '04379243',
                         'train': '04468005',
                         'tvmonitor': '03211117',
                         }
        cad_id = category_dict[category]

        splitpath = os.path.join(src_dir, 'all.csv')
        with open(splitpath, mode ='r')as splitfile:
            splits = list(csv.reader(splitfile))
        splits = np.array(splits[1:])
        if split in ['train','val','test']:
            samples = [sample[-2] for sample in splits if sample[1] == cad_id and sample[-1]==split]
        elif split=='full':
            samples = [sample[-2] for sample in splits if sample[1] == cad_id]
        else:
            print("Wrong split specified.")
            print("Split should be train, val, test or full but got {} instead.".format(split))
            exit()
        

        self.src_dir = src_dir
        self.transform = transform
        self.pose_augment = self.bg_trans = torchvision.transforms.Compose([
            bbcrop,
#            torchvision.transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=.5),
            torchvision.transforms.RandomResizedCrop(128, scale=(.8, 1.2), ratio=(1., 1.), interpolation=2),
            torchvision.transforms.RandomRotation(3,),
        ])
        self.objects = [os.path.join(src_dir, cad_id, obj_id) for obj_id in samples]
        self.views_per_model = 10
        self.images = self.objects*self.views_per_model
        self.images.sort()
        self.bg = bg
        self.bg_path = bg_path
        self.len = len(self.images)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.bg:
            np.random.seed()
            bg_dir = self.bg_path
            with open(bg_dir+'/ClassName.txt') as f:
                class_names = f.read().splitlines()
            class_id = np.random.choice(class_names)
            bg_list = os.listdir(bg_dir+class_id)
            bg_id = np.random.choice(bg_list)
            bg_name = os.path.join(bg_dir+class_id, bg_id)
            bg = Image.open(bg_name).convert('RGB')
            bg = self.transform(bg)

            aux_bg_id = np.random.choice(bg_list)
            aux_bg_name = os.path.join(bg_dir+class_id, aux_bg_id)
            aux_bg = Image.open(aux_bg_name).convert('RGB')
            aux_bg = self.transform(aux_bg)



        view_id = idx%self.views_per_model
        obj_id = idx//self.views_per_model

        angle = np.loadtxt(os.path.join(self.objects[obj_id],
                                         "view.txt"))[view_id]
        img_name = os.path.join(self.objects[obj_id],
                                "render_"+str(view_id)+".png")
        pose_im = Image.open(img_name).convert('RGBA')
        if self.transform:
            tar_im = pose_im
            tar_im = self.transform(tar_im)[:3]
            pose_im = self.pose_augment(pose_im)
            pose_im = self.transform(pose_im)
            seg_mask = pose_im[3].float()
            if self.bg:
                pose_im = pose_im[:3]*seg_mask + bg*(1-seg_mask)
            else:
                pose_im = pose_im[:3]

        buddy_id = np.random.randint(self.views_per_model)
        while buddy_id==view_id:
            buddy_id = np.random.randint(self.views_per_model)
        img_name = os.path.join(self.objects[obj_id],
                                "render_"+str(buddy_id)+".png")
        cont_im = Image.open(img_name).convert('RGBA')
        if self.transform:
            cont_im = self.transform(cont_im)
            seg_mask = cont_im[3].float()
            if self.bg:
                cont_im = cont_im[:3]*seg_mask + aux_bg*(1-seg_mask)
            else:
                cont_im = cont_im[:3]
        
        cont_angle = np.loadtxt(os.path.join(self.objects[obj_id],
                                         "view.txt"))[buddy_id]
        azimuth = angle[0]
        elev = angle[1]
        cont_azimuth = cont_angle[0]
        cont_elev = cont_angle[1]

        return pose_im, cont_im, tar_im, azimuth, elev



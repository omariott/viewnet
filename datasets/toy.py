import os
import time
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import numpy as np
from skimage import io, transform
import matplotlib.image
import matplotlib.pyplot as plt


class ico_dataset(Dataset):
    """icosahedron dataset."""

    def __init__(self, set, src_dir="/ssddata/octave/Toy/ico", ids=None, anchored_rate=1, transform=torchvision.transforms.ToTensor(), extended=True):
        """

        """

        if set=='train':
            start = 0
            end = 0.7
        elif set=='val':
            start = 0.7
            end = 0.8
        elif set=='test':
            start = 0.8
            end = 1
        elif set=='full':
            start = 0
            end = 1
        else:
            print("Wrong set specified.")
            print("Set should be train, val, test or full but got {} instead.".format(set))
            exit()

        self.src_dir = src_dir
        if extended:
            self.src_dir = self.src_dir+str(2)
            self.elev_resolution = 50
            self.azi_resolution = 100
        else:
            self.elev_resolution = 18
            self.azi_resolution = 72

        self.len = self.elev_resolution * self.azi_resolution
        if ids is None:
            self.ids = np.arange(self.len)
            np.random.shuffle(self.ids)
        else:
            self.ids = ids
        start_ind = int(len(self.ids)*start)
        end_ind = int(len(self.ids)*end)
        self.images = self.ids[start_ind:end_ind]
        self.len = len(self.images)

        self.anchored = np.arange(self.len)
        np.random.shuffle(self.anchored)
        self.anchored = self.anchored[:int(anchored_rate*self.len)]
        self.transform = transform



    def __getids__(self):
        return self.ids

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        image_id = self.images[idx]
        img_name = os.path.join(self.src_dir,
                                str(image_id).zfill(4)+".png")
        image = Image.open(img_name).convert('RGB')
        elevation = (image_id//self.azi_resolution)*180.0/self.elev_resolution
        azimuth = (image_id%self.azi_resolution)*360.0/self.azi_resolution
        anchor = int(idx in self.anchored)
        if self.transform:
            image = self.transform(image)

        return image, azimuth, elevation


class ico_paired_dataset(Dataset):
    """ico dataset."""


    """icosahedron dataset."""

    def __init__(self, src_dir="/ssddata/octave/Toy/ico", ids=None, transform=torchvision.transforms.ToTensor(), set='full', extended=True):
        """

        """
        if set=='train':
            start = 0
            end = 0.07
        elif set=='val':
            start = 0.7
            end = 0.8
        elif set=='test':
            start = 0.8
            end = 1
        elif set=='full':
            start = 0
            end = 1
        else:
            print("Wrong set specified.")
            print("Set should be train, val or test but got {} instead.".format(set))
            exit()

        self.src_dir = src_dir
        if extended:
            self.src_dir = self.src_dir+str(2)
            self.elev_resolution = 50
            self.azi_resolution = 100
        else:
            self.elev_resolution = 18
            self.azi_resolution = 72

        self.len = self.elev_resolution * self.azi_resolution
        if ids==None:
            self.ids = np.arange(self.len)
            np.random.shuffle(self.ids)
        else:
            self.ids = ids
        start_ind = int(len(self.ids)*start)
        end_ind = int(len(self.ids)*end)
        self.images = self.ids[start_ind:end_ind]

        self.indexes = np.arange(self.len)
        np.random.shuffle(self.indexes)
        self.transform = transform

    def reshuffle(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_id = self.images[idx]
        img_name = os.path.join(self.src_dir,
                                str(image_id).zfill(4)+".png")
        image_pose = io.imread(img_name)
        rand_ind = np.int64(np.random.randint(self.len))
        img_name = os.path.join(self.src_dir,
                                str(rand_ind).zfill(4)+".png")
        image_style = io.imread(img_name)
        elevation = (image_id//self.azi_resolution)*180.0/self.elev_resolution - 90
        azimuth = (image_id%self.azi_resolution)*360.0/self.azi_resolution - 180
        c_elevation = (rand_ind//self.azi_resolution)*180.0/self.elev_resolution - 90
        c_azimuth = (rand_ind%self.azi_resolution)*360.0/self.azi_resolution - 180
        if self.transform:
            image_pose = self.transform(image_pose)
            image_style = self.transform(image_style)

        return image_pose, image_style, azimuth, elevation, c_azimuth, c_elevation

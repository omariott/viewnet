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
import scipy.io


class Pascal3D_dataset(Dataset):
    """Pascal 3D+ dataset."""

    def __init__(self, category, subset, src_dir, ids=None, transform=torchvision.transforms.ToTensor(), split=None):

        assert subset in ['imagenet', 'pascal']
        assert split in ['train', 'val']

        classname = category+"_"+subset
        self.subset = subset
        self.cat = category
        self.extension = '.PNG' if self.subset=='pascal' else '.JPEG'
        self.extension_len = 4 if self.subset=='pascal' else 5
        self.src_dir = os.path.join(src_dir, "Images", classname)
        self.lab_dir = os.path.join(src_dir, "Annotations", classname)
        self.mask_dir = os.path.join(src_dir, "Masks", classname)

        self.transform = transform
        self.ids = os.listdir(self.src_dir)
        ids = []
        for im_id in self.ids:
            lab_name = os.path.join(self.lab_dir, im_id[:-self.extension_len]+".mat")
            labels = scipy.io.loadmat(lab_name)
            ids.append(im_id)
        self.ids = ids

        if split=='val':
            set_file = os.path.join(src_dir, "Image_sets", classname + "_val.txt")
            with open(set_file, 'r') as val_ids_file:
                val_ids = val_ids_file.read().splitlines() 
            self.ids = [im_id  + self.extension for im_id in val_ids]
        if split=='train':
            set_file = os.path.join(src_dir, "Image_sets", classname + "_train.txt")
            with open(set_file, 'r') as val_ids_file:
                val_ids = val_ids_file.read().splitlines() 
            self.ids = [im_id  + self.extension for im_id in val_ids]


        self.images = self.ids
        self.len = len(self.images)



    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        lab_name = os.path.join(self.lab_dir,
                            self.images[idx][:-self.extension_len]+".mat")
        labels = scipy.io.loadmat(lab_name)#, simplify_cells=True)
        i = 0
        while labels['record'][0][0]['objects'][0][i]['class'].item() != self.cat:
            i += 1
        dat = labels['record'][0][0]['objects'][0][i]
        bbox = dat['bbox'][0]
        azimuth = dat['viewpoint'][0][0]['azimuth'].item()
        elevation = dat['viewpoint'][0][0]['elevation'].item()

        if azimuth > 180:
            azimuth = azimuth - 360

        img_name = os.path.join(self.src_dir,
                                self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        
        image = image.crop(bbox)
        max_dim = max(image.width, image.height)
        
        canvas = Image.new('RGB', (max_dim, max_dim))
        origin = ((max_dim-image.width)//2, (max_dim-image.height)//2)
        canvas.paste(image, origin)
        image = canvas
        if self.transform:
            image = self.transform(image)
        
        return image, image.flip(-1), image, azimuth, elevation


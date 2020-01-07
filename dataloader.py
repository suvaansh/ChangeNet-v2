import torch 
import torch.nn as nn
from torch.nn import functional as F
import csv
import os
from PIL import Image
import numpy as np


class VL_CMU_CD(torch.utils.data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        """
            Args:
                root (string): root directory for dataset
                set (string): training or test set
                transform (torchvision.transforms): transforms to be applied to dataset
                target_transform (torchvision.transforms): transforms to be applied to labels
        """

        self.root = root
        self.path_images = root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        path_csv = os.path.join(self.root)
        file_csv = os.path.join('../../', set + '.csv')

        print(file_csv)

        self.images = self.read_object_labels_csv(file_csv)

    def __getitem__(self, index):

        first, second, label, mask = self.images[index]
#         print(first, second, label)
        
        first_path = os.path.join(self.path_images, 'left', first)
        second_path = os.path.join(self.path_images, 'right', second)
        label_path = os.path.join(self.path_images, 'GT_MULTICLASS', label)
        mask_path = os.path.join(self.path_images, 'mask', mask)

        # Loading image and labels(segmentation masks)
        
        img1 = np.array(Image.open(first_path).resize((256,192), Image.ANTIALIAS))
        img1 = img1/255.0 # Normalise image

        img2 = np.array(Image.open(second_path).resize((256,192), Image.ANTIALIAS))
        img2 = img2/255.0

        target = np.load(label_path)
        

        mask_img = np.array(Image.open(mask_path).resize((256,192), Image.ANTIALIAS))
        mask_img = mask_img/255.0

        mask_img = np.expand_dims((mask_img>0.2).astype(np.float32), axis=2)

        # Applying tranforms to images
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Applying transforms to labels
        if self.target_transform is not None:
            target = self.target_transform(target)
            mask_img = self.target_transform(mask_img)

        
        return (img1, first_path), (img2, second_path), (target, label_path), (mask_img, mask_path)

    def read_object_labels_csv(self, file, header=True):
        """
            Reads data csv file and create returns list of datapoints
            
            Args:
                file (string): CSV file name
                header (bool): read first row in csv as datapoint or not

            Returns:
                images (list): A list of tuples containing input-images & label file-names 
        """
        images = []
        num_categories = 0
        print('[dataset] read', file)
        with open(file, 'r') as f:
            reader = csv.reader(f)
            rownum = 0
            for row in reader:
                if header and rownum == 0:
                    header = row
                else:
                    if num_categories == 0:
                        num_categories = len(row) - 1
                    first = row[0] + '.jpeg'
                    second = row[0] +'.jpeg'
                    label = row[0] + '.npy'
                    mask = row[0] + '.png'
                    item = (first, second, label, mask)
                    images.append(item)
                rownum += 1
        return images
    
    def __len__(self):
        """Returns total number of examples in dataset"""
        return len(self.images)

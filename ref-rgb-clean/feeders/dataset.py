import os

import numpy as np
# import cupy as cp
import cv2
import pandas as pd
#
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
#
# from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from random import choice
import random
# import utils
# import transform
# import PIL
import os
import numpy as np
import cv2
import pandas as pd
from scipy.special import factorial
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter

from feeders import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines



def get_data(data_dir, file_path):
    filename_list = get_list_from_filenames(file_path)
    all_pose = []
    all_img = []
    for i in range(len(filename_list)):
        file_path = data_dir + filename_list[i]
        data = np.load(file_path, allow_pickle=True)
        pose = np.array(data['pose'])
        img = np.array(data['image'])

    
        all_pose.append(pose)
        all_img.append(img)


    poses = np.concatenate(all_pose)
    imgs = np.concatenate(all_img)
    return imgs, poses


def euler_to_mat(yaw, pitch, roll):
    yaw = torch.deg2rad(yaw)
    pitch = torch.deg2rad(pitch)
    roll = torch.deg2rad(roll)
    
    R = torch.cat((
            torch.cos(yaw)*torch.cos(roll),
            -torch.cos(yaw)*torch.sin(roll),
            torch.sin(yaw),
            torch.cos(pitch)*torch.sin(roll) + torch.cos(roll)*torch.sin(pitch)*torch.sin(yaw),
            torch.cos(pitch)*torch.cos(roll) - torch.sin(pitch)*torch.sin(yaw)*torch.sin(roll),
            -torch.cos(yaw)*torch.sin(pitch),
            torch.sin(pitch)*torch.sin(roll)-torch.cos(pitch)*torch.cos(roll)*torch.sin(yaw),
            torch.cos(roll)*torch.sin(pitch) + torch.cos(pitch)*torch.sin(yaw)*torch.sin(roll),
            torch.cos(pitch)*torch.cos(yaw)
    ), dim=0)      
    return R

def embed_mat(yaw, pitch, roll):
    yaw = torch.deg2rad(yaw)
    pitch = torch.deg2rad(pitch)
    roll = torch.deg2rad(roll)

    def poly_expand(x,n=16,kind='sin'):
        i = torch.arange(n)
        p = x**i
        
        if kind == 'sin':
            f = i % 2
            c = (-(i%4)+2)/factorial(i)
        elif kind =='cos':
            f = ~i % 2
            c = (-(i%4)+1)/factorial(i)
        return (f*p*c).float()
    sin_yaw = torch.Tensor(poly_expand(yaw)).unsqueeze(1)
    cos_yaw = torch.Tensor(poly_expand(yaw, kind='cos')).unsqueeze(1)
    sin_pitch = torch.Tensor(poly_expand(pitch)).unsqueeze(1)
    cos_pitch = torch.Tensor(poly_expand(pitch,kind='cos')).unsqueeze(1)
    sin_roll = torch.Tensor(poly_expand(roll)).unsqueeze(1)
    cos_roll = torch.Tensor(poly_expand(roll,kind='cos')).unsqueeze(1)
    R_embed = torch.cat((
            cos_yaw*cos_roll,
            -cos_yaw*sin_roll,
            sin_yaw,
            cos_pitch*sin_roll + cos_roll*sin_pitch*sin_yaw,
            cos_pitch*cos_roll - sin_pitch*sin_yaw*sin_roll,
            -cos_yaw*sin_pitch,
            sin_pitch*sin_roll-cos_pitch*cos_roll*sin_yaw,
            cos_roll*sin_pitch + cos_pitch*sin_yaw*sin_roll,
            cos_pitch*cos_yaw
    ), dim=1).flatten(0)     
    return R_embed

def gaussian_sample_indices(yaw, ref_poses, k=10, bin_size=20, sigma=30):
    """
    Gaussian-based sampling of reference indices based on yaw angle proximity.
    """
    num_bins = int(360 // bin_size)
    bin_centers = np.linspace(-180 + bin_size/2, 180 - bin_size/2, num_bins)

    weights = np.exp(-((bin_centers - yaw) ** 2) / (2 * sigma ** 2))
    weights = weights / np.sum(weights)

    indices_per_bin = (weights * k).astype(int)
    while np.sum(indices_per_bin) < k:
        indices_per_bin[np.argmax(weights)] += 1

    sampled_indices = []
    ref_yaws = np.array([pose[0] for pose in ref_poses])

    for bin_idx, count in enumerate(indices_per_bin):
        if count > 0:
            center = bin_centers[bin_idx]
            in_bin = np.where(
                (ref_yaws >= center - bin_size/2) & 
                (ref_yaws <  center + bin_size/2)
            )[0]
            if len(in_bin) > 0:
                chosen = random.choices(in_bin.tolist(), k=count)
                sampled_indices.extend(chosen)

    if len(sampled_indices) < k:
        extra = random.choices(range(len(ref_poses)), k=k-len(sampled_indices))
        sampled_indices.extend(extra)

    return sampled_indices

class Pose_300W_LP(Dataset):
  
    def __init__(self, data_dir, filename_path, ref_dir, ref_filename_path, transform=None):
        self.data_dir = data_dir
        self.ref_dir = ref_dir
        
        self.transform =  transforms.Compose([transforms.ToTensor(),transforms.RandomResizedCrop(size=(64,64),scale=(0.8,1.2)), transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2),hue=(-0.2,0.2),saturation=(0.8,1.2)), transforms.Normalize(0,1)])
        self.filename_path = filename_path
        self.ref_filename_path = ref_filename_path
        if data_dir == ref_dir:
            self.imgs, self.poses = get_data(self.data_dir, self.filename_path)
            self.refs, self.ref_poses = self.imgs, self.poses
        else:
            self.imgs, self.poses = get_data(self.data_dir, self.filename_path)
            self.refs, self.ref_poses = get_data(self.ref_dir, self.ref_filename_path)
        self.length = len(self.poses)
        self.ref_length = len(self.ref_poses)
        self.cut_out = Cutout()
    
    def __getitem__(self, index):
        # imgs = self.landmark[index]
        pose = self.poses[index]
        img = self.transform(self.imgs[index])
        img = self.cut_out(img)
        k = 10
        if self.data_dir == self.ref_dir:
            yaw = pose[0]  # current sample's yaw
            ref_idx = gaussian_sample_indices(yaw, self.ref_poses, k=k, bin_size=20, sigma=30)
        else:
            yaw = pose[0]  # still use current yaw to sample references
            ref_idx = gaussian_sample_indices(yaw, self.ref_poses, k=k, bin_size=20, sigma=30)

        ref = []
        for i in range(k):
            r = self.transform(self.refs[ref_idx[i]])
            r = self.cut_out(r)
            ref.append(r)
        ref_pose = [self.ref_poses[ref_idx[i]] for i in range(k)]
        
        
        yaw = pose[0] 
        pitch = pose[1]
        roll = pose[2] 
        yaw = -torch.Tensor([yaw])
        pitch = torch.Tensor([pitch])
        roll = torch.Tensor([roll])
        
        ref_yaw = [ref_pose[i][0]  for i in range(k)]
        ref_pitch = [ref_pose[i][1]  for i in range(k)]
        ref_roll = [ref_pose[i][2]  for i in range(k)]
        ref_yaw = [-torch.Tensor([ref_yaw[i]]) for i in range(k)]
        ref_pitch = [torch.Tensor([ref_pitch[i]]) for i in range(k)]
        ref_roll = [torch.Tensor([ref_roll[i]]) for i in range(k)]
        
        matrix_label = euler_to_mat(yaw,pitch,roll)
        matrix_embed_label = embed_mat(yaw,pitch,roll)

        ref_matrix_label = [euler_to_mat(ref_yaw[i],ref_pitch[i],ref_roll[i]) for i in range(k)]
        ref_matrix_embed_label = [embed_mat(ref_yaw[i],ref_pitch[i],ref_roll[i]) for i in range(k)]
      
        euler_label = torch.FloatTensor([yaw, pitch, roll])
        matrix_label = torch.FloatTensor(matrix_label)

        ref_euler_label = [torch.FloatTensor([ref_yaw[i], ref_pitch[i], ref_roll[i]]) for i in range(k)]
        ref_matrix_label = [torch.FloatTensor(ref_matrix_label[i]) for i in range(k)]
        return img, ref, matrix_label, euler_label, ref_matrix_label, ref_euler_label, index

    def __len__(self):
        # 122,450
        return self.length


class AFLW2000(Dataset):
   
    def __init__(self, data_dir, filename_path, ref_dir, ref_filename_path, transform=None):
        self.data_dir = data_dir
        self.ref_dir = ref_dir
        self.transform =  transforms.Compose([transforms.ToTensor(),transforms.Normalize(0,1)])
        self.filename_path = filename_path
        self.ref_filename_path = ref_filename_path
        # print(data_dir, ref_dir)
        if data_dir == ref_dir:
            self.imgs, self.poses = get_data(self.data_dir, self.filename_path)
            self.refs, self.ref_poses = self.imgs, self.poses
        else:
            self.imgs, self.poses = get_data(self.data_dir, self.filename_path)
            self.refs, self.ref_poses = get_data(self.ref_dir, self.ref_filename_path)
        self.length = len(self.poses)
        self.ref_length = len(self.ref_poses)

        
    def __getitem__(self, index):
        # imgs = self.landmark[index]
        pose = self.poses[index]
        img = self.transform(self.imgs[index])
        k = 10
        if self.data_dir == self.ref_dir:
            yaw = pose[0]  # current sample's yaw
            ref_idx = gaussian_sample_indices(yaw, self.ref_poses, k=k, bin_size=20, sigma=30)
        else:
            yaw = pose[0]  # still use current yaw to sample references
            ref_idx = gaussian_sample_indices(yaw, self.ref_poses, k=k, bin_size=20, sigma=30)

        ref = [self.transform(self.refs[ref_idx[i]]) for i in range(k)]
        ref_pose = [self.ref_poses[ref_idx[i]] for i in range(k)]
        
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]
        yaw = -torch.Tensor([yaw])
        pitch = torch.Tensor([pitch])
        roll = torch.Tensor([roll])
        
        ref_yaw = [ref_pose[i][0] for i in range(k)]
        ref_pitch = [ref_pose[i][1] for i in range(k)]
        ref_roll = [ref_pose[i][2] for i in range(k)]
        ref_yaw = [-torch.Tensor([ref_yaw[i]]) for i in range(k)]
        ref_pitch = [torch.Tensor([ref_pitch[i]]) for i in range(k)]
        ref_roll = [torch.Tensor([ref_roll[i]]) for i in range(k)]
        
        matrix_label = euler_to_mat(yaw,pitch,roll)
        matrix_embed_label = embed_mat(yaw,pitch,roll)

        ref_matrix_label = [euler_to_mat(ref_yaw[i],ref_pitch[i],ref_roll[i]) for i in range(k)]
        ref_matrix_embed_label = [embed_mat(ref_yaw[i],ref_pitch[i],ref_roll[i]) for i in range(k)]
      
        euler_label = torch.FloatTensor([yaw, pitch, roll])
        matrix_label = torch.FloatTensor(matrix_label)

        ref_euler_label = [torch.FloatTensor([ref_yaw[i], ref_pitch[i], ref_roll[i]]) for i in range(k)]
        ref_matrix_label = [torch.FloatTensor(ref_matrix_label[i]) for i in range(k)]

        return img, ref, matrix_label, euler_label, ref_matrix_label, ref_euler_label, index

    def __len__(self):
        # 122,450
        return self.length


class BIWI(Dataset):
    
    def __init__(self, data_dir, filename_path, ref_dir, ref_filename_path, transform=None):
        self.data_dir = data_dir
        self.ref_dir = ref_dir
        # print(data_dir, ref_dir)
        self.transform =  transforms.Compose([transforms.ToTensor(),transforms.Normalize(0,1)])
        self.filename_path = filename_path
        self.ref_filename_path = ref_filename_path

        if data_dir == ref_dir:
            self.imgs, self.poses = get_data(self.data_dir, self.filename_path)
            self.refs, self.ref_poses = self.imgs, self.poses
        else:
            self.imgs, self.poses = get_data(self.data_dir, self.filename_path)
            self.refs, self.ref_poses = get_data(self.ref_dir, self.ref_filename_path)
        self.length = len(self.poses)
        self.ref_length = len(self.ref_poses)

    def __getitem__(self, index):
        # imgs = self.landmark[index]
        pose = self.poses[index]
        img = self.transform(self.imgs[index])
        k = 10
        if self.data_dir == self.ref_dir:
            yaw = pose[0]  # current sample's yaw
            ref_idx = gaussian_sample_indices(yaw, self.ref_poses, k=k, bin_size=20, sigma=30)
        else:
            yaw = pose[0]  # still use current yaw to sample references
            ref_idx = gaussian_sample_indices(yaw, self.ref_poses, k=k, bin_size=20, sigma=30)
        ref = [self.transform(self.refs[ref_idx[i]]) for i in range(k)]
        ref_pose = [self.ref_poses[ref_idx[i]] for i in range(k)]
        
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]
        yaw = -torch.Tensor([yaw])
        pitch = torch.Tensor([pitch])
        roll = torch.Tensor([roll])
        
        ref_yaw = [ref_pose[i][0] for i in range(k)]
        ref_pitch = [ref_pose[i][1] for i in range(k)]
        ref_roll = [ref_pose[i][2] for i in range(k)]
        ref_yaw = [-torch.Tensor([ref_yaw[i]]) for i in range(k)]
        ref_pitch = [torch.Tensor([ref_pitch[i]]) for i in range(k)]
        ref_roll = [torch.Tensor([ref_roll[i]]) for i in range(k)]
        
        matrix_label = euler_to_mat(yaw,pitch,roll)
        matrix_embed_label = embed_mat(yaw,pitch,roll)

        ref_matrix_label = [euler_to_mat(ref_yaw[i],ref_pitch[i],ref_roll[i]) for i in range(k)]
        ref_matrix_embed_label = [embed_mat(ref_yaw[i],ref_pitch[i],ref_roll[i]) for i in range(k)]
      
        euler_label = torch.FloatTensor([yaw, pitch, roll])
        matrix_label = torch.FloatTensor(matrix_label)

        ref_euler_label = [torch.FloatTensor([ref_yaw[i], ref_pitch[i], ref_roll[i]]) for i in range(k)]
        ref_matrix_label = [torch.FloatTensor(ref_matrix_label[i]) for i in range(k)]

        return img, ref, matrix_label, euler_label, ref_matrix_label, ref_euler_label, index

    def __len__(self):
        # 122,450
        return self.length


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=3, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        self.n_holes = np.random.choice([0,1], p=(0.6,0.4))
        self.length = random.randint(16,24)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
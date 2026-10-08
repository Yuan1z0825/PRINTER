import os.path
import random

import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class aligned3dataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')
        self.dir_B_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.has_mask_A = os.path.exists(self.dir_A_mask)
        self.has_mask_B = os.path.exists(self.dir_B_mask)
        self.A_paths = self._get_paired_paths(self.dir_A, self.dir_A_mask) if self.has_mask_A else sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = self._get_paired_paths(self.dir_B, self.dir_B_mask) if self.has_mask_B else sorted(make_dataset(self.dir_B, opt.max_dataset_size))


        train_percentage = opt.train_percentage
        assert train_percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], \
            "train_percentage must be one of [10, 20, 30, 40, 50, 60, 70, 80, 90]"
        num_samples = int(len(self.A_paths) * train_percentage / 100)
        self.A_paths = self.A_paths[:num_samples]
        self.B_paths = self.B_paths[:num_samples]

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def _get_paired_paths(self, image_dir, mask_dir):
        all_img_paths = sorted(make_dataset(image_dir, self.opt.max_dataset_size))
        paired_paths = [p for p in all_img_paths if os.path.exists(os.path.join(mask_dir, os.path.basename(p)))]
        return paired_paths

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # H&E
        B_path = self.B_paths[index % self.B_size]  # IHC
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = transform(A_img)
        B = transform(B_img)
        data = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        if self.has_mask_A:
            A_mask_path = os.path.join(self.dir_A_mask, os.path.basename(A_path))
            A_mask_img = Image.open(A_mask_path).convert('L').resize((self.opt.load_size, self.opt.load_size))
            A_mask_np = np.array(A_mask_img) > 127
            A_mask_tensor = torch.from_numpy(A_mask_np.astype(np.float32)).unsqueeze(0)
            data['A_mask'] = A_mask_tensor

        if self.has_mask_B:
            B_mask_path = os.path.join(self.dir_B_mask, os.path.basename(B_path))
            B_mask_img = Image.open(B_mask_path).convert('L').resize((self.opt.load_size, self.opt.load_size))
            B_mask_np = np.array(B_mask_img) > 127
            B_mask_tensor = torch.from_numpy(B_mask_np.astype(np.float32)).unsqueeze(0)
            data['B_mask'] = B_mask_tensor
        return data

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
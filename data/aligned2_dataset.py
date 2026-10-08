import os.path
import random
import torchvision.transforms as transforms
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class aligned2dataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
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

    def __getitem__(self, index):
        if self.opt.phase == 'test':
            A_path = self.A_paths[index % self.A_size]  # H&E
            B_path = self.B_paths[index % self.B_size]  # IHC
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            A = transform(A_img)
            B = transform(B_img)


            return {'A': A, 'B': B,
                    'A_paths': A_path, 'B_paths': B_path}
        else:
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

            return {'A': A, 'B': B,
                    'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
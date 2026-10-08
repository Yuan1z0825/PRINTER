import os
import re
import random
import numpy as np
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import util.util as util


def classify_center(filename):
    """根据文件名判断所属中心并返回数字标签（若无法判断则抛出 ValueError）"""
    name = filename.lower()
    if "patch" in name:
        return 0   # 301
    elif "of" in name:
        return 1   # AKMP
    else:
        match = re.search(r"(\d+)", name)
        if match:
            num = match.group(1)
            if len(num) == 6:
                return 2   # KZ
            elif len(num) == 4 and "patch" not in name:
                return 3   # youyi
            elif len(num) == 3:
                return 4   # HZ
    # 无法识别时抛出错误（便于立即发现数据问题）
    raise ValueError(f"Cannot classify center from filename: '{filename}'")


class Unaligned2Dataset(BaseDataset):
    """
    支持同时加载图像与mask的非配对图像数据集（如CycleGAN等）

    目录结构要求：
        - dataroot/trainA/, trainB/：图像
        - dataroot/trainA_mask/, trainB_mask/：与图像同名的mask（可选）
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.current_epoch = 0

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')
        self.dir_B_mask = os.path.join(opt.dataroot, opt.phase + 'B_mask')

        if opt.phase == "test" and not os.path.exists(self.dir_A) and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
            self.dir_A_mask = os.path.join(opt.dataroot, "valA_mask")
            self.dir_B_mask = os.path.join(opt.dataroot, "valB_mask")

        self.has_mask_A = os.path.exists(self.dir_A_mask)
        self.has_mask_B = os.path.exists(self.dir_B_mask)

        # 筛选出图像和mask均存在的配对样本
        self.A_paths = self._get_paired_paths(self.dir_A, self.dir_A_mask) if self.has_mask_A else sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = self._get_paired_paths(self.dir_B, self.dir_B_mask) if self.has_mask_B else sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def _get_paired_paths(self, image_dir, mask_dir):
        all_img_paths = sorted(make_dataset(image_dir, self.opt.max_dataset_size))
        paired_paths = [p for p in all_img_paths if os.path.exists(os.path.join(mask_dir, os.path.basename(p)))]
        return paired_paths

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_B = index % self.B_size if self.opt.serial_batches else random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB').resize((self.opt.load_size, self.opt.load_size))
        B_img = Image.open(B_path).convert('RGB').resize((self.opt.load_size, self.opt.load_size))

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        A = transform(A_img)
        B = transform(B_img)

        # === 新增部分：中心标识（如果无法识别会抛出 ValueError） ===
        # A_center = classify_center(os.path.basename(A_path))
        # B_center = classify_center(os.path.basename(B_path))
        A_center = 0
        B_center = 0

        data = {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path,
            'A_center': A_center,
            'B_center': B_center
        }

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
        return self.A_size

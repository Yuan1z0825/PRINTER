import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

def is_image_file(filename):
    """Check if a file is an image file based on its extension."""
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def extract_modal_part(path):
    """Extract the relevant part of the filename (X_Y) ignoring modality name."""
    # Assuming the format is 'modality_X_Y.ext' (e.g., 'modality_1_100.png')
    patient_id = path.split('/')[-2]
    coord_x = path.split('/')[-1].split('_')[0]
    coord_y = path.split('/')[-1].split('_')[1]
    return '_'.join([patient_id, coord_x, coord_y])

class AlignedDataset(BaseDataset):
    """
    This dataset class loads aligned datasets where each image in domain A has a corresponding image in domain B.

    It assumes that the filenames in 'trainA' and 'trainB' correspond to each other.
    For example, 'trainA/img001.png' and 'trainB/img001.png' are paired.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # path to domain A images
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # path to domain B images

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
                and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        # Ensure the files in both domains are sorted and aligned
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        # Extract the relevant part (X_Y) from the filenames in both directories
        A_relevant_parts = {extract_modal_part(path) for path in self.A_paths}
        B_relevant_parts = {extract_modal_part(path) for path in self.B_paths}

        # Find the common X_Y parts between domain A and domain B
        common_relevant_parts = A_relevant_parts.intersection(B_relevant_parts)

        # Find the common filenames between domain A and domain B
        # Now filter the paths to only include the common filenames
        self.A_paths = sorted([path for path in self.A_paths if extract_modal_part(path) in common_relevant_parts])
        self.B_paths = sorted([path for path in self.B_paths if extract_modal_part(path) in common_relevant_parts])

        # self.A_paths = self.A_paths[:10]
        # self.B_paths = self.B_paths[:10]

        # Check if the number of images in both domains is the same

        self.A_size = len(self.A_paths)  # size of dataset A
        self.B_size = len(self.B_paths)  # size of dataset B

        assert self.A_size == self.B_size, f"Dataset sizes do not match. A: {self.A_size}, B: {self.B_size}"

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary containing A, B, A_paths, and B_paths.
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within range
        B_path = self.B_paths[index % self.B_size]  # make sure index is within range

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        Since A and B are aligned, the length is the same for both datasets.
        """
        return self.A_size  # or self.B_size, they should be equal

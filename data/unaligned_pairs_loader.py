from torch.utils.data import Dataset
import random
import h5py
import time
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from typing import Tuple


class UnalignedPairsDataset(Dataset):
    def __init__(
            self,
            image_paths: list,
            shuffle: bool = True,
            transform=None,
            train: bool = False):
        if shuffle:
            random.shuffle(image_paths)

        self.nSamples = len(image_paths)
        self.lines = image_paths
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index) -> dict:
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = self.load_data(img_path)

        target_transformed = transforms.ToTensor()(target)
        if self.transform is not None:
            img_transformed = self.transform(img)
        else:
            img_transformed = transforms.ToTensor()(img)

        # Return the image in domain A and the density map as the equivalent domain B image...
        return {
            'A': img_transformed.unsqueeze(0),
            'B': target_transformed.unsqueeze(0),
            'A_paths': img_path,
            'B_paths': img_path
        }

    @staticmethod
    def load_gt_path(gt_path: str) -> h5py.File:
        r"""Loads an h5py file into memory. In case of an error in loading, execution will pause for 30 seconds before
        trying again. After 8 retries, the error will be rethrown.
        Args:
            gt_path: str. The path to the data to load.
        Returns
        -------
        The data in the file.
        """
        retry_count = 8
        for i in range(retry_count + 1):
            try:
                gt_file = h5py.File(gt_path)
                return gt_file
            except OSError:
                if i >= retry_count:
                    print('OS Error. Giving Up')
                    raise
                else:
                    print('OS Error. Waiting 30 seconds and trying again...')
                    time.sleep(30)

    def load_data(self, img_path) -> Tuple[Image.Image, np.ndarray]:
        gt_dir = 'csrnet_ground_truth'
        gt_path = img_path.replace('.jpg', '.h5').replace('images', gt_dir)
        img = Image.open(img_path).convert('RGB')
        gt_file = self.load_gt_path(gt_path)
        target = np.asarray(gt_file['density'])

        # CSRNet required a 1/8 size output. We don't...
        # target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

        return img, target

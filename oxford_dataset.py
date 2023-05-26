import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 480

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename.strip('\n') + ".jpg").replace('%', "_")
        mask_path = os.path.join(self.masks_directory, filename.strip('\n') + ".png").replace('%', '_')

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

               # resize images
        image = np.array(Image.fromarray(image).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LINEAR))
        mask = np.array(Image.fromarray(mask).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST))
        trimap = np.array(Image.fromarray(trimap).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST))

        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)
        trimap = np.expand_dims(trimap, 0)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        if self.mode == 'test':
            return open(os.path.join(self.root, 'annotations', 'test.txt'), mode='r').readlines()
        if self.mode == 'train':
            return open(os.path.join(self.root, 'annotations', 'train.txt'), mode='r').readlines()
        if self.mode == 'valid':
            return open(os.path.join(self.root, 'annotations', 'validate.txt'), mode='r').readlines()

    def extract_archive(filepath):
      extract_dir = os.path.dirname(os.path.abspath(filepath))
      dst_dir = os.path.splitext(filepath)[0]
      if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


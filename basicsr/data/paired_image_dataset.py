#basicsr/data/pairedimgdataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, img2tensor, padding, padding_DP

import random
import numpy as np
import torch
import io
import tifffile
from pdb import set_trace as stx


class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration supporting TIFF files with any number of dimensions."""

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load GT and LQ images
        gt_path = self.paths[index]['gt_path']
        img_gt = self.load_tiff_image(gt_path, 'gt')

        lq_path = self.paths[index]['lq_path']
        img_lq = self.load_tiff_image(lq_path, 'lq')

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Padding
            img_gt, img_lq = self.padding(img_gt, img_lq, gt_size)
            # Random crop
            img_gt, img_lq = self.paired_random_crop(img_gt, img_lq, gt_size)
            # Flip and rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = self.random_augmentation(img_gt, img_lq)

        # Convert images to tensors
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def load_tiff_image(self, path, key):
        """Load a TIFF image from the given path."""
        img_bytes = self.file_client.get(path, key)
        try:
            img = tifffile.imread(io.BytesIO(img_bytes))
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            raise Exception(f"{key} path {path} not working: {e}")
        return img

    @staticmethod
    def padding(img_gt, img_lq, gt_size):
        """Pad images to ensure they are at least gt_size in height and width."""
        h, w = img_gt.shape[:2]
        pad_h = max(0, gt_size - h)
        pad_w = max(0, gt_size - w)
        if pad_h > 0 or pad_w > 0:
            padding = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (img_gt.ndim - 2)
            img_gt = np.pad(img_gt, padding, mode='reflect')
            img_lq = np.pad(img_lq, padding, mode='reflect')
        return img_gt, img_lq

    @staticmethod
    def paired_random_crop(img_gt, img_lq, gt_size):
        """Randomly crop images to gt_size."""
        h, w = img_gt.shape[:2]
        top = random.randint(0, h - gt_size)
        left = random.randint(0, w - gt_size)
        slices = [slice(top, top + gt_size), slice(left, left + gt_size)] + [slice(None)] * (img_gt.ndim - 2)
        img_gt = img_gt[tuple(slices)]
        img_lq = img_lq[tuple(slices)]
        return img_gt, img_lq

    @staticmethod
    def random_augmentation(img_gt, img_lq):
        """Apply random augmentations (rotation and flips)."""
        rot_times = random.randint(0, 3)
        v_flip = random.choice([True, False])
        h_flip = random.choice([True, False])

        def augment(img):
            if rot_times:
                img = np.rot90(img, k=rot_times, axes=(0, 1))
            if v_flip:
                img = np.flip(img, axis=0)
            if h_flip:
                img = np.flip(img, axis=1)
            return img

        img_gt = augment(img_gt)
        img_lq = augment(img_lq)
        return img_gt, img_lq


class Dataset_PairedImage_Slide(data.Dataset):
    """Paired image dataset with sliding window support for TIFF images with any number of dimensions."""

    def __init__(self, opt):
        super(Dataset_PairedImage_Slide, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        # Initialize paths
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        # Get image dimensions dynamically
        sample_img = self.load_tiff_image(self.paths[0]['gt_path'], 'gt')
        h, w = sample_img.shape[:2]
        stride = self.opt['stride']
        crop_size = self.opt['gt_size']
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

        print('Patches per line:', self.patch_per_line)
        print('Patches per column:', self.patch_per_colum)
        print('Number of images:', len(self.paths))
        print('Total number of patches:', len(self.paths) * self.patch_per_img)

    def __getitem__(self, index):
        # Determine image and patch indices
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        stride = self.opt['stride']
        crop_size = self.opt['gt_size']
        img_idx = index // self.patch_per_img
        patch_idx = index % self.patch_per_img
        h_idx = patch_idx // self.patch_per_line
        w_idx = patch_idx % self.patch_per_line

        img_idx = img_idx % len(self.paths)

        # Load images
        gt_path = self.paths[img_idx]['gt_path']
        img_gt = self.load_tiff_image(gt_path, 'gt')

        lq_path = self.paths[img_idx]['lq_path']
        img_lq = self.load_tiff_image(lq_path, 'lq')

        # Extract patches
        slices = [slice(h_idx * stride, h_idx * stride + crop_size),
                  slice(w_idx * stride, w_idx * stride + crop_size)] + [slice(None)] * (img_gt.ndim - 2)
        img_gt = img_gt[tuple(slices)]
        img_lq = img_lq[tuple(slices)]

        # Augmentation for training
        if self.opt['phase'] == 'train':
            rot_times = random.randint(0, 3)
            v_flip = random.randint(0, 1)
            h_flip = random.randint(0, 1)
            img_gt = self.augment(img_gt, rot_times, v_flip, h_flip)
            img_lq = self.augment(img_lq, rot_times, v_flip, h_flip)

        # Convert images to tensors
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths) * self.patch_per_img

    def load_tiff_image(self, path, key):
        """Load a TIFF image from the given path."""
        img_bytes = self.file_client.get(path, key)
        try:
            img = tifffile.imread(io.BytesIO(img_bytes))
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            raise Exception(f"{key} path {path} not working: {e}")
        return img

    @staticmethod
    def augment(img, rot_times, v_flip, h_flip):
        """Apply random augmentations (rotation and flips)."""
        if rot_times:
            img = np.rot90(img, k=rot_times, axes=(0, 1))
        if v_flip:
            img = np.flip(img, axis=0)
        if h_flip:
            img = np.flip(img, axis=1)
        return img


class Dataset_PairedImage_Norm(Dataset_PairedImage):
    """Paired image dataset with normalization support for TIFF images."""

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # Additional normalization
        img_lq = data['lq']
        img_lq = (img_lq - img_lq.min()) / (img_lq.max() - img_lq.min())
        data['lq'] = img_lq
        return data


class Dataset_GaussianDenoising(data.Dataset):
    """Dataset for Gaussian denoising supporting TIFF images with any number of dimensions."""

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder, line.strip()) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt.get('geometric_augs', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load GT image
        gt_path = self.paths[index]
        img_gt = self.load_tiff_image(gt_path, 'gt')
        img_lq = img_gt.copy()

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Padding
            img_gt, img_lq = Dataset_PairedImage.padding(img_gt, img_lq, gt_size)
            # Random crop
            img_gt, img_lq = Dataset_PairedImage.paired_random_crop(img_gt, img_lq, gt_size)
            # Flip and rotation
            if self.geometric_augs:
                img_gt, img_lq = Dataset_PairedImage.random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def load_tiff_image(self, path, key):
        """Load a TIFF image from the given path."""
        img_bytes = self.file_client.get(path, key)
        try:
            img = tifffile.imread(io.BytesIO(img_bytes))
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            raise Exception(f"{key} path {path} not working: {e}")
        return img


class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    """Dataset for defocus deblurring with dual-pixel images supporting TIFF files."""

    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lqL_folder = opt['dataroot_lqL']
        self.lqR_folder = opt['dataroot_lqR']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load images
        gt_path = self.paths[index]['gt_path']
        img_gt = self.load_tiff_image(gt_path, 'gt')

        lqL_path = self.paths[index]['lqL_path']
        img_lqL = self.load_tiff_image(lqL_path, 'lqL')

        lqR_path = self.paths[index]['lqR_path']
        img_lqR = self.load_tiff_image(lqR_path, 'lqR')

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Padding
            img_lqL, img_lqR, img_gt = self.padding_dp(img_lqL, img_lqR, img_gt, gt_size)
            # Random crop
            img_lqL, img_lqR, img_gt = self.paired_random_crop_dp(img_lqL, img_lqR, img_gt, gt_size)
            # Flip and rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)

        # Convert images to tensors
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                              bgr2rgb=False, float32=True)
        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], dim=0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def load_tiff_image(self, path, key):
        """Load a TIFF image from the given path."""
        img_bytes = self.file_client.get(path, key)
        try:
            img = tifffile.imread(io.BytesIO(img_bytes))
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            raise Exception(f"{key} path {path} not working: {e}")
        return img

    @staticmethod
    def padding_dp(img_lqL, img_lqR, img_gt, gt_size):
        """Pad dual-pixel images and GT to the specified size."""
        h, w = img_gt.shape[:2]
        pad_h = max(0, gt_size - h)
        pad_w = max(0, gt_size - w)
        if pad_h > 0 or pad_w > 0:
            padding = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (img_gt.ndim - 2)
            img_lqL = np.pad(img_lqL, padding, mode='reflect')
            img_lqR = np.pad(img_lqR, padding, mode='reflect')
            img_gt = np.pad(img_gt, padding, mode='reflect')
        return img_lqL, img_lqR, img_gt

    @staticmethod
    def paired_random_crop_dp(img_lqL, img_lqR, img_gt, gt_size):
        """Randomly crop dual-pixel images and GT to the specified size."""
        h, w = img_gt.shape[:2]
        top = random.randint(0, h - gt_size)
        left = random.randint(0, w - gt_size)
        slices = [slice(top, top + gt_size), slice(left, left + gt_size)] + [slice(None)] * (img_gt.ndim - 2)
        img_lqL = img_lqL[tuple(slices)]
        img_lqR = img_lqR[tuple(slices)]
        img_gt = img_gt[tuple(slices)]
        return img_lqL, img_lqR, img_gt

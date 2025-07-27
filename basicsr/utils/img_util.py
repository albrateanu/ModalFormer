#basicsr/utils/img_util.py
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
import tifffile
import io


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Convert numpy arrays to torch tensors.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to convert BGR to RGB.
        float32 (bool): Whether to convert to float32.

    Returns:
        list[tensor] | tensor: Converted tensor images.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.ndim == 3 and img.shape[2] == 3 and bgr2rgb:
            img = img[..., ::-1]  # BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch tensor to image numpy array.

    Args:
        tensor (Tensor or list[Tensor]): Input tensor(s).
        rgb2bgr (bool): Whether to convert RGB to BGR.
        out_type (numpy type): Output data type.
        min_max (tuple[int]): Min and max values for clamping.

    Returns:
        ndarray: Converted image array.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'Expected tensor or list of tensors, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        img_np = _tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC

        if img_np.shape[2] == 3 and rgb2bgr:
            img_np = img_np[..., ::-1]  # RGB to BGR

        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    return result[0] if len(result) == 1 else result


def imfrombytes(content, float32=False):
    """Read a TIFF image from bytes.

    Args:
        content (bytes): Image bytes from files or streams.
        float32 (bool): Whether to convert to float32 and normalize to [0, 1].

    Returns:
        ndarray: Loaded image array.
    """
    try:
        img = tifffile.imread(io.BytesIO(content))
        if float32:
            img = img.astype(np.float32) / np.iinfo(img.dtype).max
        else:
            img = img.astype(np.float32)
        return img
    except Exception as e:
        raise Exception(f'Error reading image from bytes: {e}')


def imfrombytesDP(content, float32=False):
    """Read a 16-bit TIFF image from bytes.

    Args:
        content (bytes): Image bytes from files or streams.
        float32 (bool): Whether to convert to float32 and normalize to [0, 1].

    Returns:
        ndarray: Loaded image array.
    """
    return imfrombytes(content, float32=float32)


def padding(img_lq, img_gt, gt_size):
    """Pad images to the specified size.

    Args:
        img_lq (ndarray): Low-quality image.
        img_gt (ndarray): Ground truth image.
        gt_size (int): Target size.

    Returns:
        tuple: Padded images.
    """
    h, w = img_lq.shape[:2]
    pad_h = max(0, gt_size - h)
    pad_w = max(0, gt_size - w)
    if pad_h > 0 or pad_w > 0:
        pad_width = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (img_lq.ndim - 2)
        img_lq = np.pad(img_lq, pad_width, mode='reflect')
        img_gt = np.pad(img_gt, pad_width, mode='reflect')
    return img_lq, img_gt


def padding_DP(img_lqL, img_lqR, img_gt, gt_size):
    """Pad dual-pixel images and GT to the specified size.

    Args:
        img_lqL (ndarray): Left low-quality image.
        img_lqR (ndarray): Right low-quality image.
        img_gt (ndarray): Ground truth image.
        gt_size (int): Target size.

    Returns:
        tuple: Padded images.
    """
    h, w = img_gt.shape[:2]
    pad_h = max(0, gt_size - h)
    pad_w = max(0, gt_size - w)
    if pad_h > 0 or pad_w > 0:
        pad_width = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (img_gt.ndim - 2)
        img_lqL = np.pad(img_lqL, pad_width, mode='reflect')
        img_lqR = np.pad(img_lqR, pad_width, mode='reflect')
        img_gt = np.pad(img_gt, pad_width, mode='reflect')
    return img_lqL, img_lqR, img_gt


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (dict): Parameters for tifffile.imwrite.
        auto_mkdir (bool): Whether to create parent directories automatically.

    Returns:
        None
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    tifffile.imwrite(file_path, img, **(params or {}))


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images to be cropped.
        crop_border (int): Number of pixels to crop from each border.

    Returns:
        list[ndarray] | ndarray: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        slices = [slice(crop_border, -crop_border), slice(crop_border, -crop_border)]
        if isinstance(imgs, list):
            return [img[tuple(slices) + (slice(None),) * (img.ndim - 2)] for img in imgs]
        else:
            return imgs[tuple(slices) + (slice(None),) * (imgs.ndim - 2)]

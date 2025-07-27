import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
import skimage.metrics
import torch


def calculate_psnr(img1, img2, crop_border=0, input_order='CHW', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between RGB channels only.

    Args:
        img1 (ndarray or Tensor): Predicted images with shape (C, H, W) or (N, C, H, W).
        img2 (ndarray or Tensor): Target images with shape (C, H, W) or (N, C, H, W).
        crop_border (int): Number of pixels to crop from the border.
        input_order (str): 'HWC' or 'CHW'. Default: 'CHW'.

    Returns:
        float: PSNR value.
    """
    # Ensure inputs are numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # If input is batch, take the first image
    if img1.ndim == 4:
        img1 = img1[0]
    if img2.ndim == 4:
        img2 = img2[0]

    # Extract RGB channels
    if input_order == 'CHW':
        img1_rgb = img1[:3, :, :]
        img2_rgb = img2[:3, :, :]
    elif input_order == 'HWC':
        img1_rgb = img1[:, :, :3]
        img2_rgb = img2[:, :, :3]
        img1_rgb = img1_rgb.transpose(2, 0, 1)  # Convert to CHW
        img2_rgb = img2_rgb.transpose(2, 0, 1)
    else:
        raise ValueError("input_order must be 'HWC' or 'CHW'")

    # Convert to float64 for precision
    img1_rgb = img1_rgb.astype(np.float64)
    img2_rgb = img2_rgb.astype(np.float64)

    # Brightness normalization
    # img1_gray = img1_rgb.mean(axis=0)  # Average over channels
    # img2_gray = img2_rgb.mean(axis=0)
    # mean_pred = img1_gray.mean()
    # mean_target = img2_gray.mean()
    # img1_rgb = np.clip(img1_rgb * (mean_target / mean_pred), 0, 1)

    # Crop borders if needed
    if crop_border > 0:
        img1_rgb = img1_rgb[:, crop_border:-crop_border, crop_border:-crop_border]
        img2_rgb = img2_rgb[:, crop_border:-crop_border, crop_border:-crop_border]

    # Compute MSE
    mse = np.mean((img1_rgb - img2_rgb) ** 2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def prepare_for_ssim(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k//2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        img = conv(img)

        img = img.squeeze(0).squeeze(0)
        img = img[0::k, 0::k]
    return img.detach().cpu().numpy()

def prepare_for_ssim_rgb(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).float() #HxWx3

        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        new_img = []

        for i in range(3):
            new_img.append(conv(img[:, :, i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[0::k, 0::k])

    return torch.stack(new_img, dim=2).detach().cpu().numpy()

def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()


    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window,borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0, input_order='CHW', test_y_channel=False):
    """Calculate SSIM (Structural Similarity Index) between RGB channels only.

    Args:
        img1 (ndarray or Tensor): Predicted images with shape (C, H, W) or (N, C, H, W).
        img2 (ndarray or Tensor): Target images with shape (C, H, W) or (N, C, H, W).
        crop_border (int): Number of pixels to crop from the border.
        input_order (str): 'HWC' or 'CHW'. Default: 'CHW'.

    Returns:
        float: SSIM value.
    """
    # Ensure inputs are numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # If input is batch, take the first image
    if img1.ndim == 4:
        img1 = img1[0]
    if img2.ndim == 4:
        img2 = img2[0]

    # Extract RGB channels
    if input_order == 'CHW':
        img1_rgb = img1[:3, :, :]
        img2_rgb = img2[:3, :, :]
    elif input_order == 'HWC':
        img1_rgb = img1[:, :, :3]
        img2_rgb = img2[:, :, :3]
        img1_rgb = img1_rgb.transpose(2, 0, 1)  # Convert to CHW
        img2_rgb = img2_rgb.transpose(2, 0, 1)
    else:
        raise ValueError("input_order must be 'HWC' or 'CHW'")

    # Convert to float64 for precision
    img1_rgb = img1_rgb.astype(np.float64)
    img2_rgb = img2_rgb.astype(np.float64)

    # Brightness normalization
    # img1_gray = img1_rgb.mean(axis=0)  # Average over channels
    # img2_gray = img2_rgb.mean(axis=0)
    # mean_pred = img1_gray.mean()
    # mean_target = img2_gray.mean()
    # max_value = 1. if img1.max() <= 1 else 255.
    # img1_rgb = np.clip(img1_rgb * (mean_target / mean_pred), 0, max_value)

    # Crop borders if needed
    if crop_border > 0:
        img1_rgb = img1_rgb[:, crop_border:-crop_border, crop_border:-crop_border]
        img2_rgb = img2_rgb[:, crop_border:-crop_border, crop_border:-crop_border]

    # Calculate SSIM for each channel
    ssims = []
    for i in range(3):
        ssim_value = ssim(img1_rgb[i], img2_rgb[i])
        ssims.append(ssim_value)
    return np.array(ssims).mean()

def ssim(img1, img2):
    """Compute SSIM between two single-channel images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Apply Gaussian blur
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1 ** 2
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2 ** 2
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

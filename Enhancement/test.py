# test_trace.py

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import calculate_psnr, calculate_ssim
import cv2

from basicsr.utils.options import parse
from basicsr.data import create_dataset, create_dataloader

def main():
    parser = argparse.ArgumentParser(description='Image Enhancement Testing Script with Traced Model')
    parser.add_argument('--dataset', type=str, required=True, choices=['LOL_v1', 'LOL_v2_Real', 'LOL_v2_Synthetic', 'SDSD-indoor', 'SDSD-outdoor'],
                        help='Dataset to test on.')
    parser.add_argument('--gpus', type=str, default='0', help='GPU device IDs.')
    parser.add_argument('--count_params', action='store_true', help='Count and print the number of parameters in the traced model.')
    parser.add_argument('--print_trace', action='store_true', help='Print the model trace.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_config = {
        'LOL_v1': {
            'opt': 'Options/ModalFormer_LOL_v1.yml',
            'traced_model': 'pretrained_model/ModalFormer_traced_LOL_v1.pt'
        },
        'LOL_v2_Real': {
            'opt': 'Options/ModalFormer_LOL_v2_Real.yml',
            'traced_model': 'pretrained_model/ModalFormer_traced_LOL_v2_Real.pt'
        },
        'LOL_v2_Synthetic': {
            'opt': 'Options/ModalFormer_LOL_v2_Synthetic.yml',
            'traced_model': 'pretrained_model/ModalFormer_traced_LOL_v2_Synthetic.pt'
        },
        'SDSD-indoor': {
            'opt': 'Options/ModalFormer_SDSD_indoor.yml',
            'traced_model': 'pretrained_model/ModalFormer_traced_SDSD_indoor.pt'
        },
        'SDSD-outdoor': {
            'opt': 'Options/ModalFormer_SDSD_outdoor.yml',
            'traced_model': 'pretrained_model/ModalFormer_traced_SDSD_outdoor.pt'
        }
    }

    dataset_name = args.dataset
    if dataset_name not in dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    opt_path = dataset_config[dataset_name]['opt']
    traced_model_path = dataset_config[dataset_name]['traced_model']

    opt = parse(opt_path, is_train=False)
    opt['dist'] = False

    traced_model = torch.jit.load(traced_model_path, map_location=device)
    traced_model.eval()
    print(f"TorchScript model loaded from '{traced_model_path}'.")

    if args.count_params:
        num_params = sum(p.numel() for p in traced_model.parameters())
        print(f"Number of parameters in the traced model: {num_params}")

    if args.print_trace:
        print("\nModel Trace:")
        print(traced_model)
        print("\n")

    test_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt, num_gpu=1, dist=False, sampler=None)
            test_loaders.append((test_loader, dataset_opt))
            print(f"Loaded test dataset '{dataset_opt['name']}' with {len(test_set)} images.")

    for test_loader, test_dataset_opt in test_loaders:
        psnr_all = []
        ssim_all = []

        output_dir = os.path.join('results', test_dataset_opt['name'])
        os.makedirs(output_dir, exist_ok=True)
        idx = 0

        for data in tqdm(test_loader, desc=f"Testing on {test_dataset_opt['name']}"):
            lq = data['lq'].to(device)
            gt = data['gt'].to(device)

            factor = 16
            _, _, h, w = lq.size()
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            lq_padded = F.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')

            restored = self_ensemble(lq_padded, traced_model)

            restored = restored[:, :, :h, :w]

            restored_np = restored.cpu().numpy().squeeze().transpose(1, 2, 0)
            gt_np = gt.cpu().numpy().squeeze().transpose(1, 2, 0)

            # extract RGB from output tensor with 26 channels
            restored_np = restored_np[:, :, :3]
            gt_np = gt_np[:, :, :3]

            assert restored_np.shape == gt_np.shape, f"Shape mismatch: {restored_np.shape} vs {gt_np.shape}"

            mean_restored = restored_np.mean()
            mean_gt = gt_np.mean()
            restored_np = restored_np * (mean_gt / mean_restored)
            restored_np = np.clip(restored_np, 0, 1)

            psnr = calculate_psnr(gt_np, restored_np)
            ssim_value = calculate_ssim(gt_np, restored_np)
            psnr_all.append(psnr)
            ssim_all.append(ssim_value)

            restored_img = (restored_np * 255.0).astype(np.uint8)
            filename = f'image_{idx:04d}.png'
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
            idx += 1

        avg_psnr = np.mean(psnr_all)
        avg_ssim = np.mean(ssim_all)
        print(f"Results for {test_dataset_opt['name']}:")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

def self_ensemble(x, model):
    """
    Apply self-ensemble by flipping and rotating the input image.
    """
    def _transform(v, op):
        if op == 'v':
            return v.flip(-2)
        elif op == 'h':
            return v.flip(-1)
        elif op == 't':
            return v.transpose(-2, -1)
        return v

    ops = ['', 'v', 'h', 't']
    outputs = []
    for op in ops:
        x_transformed = _transform(x, op)
        with torch.no_grad():
            y_transformed = model(x_transformed)
        y = _transform(y_transformed, op)
        outputs.append(y)
    output = torch.stack(outputs).mean(dim=0)
    return output

if __name__ == '__main__':
    main()

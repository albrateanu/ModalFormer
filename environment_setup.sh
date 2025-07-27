#!/bin/bash

set -e

function check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Conda could not be found. Please install Anaconda or Miniconda before running this script."
        exit 1
    fi
}

check_conda

echo "Creating conda environment 'ModalFormer' with Python 3.9..."
conda create -n ModalFormer python=3.9 -y

echo "Initializing conda..."
eval "$(conda shell.bash hook)"

echo "Activating the 'ModalFormer' environment..."
conda activate ModalFormer

echo "Installing PyTorch and CUDA toolkit..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Installing additional Python packages via pip..."
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm

echo "Running setup.py..."
python setup.py develop --no_cuda_ext

echo "Environment setup is complete!"

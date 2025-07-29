# ✨ ModalFormer: Multimodal Transformer for Low-Light Image Enhancement

<div align="center">
  
**[Alexandru Brateanu](https://scholar.google.com/citations?user=ru0meGgAAAAJ&hl=en), [Raul Balmez](https://scholar.google.com/citations?user=vPC7raQAAAAJ&hl=en), [Ciprian Orhei](https://scholar.google.com/citations?user=DZHdq3wAAAAJ&hl=en), [Codruta Ancuti](https://scholar.google.com/citations?user=5PA43eEAAAAJ&hl=en), [Cosmin Ancuti](https://scholar.google.com/citations?user=zVTgt8IAAAAJ&hl=en)**

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2401.15204)
[![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-green)](https://huggingface.co/papers/2507.20388)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-green)](https://huggingface.co/albrateanu/ModalFormer)
</div>

### Abstract
*Low-light image enhancement (LLIE) is a fundamental yet challenging task due to the presence of noise, loss of detail, and poor contrast in images captured under insufficient lighting conditions. Recent methods often rely solely on pixel-level transformations of RGB images, neglecting the rich contextual information available from multiple visual modalities. In this paper, we present ModalFormer, the first large-scale multimodal framework for LLIE that fully exploits nine auxiliary modalities to achieve state-of-the-art performance. Our model comprises two main components: a Cross-modal Transformer (CM-T) designed to restore corrupted images while seamlessly integrating multimodal information, and multiple auxiliary subnetworks dedicated to multimodal feature reconstruction. Central to the CM-T is our novel Cross-modal Multi-headed Self-Attention mechanism (CM-MSA), which effectively fuses RGB data with modality-specific features—including deep feature embeddings, segmentation information, geometric cues, and color information—to generate information-rich hybrid attention maps. Extensive experiments on multiple benchmark datasets demonstrate ModalFormer’s state-of-the-art performance in LLIE. Pre-trained models and results are made available at https://github.com/albrateanu/ModalFormer*

## 🆕 Updates
- `29.07.2025` 🎉 The [**ModalFormer**](https://arxiv.org/abs/2401.15204) paper is now available! Check it out and explore our results and methodology.
- `28.07.2025` 📦 Pre-trained models and test data published! ArXiv paper version and HuggingFace demo coming soon, stay tuned!

## 📦 Contents
This repository contains pre-trained versions of ModalFormer, alongside all necessary code and data for testing and generating visual results on LOL-v1, LOL-v2 (Synthetic and Real), and SDSD (indoor and outdoor).

## ⚙️ Setup
For ease, utilize a Linux machine with CUDA-ready devices (GPUs).

To setup the environment, first run the provided setup script:

```bash
./environment_setup.sh
# or 
bash environment_setup.sh
```

Note: in case of difficulties, ensure ```environment_setup.sh``` is executable by running:

```bash
chmod +x environment_setup.sh
```

Give the setup a couple of minutes to run.

## 📁 Data and Pre-trained Models
We provide test datasets for LOL-v1, LOL-v2 (Real and Synthetic), and SDSD (indoor and outdoor) with multimodal information for inference at this [Google Drive](https://drive.google.com/file/d/1BRRvr30qnoz7fmniU3IkVSBbIkss9vYq/view?usp=drive_link) address. Unzip and place the ```data``` folder under the root directory of the project.

We also provide pre-trained models at this [Google Drive](https://drive.google.com/file/d/1qCC2x2Cj9ijLS9jqx9VQ3DTLacA7Xtt8/view?usp=drive_link) address. Unzip and place the ```pretrained_model``` folder under the root directory again.

## 🧪 Testing
For testing, we recommend using a GPU with at least 4 GB of VRAM. CPU is also an option, but that will make the process time-consuming.

Testing can be done by running:

```bash

# For LOL_v1
python Enhancement/test.py --dataset LOL_v1 

# For LOL_v2_Real
python Enhancement/test.py --dataset LOL_v2_Real

# For LOL_v2_Synthetic
python Enhancement/test.py --dataset LOL_v2_Synthetic

# For SDSD-indoor
python Enhancement/test.py --dataset SDSD-indoor

# For SDSD-outdoor
python Enhancement/test.py --dataset SDSD-outdoor

```

Note: the testing script contains two additional toggle arguments:
- ```--count_params```: prints the number of parameters in the model
- ```--print_trace```: prints the model trace (architecture)

Inference results will be saved under ```results/ValSet```. Please make sure you rename the ```ValSet``` subfolder as per your requirements, as re-running the testing script will overwrite its contents.

## 📚 Citation

```
@misc{brateanu2025modalformer,
      title={ModalFormer: Multimodal Transformer for Low-Light Image Enhancement}, 
      author={Alexandru Brateanu and Raul Balmez and Ciprian Orhei and Codruta Ancuti and Cosmin Ancuti},
      year={2025},
      eprint={2507.20388},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.20388}, 
}
```

## 🙏 Acknowledgements
We use [this codebase](https://github.com/caiyuanhao1998/Retinexformer) as foundation for our implementation.

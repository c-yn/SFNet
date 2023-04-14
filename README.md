# Selective Frequency Network for Image Restoration (ICLR2023)

Yuning Cui, Yi Tao, [Zhenshan Bing](https://scholar.google.com.hk/citations?user=eIz0XvMAAAAJ&hl=zh-CN&oi=ao), [Wenqi Ren](https://scholar.google.com.hk/citations?user=VwfgfR8AAAAJ&hl=zh-CN&oi=ao), Xinwei Gao, [Xiaochun Cao](https://scholar.google.com.hk/citations?user=PDgp6OkAAAAJ&hl=zh-CN&oi=ao), [Kai Huang](https://scholar.google.com.hk/citations?hl=zh-CN&user=70_wl8kAAAAJ), [Alois Knoll](https://scholar.google.com.hk/citations?user=-CA8QgwAAAAJ&hl=zh-CN&oi=ao)

[![](https://img.shields.io/badge/ICLR-Paper-blue.svg)](https://openreview.net/forum?id=tyZ1ChGZIKO)

## Update 
- 2023.04.14 	:tada: Release pre-trained models, code for three tasks (dehazing, motion deblurring, desnowing), and resulting images on most datasets.

>Image restoration aims to reconstruct the latent sharp image from its corrupted counterpart. Besides dealing with this long-standing task in the spatial domain, a few approaches seek solutions in the frequency domain in consideration of the large discrepancy between spectra of sharp/degraded image pairs. However, these works commonly utilize transformation tools, e.g., wavelet transform, to split features into several frequency parts, which is not flexible enough to select the most informative frequency component to recover. In this paper, we exploit a multi-branch and content-aware module to decompose features into separate frequency subbands dynamically and locally, and then accentuate the useful ones via channel-wise attention weights. In addition, to handle large-scale degradation blurs, we propose an extremely simple decoupling and modulation module to enlarge the receptive field via global and window-based average pooling. Integrating two developed modules into a U-Net backbone, the proposed Selective Frequency Network (SFNet) performs favorably against state-of-the-art algorithms on five image restoration tasks, including single-image defocus deblurring, image dehazing, image motion deblurring, image desnowing, and image deraining.


## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation
Please refer to respective directories.
## Results


## Citation
If you find this project useful for your research, please consider citing:
~~~
@inproceedings{cui2023selective,
  title={Selective Frequency Network for Image Restoration},
  author={Cui, Yuning and Tao, Yi and Bing, Zhenshan and Ren, Wenqi and Gao, Xinwei and Cao, Xiaochun and Huang, Kai and Knoll, Alois},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
~~~
## Acknowledgement
This project is mainly based on [MIMO-UNet](https://github.com/chosj95/MIMO-UNet).
## Contact
Should you have any question, please contact Yuning Cui.

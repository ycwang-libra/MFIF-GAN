# MFIF-GAN
This is an implementation for our paper ["MFIF-GAN: A New Generative Adversarial Network for Multi-Focus Image Fusion"](https://www.sciencedirect.com/science/article/abs/pii/S0923596521001260).
## Usage
### Data preparation
If you want to train MFIF-GAN on the proposed synthetic dataset based on an $\alpha$-matte model. Please download the [Pascal VOC2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and:
* follow the ```data_prepare.py``` to transform the ''segmentationclass'' image to ''focus_map_png'' and extract ''image_jpg'' from ''JPEGImage'' corresponding to ''focus_map_png''
* follow the ```data_generation.m``` to use ''focus_map'' and ''image_jpg'' to generate the training dataset ''A_jpg'' and ''B_jpg''
### Training


## Citation MFIF-GAN
If you find this work useful for your research, please cite our [paper](https://www.sciencedirect.com/science/article/abs/pii/S0923596521001260):
```
@article{wang2021mfif,
  title={MFIF-GAN: A new generative adversarial network for multi-focus image fusion},
  author={Wang, Yicheng and Xu, Shuang and Liu, Junmin and Zhao, Zixiang and Zhang, Chunxia and Zhang, Jiangshe},
  journal={Signal Processing: Image Communication},
  volume={96},
  pages={116295},
  year={2021},
  publisher={Elsevier}
}
```
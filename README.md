# MFIF-GAN
This is an implementation for our paper ["MFIF-GAN: A New Generative Adversarial Network for Multi-Focus Image Fusion"](https://www.sciencedirect.com/science/article/abs/pii/S0923596521001260).

## Usage
## Install
- Clone this repo:
```bash
git clone https://github.com/ycwang-libra/MFIF-GAN.git
cd MFIF-GAN
```
- Create a conda virtual environment and activate it:
```bash
conda create -n MFIF python=3.8 -y
conda activate MFIF
```
- Install `CUDA==10.2` with `cudnn7`
- Install `PyTorch==1.7.0` and `torchvision==0.8.1` with `CUDA==10.2`
- Install `matplotlab==3.2.2`, `numpy==1.18.5`

### Data preparation
If you want to train MFIF-GAN on the proposed synthetic dataset based on an $\alpha$-matte model. Please download the [Pascal VOC2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and then:

* follow the [data_preparation/VOC_prepare.py](data_preparation/VOC_prepare.py) to extract ''image_jpg'' from ''JPEGImage'' corresponding to ''SegmentationClass''.
* follow the [data_preparation/data_generation.m](data_preparation/data_generation.m) to transform the ''SegmentationClass'' image to ''focus_map_png''. Then use ''focus_map_png'' and ''image_jpg'' to generate the training dataset ''A_jpg'' and ''B_jpg''.
  
### Training
Use this dataset to train your MFIF-GAN, which may have better performance than ours:
```bash
python main.py --mode train --root_train [training data path]
```

### Testing with pre-trained model
The pre-trained models are also provided as [MFIF-GAN/models/110000-D.ckpt](MFIF_GAN/models/110000-D.ckpt) and [MFIF-GAN/models/110000-G.ckpt](MFIF_GAN/models/110000-G.ckpt). You can use
```bash
python main.py --mode test --batch_size 1 --test_iters 110000 --test_dataset Lytro --root_test [test data path]
```
to fuse ```Lytro``` or other multi-focus images. And the test result will be located in [MFIF_GAN/Fusion_result](MFIF_GAN/Fusion_result).
### Results
And we provide our test results on three datasets ('Lytro', 'MFFW', 'Grayscale') in [Fusion_results](Fusion_results).

## Citing MFIF-GAN
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

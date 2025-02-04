# UD-Mamba
***
#### [UD-Mamba: A pixel-level uncertainty-driven Mamba model for medical image segmentation]

**Weiren Zhao, Feng Wang, Yanran Wang, Yutong Xie, Qi Wu, and Yuyin Zhou**

<img src="img\Ud_block.png" width="70%" height="auto">
<img src="img\UD-Mamba.png" width="100%" height="auto">


## Results
<img src="img\ISIC-Path.png" width="70%" height="auto">

<img src="img\ACDC.png" width="70%" height="auto">


## Requirements
* Pytorch, MONAI 
* Some basic python packages: Torchio, Numpy, Scikit-image, SimpleITK, Scipy, Medpy, nibabel, tqdm ......

```shell
cd casual-conv1d

python setup.py install
```

```shell
cd mamba

python setup.py install
```


## Usage
Train Mamba-UNet
```shell
python train_fully_supervised_2D_UD-Mamba.py --root_path ../data/ACDC --exp ACDC/VIM --model mambaunet --max_iterations 10000 --batch_size 24  --num_classes 4 
```

## Acknowledgement
Mamba-Unet [Link](https://github.com/ziyangwang007/Mamba-UNet), SSL4MIS [Link](https://github.com/HiLab-git/SSL4MIS)
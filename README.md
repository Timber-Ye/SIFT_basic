# sift-remake
Personal implementation of SIFT(Scale-Invariant Feature Transform) Operator

## Brief Introduction
代码参考自苏黎世大学(UZH)[2022年秋季移动机器人视觉算法课程](课程网站：https://rpg.ifi.uzh.ch/teaching.html)，在其提供代码的基础上，对构造高斯金字塔、生成特征描述子等过程均进行了优化。算法被封装在`sift`目录下，构成一个`Python`包，可以在其他脚本文件中调用其所实现的**SIFT特征提取和匹配**方法。

`sift`目录下各文件介绍：

```zsh
.
├── __init__.py  # 包（package）标识文件
├── descriptor.py  # 计算特征描述子
├── mysift.py  # 匹配描述子
└── pyramid.py  # 构建DoG Pyramid，并提取SIFT特征点
```

开发过程中使用了python 3.8，虚拟环境中安装[numpy](https://pypi.org/project/numpy/)，[opencv](https://pypi.org/project/opencv-python/)即可。环境安装说明（未验证，仅供参考）：

```zsh
# 创建conda虚拟环境
conda create -n mysift python=3.8 numpy
# 进入conda虚拟环境
conda activate mysift
# 安装opencv
pip install opencv-python
```

## Keypoint Detection
输入一幅图像，提取其中的SIFT关键点。脚本运行命令示例：

```zsh
python sift_detect.py \
       --file_dir images/nvidia-3.jpg \ # input image
       -o 5 \ # number of octaves
       -s 3 \ # number of scales per octave
       -rescale 0.3 \ # rescale images to make it faster
       --sigma 1.0 \ # use for Gaussian blurring
       --output_dir images/experiment/nvidia-3-keypoints.jpg \ # output destination
       -t 5e-2 # threshold for detection
```

## Keypoint Matching
输入两幅图像，分别对它们进行\sift 特征提取与描述，并在两组描述子之间进行匹配。脚本运行命令示例：

```zsh
python image_matching.py \
       --file_dir_1 images/nvidia-3.jpg \ # input image 1
       --file_dir_2 images/nvidia-4.jpg \ # input image 2
       -o 5 \ # number of octaves
       -s 3 \ # number of scales per octave
       --rescale 0.3 \
       --output_dir images/experiment/image_matching.pdf \ # output destination
       --sigma 1.0 \ # use for Gaussian blurring
       -t 0.05 # threshold for detection
```

### If practitioners are having any confusion about this repository, plz feel free to open an Issue to have a discussion.

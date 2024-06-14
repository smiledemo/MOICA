# MOICA 项目

## 项目简介

此项目实现了“基于深度对抗域适应的多源开放集图像分类模型”中的 MOICA 模型。该模型通过特征提取、域分类和细粒度预测来实现对多源开放集图像的分类。

## 数据准备
---
- [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/): Art, Clipart, Product, Real World
- [Office31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/): Amazon, DSLR, Webcam
- [Digits](MNIST, SVHN, USPS)

| 数据集 | 域 | 角色 | 图片数量 | 类别数量 |
|:-:|:-:|:-:|:-:|:-:|
| Office-Home | Art <br> Clipart <br> Product <br> Real World | 源域 / 目标域 | 各有多个 | 65 |
| Office31 | Amazon <br> DSLR <br> Webcam | 源域 / 目标域 | 各有多个 | 31 |
| Digits | MNIST <br> SVHN <br> USPS | 源域 / 目标域 | 各有多个 | 10 |

(1) 要提取预训练的 ResNet-50 特征，
```shell
./data/extract_resnet_features.ipynb
```

（2）基于样本的标签收集所有样本的属性
```shell
./data/check_data.ipynb
```

## Dependencies
---
- Python 3.7
- Pytorch 1.1


## Training
---
### Step 1: 
```shell
./data/refine_cluster_samples.ipynb

```

### Step 2: 
```shell
python main.py --mode train --dataset <数据集名称> --domain <域名称> --epochs <训练轮数> --batch_size <批次大小> --learning_rate <学习率> --model_name <模型名称>

```





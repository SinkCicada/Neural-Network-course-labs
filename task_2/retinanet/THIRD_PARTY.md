# 第三方来源

## 模型实现

使用PyTorch/Torchvision 0.23.0的 `retinanet_resnet50_fpn`。

- 官方代码：https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/detection/retinanet.py
- 官方许可证：https://github.com/pytorch/vision/blob/v0.23.0/LICENSE
- 权重文档：https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html

本项目调用已有库中的开源网络，没有把上游网络源文件复制成自编网络。

## 预训练权重

`RetinaNet_ResNet50_FPN_Weights.COCO_V1`，来源：

https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth

本项目不重新训练权重。

## 示例图片

从相邻的Faster R-CNN项目复用公开演示图 `dog.jpg`，最初来源：

https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg

仓库：https://github.com/pytorch/hub

此图用于验证推理链路，不是书中的遥感图像。网络代码、权重和示例图的来源分别列出，未对第三方素材重新授予许可证。

# 第三方来源

## 网络实现

- 项目：PyTorch / Torchvision，版本0.23.0。
- 使用接口：`torchvision.models.detection.fasterrcnn_resnet50_fpn`。
- 官方源代码：https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/detection/faster_rcnn.py
- 项目许可证：https://github.com/pytorch/vision/blob/v0.23.0/LICENSE
- 本项目通过已安装的库调用网络，没有把上游网络源文件复制成自编网络。

## 预训练权重

- 标识：`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`。
- 下载：https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
- 说明：https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
- 通过 Torchvision 官方权重加载接口获取及校验，不在本项目中重新训练。

## 示例图片

- 来源：PyTorch Hub 仓库的公开演示图片。
- 原图：https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
- 仓库：https://github.com/pytorch/hub
- 用途：验证推理链路；不是本地采集图，也不是书中的水下图像。

本项目中的网络实现许可、权重及图片来源分别列出；图片和权重不因本项目封装而重新授予许可证。

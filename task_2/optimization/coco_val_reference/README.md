# COCO 2017 验证集汽车专项对照

本实验使用 Torchvision Faster R-CNN、RetinaNet 的 COCO_V1 权重和 Ultralytics YOLOv8s 的 COCO 预训练权重。

## 共同验证样本

图像和标注来自 [COCO 官方数据](https://cocodataset.org/#download) 的 val2017。标注从官方压缩包中提取，仅下载所需文件段，并核验 ZIP 成员 CRC。先按 car、truck、bus 的真值框数分为 1–2、4–6、10 个及以上三组，再各取图像编号最小的两张。排除上述类别中有 crowd 或无效框的图片。选择在推理前完成。

| 图像 | 汽车真值框数 |
|---|---:|
| [000000000724.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000724.jpg) | 2 |
| [000000002006.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000002006.jpg) | 1 |
| [000000006723.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000006723.jpg) | 6 |
| [000000009891.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000009891.jpg) | 4 |
| [000000017627.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000017627.jpg) | 12 |
| [000000039484.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000039484.jpg) | 11 |

## 结果

三模型使用同一组 6 张图和 36 个汽车真值框，整图推理，置信度 0.25，模型常规 NMS IoU 0.50。YOLOv8s 输入为 640，Torchvision 模型沿用自身输入缩放。按分数降序在同类别中进行一对一匹配，匹配 IoU ≥ 0.50。

| 模型 | TP | FP | FN | Precision | Recall | F1 | 加车辆去重后 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 30 | 26 | 6 | 0.536 | 0.833 | 0.652 | 0.682 |
| RetinaNet | 32 | 27 | 4 | 0.542 | 0.889 | 0.674 | 0.697 |
| YOLOv8s | 24 | 7 | 12 | 0.774 | 0.667 | 0.716 | 0.716 |

车辆去重对照直接使用相同候选，追加项目已有的车辆组 NMS（IoU 0.60）。Faster R-CNN 的 FP 从 26 降到 22，TP 保持 30。RetinaNet 的 FP 从 27 降到 22，同时少保留 1 个 TP。YOLOv8s 的计数保持一致。该结果是固定阈值下的小样本检查，模型总体性能需要完整数据评估。

结果表的逐图匹配、框坐标和分数存于 `results.json`，真值及图片许可信息见 `annotations_subset.json`。安装目标检测依赖后运行 `evaluate_reference.py` 可从已有推理结果重新计算。


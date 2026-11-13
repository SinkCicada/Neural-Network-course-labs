# RetinaNet

在仓库根目录安装 `requirements-detection.txt`。

权重下载：[官方文件](https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth)，保存为 `task_2/retinanet/weights/checkpoints/retinanet_resnet50_fpn_coco-eeacb38b.pth`。

```powershell
python task_2/retinanet/run.py --device cpu
python task_2/retinanet/run.py --help
```

支持 `--source` 指定图片或目录，`--output` 指定结果目录。默认输出到本模型目录下的 runs。

`--nms-method soft --soft-nms-iou 0.40` 启用线性 Soft-NMS，默认使用 hard 流程。

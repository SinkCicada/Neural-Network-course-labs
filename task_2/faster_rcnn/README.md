# Faster R-CNN

在仓库根目录安装 `requirements-detection.txt`。

在仓库根目录执行 `python download_weights.py` 获取 Release 中的权重。也可手动下载：[官方文件](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)，保存为 `task_2/faster_rcnn/weights/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`。

首次运行也可由 Torchvision 自动下载权重。

```powershell
python task_2/faster_rcnn/run.py --device cpu
python task_2/faster_rcnn/run.py --help
```

支持 `--source` 指定图片或目录，`--output` 指定结果目录。默认输出到本模型目录下的 runs。

# YOLOv8

在仓库根目录安装 `requirements-detection.txt`。

权重下载：[官方文件](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)，保存为 `task_2/yolov8/weights/yolov8n.pt`。

YOLOv8s：[官方下载](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt)。保存到同一 weights 目录，运行时通过 `--weights task_2/yolov8/weights/yolov8s.pt` 指定。

```powershell
python task_2/yolov8/run.py --device cpu
python task_2/yolov8/run.py --help
```

支持 `--source` 指定图片或目录，`--output` 指定结果目录。默认输出到本模型目录下的 runs。

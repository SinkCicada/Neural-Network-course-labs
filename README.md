# 神经网络与深度学习课程实验

两个实验：用 NumPy 实现 BP 网络识别点阵字母；使用 Faster R-CNN、RetinaNet 和 YOLOv8 做目标检测，并比较阈值、去重和切片的效果。

## 环境

建议 Python 3.11。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-bp.txt
# 运行目标检测实验时再安装：
python -m pip install -r requirements-detection.txt
```

目标检测依赖文件使用 CPU 版 PyTorch。四份检测权重放在本仓库的 [Release](https://github.com/SinkCicada/Neural-Network-course-labs/releases/tag/weights-v1) 中，BP 阈值实验的权重已保存在 `task_1/threshold_results/fixed_model.npz`。

## 下载完整权重

克隆仓库或下载并解压 ZIP 后，在仓库根目录执行：

```powershell
python download_weights.py
```

脚本下载 Faster R-CNN、RetinaNet、YOLOv8n 和 YOLOv8s 四份权重，共约 333 MB，自动保存到各模型所需目录，并核对 SHA-256。已有完整文件会跳过。文件清单见 `weights_manifest.json`，无需安装 Git LFS。

## 实验一：点阵字母识别

网络结构为 35 → H → 6，识别 A、E、F、H、L、T。手写前向传播、误差反向传播及梯度下降，NumPy 只负责数组和矩阵运算。

```powershell
python task_1/letter_bp.py
python task_1/threshold_experiment.py
```

第一个脚本比较 3～7 个隐含层节点在像素扰动下的表现；第二个脚本固定网络，在验证集选择输出阈值，再在留出测试集上比较正确、误识和拒识。结果分别写入 `letter_bp_results` 和 `threshold_results`，仓库中保留了本次实验的曲线及 CSV 数据。

## 实验二：目标检测

三个模型使用 COCO 预训练权重，在本地进行推理。代码实现图片读取、结果导出、车辆组去重和重叠切片合并；RetinaNet 另有线性 Soft-NMS 对照。

| 模型 | 入口 | 权重说明 |
|---|---|---|
| Faster R-CNN ResNet50-FPN | `task_2/faster_rcnn/run.py` | [下载与运行](task_2/faster_rcnn/README.md) |
| RetinaNet ResNet50-FPN | `task_2/retinanet/run.py` | [下载与运行](task_2/retinanet/README.md) |
| YOLOv8n / YOLOv8s | `task_2/yolov8/run.py` | [下载与运行](task_2/yolov8/README.md) |

```powershell
python task_2/faster_rcnn/run.py --source task_2/faster_rcnn/assets/dog.jpg --device cpu
python task_2/retinanet/run.py --source task_2/retinanet/assets/dog.jpg --device cpu
python task_2/yolov8/run.py --source task_2/yolov8/assets/bus.jpg --device cpu
```

检测结果和框坐标默认保存到各模型的 `runs` 目录。使用 `--help` 查看全部参数。

- `--threshold`：置信度筛选门槛。
- `--vehicle-nms-iou`：car、truck、bus 之间的重叠框去重门槛；`--no-vehicle-nms` 关闭此步骤。
- `--tile-size 640 --tile-overlap 0.25`：对重叠切片和整图分别推理，将结果还原到原图后合并。
- `--hide-labels`：密集图像只显示边框。

六张 COCO 验证图片及 COCO8 的四张验证图片和标注已包含在仓库中。共同的六张 COCO 验证图片、标注来源与比较结果见 [车辆检测对照](task_2/optimization/coco_val_reference/README.md)。

## 检查

```powershell
python task_1/test_threshold_experiment.py
python -m unittest discover -s task_2/optimization -p "test_*.py"
```

图片、权重和上游库的来源见各模型目录中的 `THIRD_PARTY.md`。

## 重新计算检测对照

```powershell
python task_2/optimization/coco_val_reference/evaluate_reference.py
python task_2/optimization/evaluate_sanity.py
```

第一条命令根据保存的检测框和真实标注重新计算六图对照；第二条使用 YOLOv8n/s 权重在 COCO8 四张验证图上重新推理。COCO8 仅用于小型运行检查。

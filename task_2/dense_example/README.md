# 补充图片

`parking_1.jpg`、`parking_2.jpg` 为实验使用的停车场图片。

`crowd_unannotated.jpg` 为未标注的人群图片，来源：https://habrastorage.org/webt/tj/gk/ch/tjgkch5v0x-tubycgzp3pfbrtas.jpeg 。该图片仅用于观察密集目标的检测情况。

示例：

```powershell
python task_2/faster_rcnn/run.py --source task_2/dense_example/crowd_unannotated.jpg --tile-size 640 --tile-overlap 0.25 --hide-labels
```

# 第三方来源

- 网络实现：Ultralytics 8.3.100，https://github.com/ultralytics/ultralytics/tree/v8.3.100
- 开源许可证：https://github.com/ultralytics/ultralytics/blob/v8.3.100/LICENSE （AGPL-3.0）。
- 权重：Ultralytics官方YOLOv8n COCO检测权重，https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
- 狗的示例图：从相邻Faster R-CNN项目复用，最初来源 https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
- 密集人群示例图：从 `dense_example/crowd_unannotated.jpg` 复用，最初来源 https://habrastorage.org/webt/tj/gk/ch/tjgkch5v0x-tubycgzp3pfbrtas.jpeg

开源库负责网络、权重读取和检测后处理，本项目编写本地输入、模型调用、Pillow画框、JSON导出及Windows入口。第三方代码、权重和图片按各自来源的许可条件使用，本项目不对其重新授予许可证。

- 公交车示例图：Ultralytics官方仓库，https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.100/ultralytics/assets/bus.jpg

"""Faster R-CNN ResNet50-FPN：仅使用官方预训练权重进行图片推理。"""

import argparse
import sys
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from inference_utils import run_tiled, suppress_vehicle_duplicates, CAR_CLASSES
DEMO_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="图片路径或图片文件夹；省略时使用示例图")
    parser.add_argument("--output", type=Path, help="结果目录；默认在 runs 下新建时间戳目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值，默认 0.5")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU 计算线程数，默认 4")
    parser.add_argument("--tile-size", type=int, default=0, help="密集目标切片边长；0关闭，建议640")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="切片重叠比例")
    parser.add_argument("--merge-iou", type=float, default=0.5, help="跨切片同类别NMS阈值")
    parser.add_argument("--max-detections", type=int, default=100, help="每张图最大框数")
    parser.add_argument("--hide-labels", action="store_true", help="只画框，避免密集目标文字遮挡")
    parser.add_argument("--vehicle-nms-iou", type=float, default=0.6, help="车辆跨类别去重IoU阈值，默认0.6")
    parser.add_argument("--no-vehicle-nms", action="store_false", dest="vehicle_nms",
                        help="关闭车辆跨类别去重，恢复原始分类别NMS结果")
    parser.add_argument("--cars-only", action="store_true", help="仅保留car/truck/bus汽车类别，排除摩托车等")
    args = parser.parse_args()
    if not 0 <= args.vehicle_nms_iou <= 1:
        parser.error("--vehicle-nms-iou 必须在[0,1]内")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须在 [0, 1] 内")
    if args.threads < 1:
        parser.error("--threads 必须大于 0")
    if args.tile_size < 0 or not 0 <= args.tile_overlap < 1 or not 0 <= args.merge_iou <= 1:
        parser.error("切片边长必须非负，重叠比例须在[0,1)，合并IoU须在[0,1]")
    if args.max_detections < 1:
        parser.error("--max-detections 必须大于0")
    return args


def get_images(source):
    if source is None:
        source = ROOT / "assets" / "dog.jpg"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading public demo image...", flush=True)
            with urlopen(DEMO_URL, timeout=60) as response:
                data = response.read()
            source.write_bytes(data)
    source = source.resolve()
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        if images:
            return images
    raise ValueError("未找到支持的图片：{}".format(source))


def draw_detections(image, detections, show_labels=True):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(16, round(min(image.size) / 40))
    font = ImageFont.load_default(size=font_size)
    line_width = max(2, round(min(image.size) / 250))
    palette = ["#00B8FF", "#FF7832", "#22DD88", "#FF55BB", "#FFCC33"]
    for item in detections:
        color = palette[item["class_id"] % len(palette)]
        box = item["box_xyxy"]
        draw.rectangle(box, outline=color, width=line_width)
        if not show_labels:
            continue
        label = "{} {:.3f}".format(item["class_name"], item["score"])
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(box[0], image.width - width - 8))
        y = max(0, box[1] - height - 10)
        draw.rectangle((x, y, x + width + 8, y + height + 8), fill=color)
        draw.text((x + 4, y + 4 - bounds[1]), label, font=font, fill="black")
    return canvas


def main():
    args = parse_args()
    images = get_images(args.source)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA，请使用 --device cpu")
    device = torch.device(device_name)
    torch.set_num_threads(args.threads)
    # 下载的完整检测模型权重保存在本项目内，后续运行可离线读取。
    torch.hub.set_dir(str(ROOT / "weights"))
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    print("Loading Faster R-CNN ResNet50-FPN (COCO_V1), device={}".format(device), flush=True)
    model = fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, progress=False,
        box_score_thresh=args.threshold,
        box_detections_per_img=args.max_detections,
    ).to(device).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    output = (args.output or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(images, start=1):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        forward_ms = [0.0]

        def predict_image(view):
            tensor = preprocess(view).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_ms[0] += (time.perf_counter() - started) * 1000
            found = []
            for box, label, score in zip(prediction["boxes"].cpu().tolist(),
                                          prediction["labels"].cpu().tolist(),
                                          prediction["scores"].cpu().tolist()):
                if score >= args.threshold and (not args.cars_only or categories[label] in CAR_CLASSES):
                    found.append({"class_id": label, "class_name": categories[label],
                                  "score": score, "box_xyxy": box})
            return found

        pipeline_start = time.perf_counter()
        if args.tile_size:
            detections, slicing = run_tiled(image, predict_image, tile_size=args.tile_size,
                overlap=args.tile_overlap, merge_iou=args.merge_iou, max_detections=args.max_detections)
        else:
            detections = predict_image(image)
            slicing = {"tile_count": 0, "total_calls": 1, "include_full": True}
        detections, vehicle_nms = suppress_vehicle_duplicates(
            detections, iou_threshold=args.vehicle_nms_iou, enabled=args.vehicle_nms)
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        inference_ms = forward_ms[0]
        # 加序号保留同名、不同扩展名图片的独立结果。
        name = "{:03d}_{}".format(index, path.stem)
        result = {
            "source": str(path), "image_width": image.width, "image_height": image.height,
            "coordinate_system": "EXIF-corrected image pixels, [x1, y1, x2, y2]",
            "threshold": args.threshold, "inference_ms": inference_ms,
            "classes_filter": sorted(CAR_CLASSES) if args.cars_only else None,
            "pipeline_ms": pipeline_ms, "slicing": slicing, "vehicle_nms": vehicle_nms,
            "detections": detections,
        }
        draw_detections(image, detections, show_labels=not args.hide_labels).save(output / (name + "_detected.png"))
        (output / (name + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(result)
        print("{}: {} detections, {:.1f} ms".format(path.name, len(detections), inference_ms), flush=True)
    summary = {
        "model": "fasterrcnn_resnet50_fpn", "weights": "COCO_V1",
        "weights_url": weights.url, "device": str(device),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cpu_threads": args.threads},
        "timing_scope": "inference_ms sums model forward across all views; pipeline_ms includes preprocessing, decoding and cross-tile merge; includes final vehicle NMS; excludes file I/O and rendering",
        "image_count": len(records), "total_detections": sum(len(r["detections"]) for r in records),
        "results": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results: {}".format(output), flush=True)


if __name__ == "__main__":
    main()

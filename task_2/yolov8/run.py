"""YOLOv8：使用本地 COCO 预训练权重进行图片推理，不训练。"""

import argparse
import sys
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from inference_utils import run_tiled, suppress_vehicle_duplicates, CAR_CLASSES
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".config")
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch
import torchvision
import ultralytics
from ultralytics import YOLO, settings
from PIL import Image, ImageDraw, ImageFont, ImageOps

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
WEIGHT_URLS = {name: WEIGHTS_URL.replace("yolov8n.pt", name) for name in ("yolov8n.pt", "yolov8s.pt")}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="图片或图片文件夹；省略时检测 assets 中的示例")
    parser.add_argument("--output", type=Path, help="输出目录；默认在 runs 下新建时间戳目录")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值，默认0.5")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU阈值，默认0.5")
    parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸，默认640")
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=0, help="密集目标切片边长；0关闭，建议640")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="切片重叠比例")
    parser.add_argument("--merge-iou", type=float, default=0.5, help="跨切片同类别NMS阈值")
    parser.add_argument("--hide-labels", action="store_true", help="只画框，避免密集目标文字遮挡")
    parser.add_argument("--vehicle-nms-iou", type=float, default=0.6, help="车辆跨类别去重IoU阈值，默认0.6")
    parser.add_argument("--no-vehicle-nms", action="store_false", dest="vehicle_nms",
                        help="关闭车辆跨类别去重，恢复原始分类别NMS结果")
    parser.add_argument("--cars-only", action="store_true", help="仅保留car/truck/bus汽车类别，排除摩托车等")
    args = parser.parse_args()
    if not 0 <= args.vehicle_nms_iou <= 1:
        parser.error("--vehicle-nms-iou 必须在[0,1]内")
    if not 0 <= args.threshold <= 1 or not 0 <= args.iou <= 1:
        parser.error("--threshold 和 --iou 必须在[0,1]内")
    if min(args.imgsz, args.max_detections, args.threads) < 1:
        parser.error("--imgsz、--max-detections、--threads 必须大于0")
    if args.tile_size < 0 or not 0 <= args.tile_overlap < 1 or not 0 <= args.merge_iou <= 1:
        parser.error("切片边长必须非负，重叠比例须在[0,1)，合并IoU须在[0,1]")
    if args.max_detections < 1:
        parser.error("--max-detections 必须大于0")
    return args


def get_images(source):
    source = (source or ROOT / "assets").resolve()
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            return images
    raise SystemExit("未找到支持的图片：{}".format(source))


def draw_detections(image, detections, show_labels=True):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(16, round(min(image.size) / 40))
    font = ImageFont.load_default(size=font_size)
    palette = ["#00B8FF", "#FF7832", "#22DD88", "#FF55BB", "#FFCC33"]
    for item in detections:
        color = palette[item["class_id"] % len(palette)]
        box = item["box_xyxy"]
        draw.rectangle(box, outline=color, width=max(2, round(min(image.size) / 250)))
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
    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit("缺少本地权重，不会自动下载。\n链接：{}\n保存到：{}".format(WEIGHT_URLS.get(weights.name, "请使用本地兼容YOLO检测权重"), weights))
    images = get_images(args.source)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("当前没有可用CUDA，请使用 --device cpu")
    torch.set_num_threads(args.threads)
    settings.update({"sync": False})
    model = YOLO(str(weights), task="detect")
    output = (args.output or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    print("Loading {} (COCO), device={}".format(weights.stem, device), flush=True)
    for index, path in enumerate(images, 1):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        total_speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

        def predict_image(view):
            prediction = model.predict(
                source=view, conf=args.threshold, iou=args.iou, imgsz=args.imgsz,
                max_det=args.max_detections, device="0" if device == "cuda" else "cpu",
                save=False, verbose=False,
            )[0]
            for key in total_speed:
                total_speed[key] += prediction.speed[key]
            boxes = prediction.boxes.cpu()
            return [{"class_id": int(cls), "class_name": prediction.names[int(cls)],
                     "score": score, "box_xyxy": box}
                    for box, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
                    if not args.cars_only or prediction.names[int(cls)] in CAR_CLASSES]

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
        name = "{:03d}_{}".format(index, path.stem)
        record = {"source": str(path), "image_width": image.width, "image_height": image.height,
                  "coordinate_system": "EXIF-corrected original image pixels, [x1,y1,x2,y2]",
                  "threshold": args.threshold, "nms_iou": args.iou, "imgsz": args.imgsz,
                  "classes_filter": sorted(CAR_CLASSES) if args.cars_only else None,
                  "speed_ms": total_speed, "pipeline_ms": pipeline_ms,
                  "slicing": slicing, "vehicle_nms": vehicle_nms, "detections": detections}
        draw_detections(image, detections, show_labels=not args.hide_labels).save(output / (name + "_detected.png"))
        (output / (name + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        print("{}: {} detections, inference {:.1f} ms".format(path.name, len(detections), total_speed["inference"]), flush=True)
    summary = {
        "model": weights.stem, "class_names": model.names, "weights_path": str(weights),
        "weights_url": WEIGHT_URLS.get(weights.name),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "device": device, "image_count": len(records), "total_detections": sum(len(r["detections"]) for r in records),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
                        "cpu_threads": args.threads},
        "timing_scope": "speed_ms sums Ultralytics stage timings across all views; pipeline_ms includes prediction and cross-tile merge, includes final vehicle NMS; excludes loading and rendering",
        "results": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results: {}".format(output), flush=True)


if __name__ == "__main__":
    main()

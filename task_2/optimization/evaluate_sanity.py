"""COCO8 val 的小型带标注回归检查；单一阈值 P/R/F1，不计算 mAP。"""

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO, settings

DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
WARNING = (
    "COCO8仅用于小型sanity/回归检查。其4张val图片来自COCO train2017，"
    "可能与预训练数据重叠，不是独立泛化评估；这里的P/R/F1不是mAP。"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", nargs="+", type=Path, help="一个或多个本地COCO检测权重；默认YOLOv8n和YOLOv8s")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "coco8")
    p.add_argument("--output", type=Path, help="输出JSON；默认写到results/时间戳/coco8_sanity.json")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if any(not 0 <= x <= 1 for x in (args.conf, args.nms_iou, args.match_iou)):
        p.error("conf和IoU阈值必须在[0,1]内")
    if min(args.imgsz, args.max_det, args.threads) < 1:
        p.error("imgsz、max-det、threads必须大于0")
    return args


def iou_xyxy(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - intersection
    return intersection / union if union else 0.0


def read_labels(path, width, height):
    # COCO8提供常规YOLO框标签，没有crowd/ignore标记；不调用COCO官方AP评估。
    if not path.is_file():
        raise FileNotFoundError("缺少真值标签：{}".format(path))
    labels = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError("需要class xc yc w h五列标签：{}:{}".format(path, line_no))
        class_value, xc, yc, bw, bh = map(float, values)
        cls = int(class_value)
        if class_value != cls or not 0 <= cls < 80 or not all(0 <= v <= 1 for v in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError("标签数值不合法：{}:{}".format(path, line_no))
        labels.append({
            "class_id": cls,
            "box_xyxy": [(xc - bw / 2) * width, (yc - bh / 2) * height,
                         (xc + bw / 2) * width, (yc + bh / 2) * height],
        })
    return labels


def match_detections(predictions, truth, threshold):
    """按分数降序逐个匹配同类别的最高IoU未匹配真值，一对一且不重复计TP。"""
    unmatched = set(range(len(truth)))
    matches = []
    false_positives = []
    ordered = sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True)
    for pred_index in ordered:
        prediction = predictions[pred_index]
        candidates = [i for i in unmatched if truth[i]["class_id"] == prediction["class_id"]]
        best = max(candidates, key=lambda i: iou_xyxy(prediction["box_xyxy"], truth[i]["box_xyxy"]), default=None)
        overlap = iou_xyxy(prediction["box_xyxy"], truth[best]["box_xyxy"]) if best is not None else 0.0
        if best is not None and overlap >= threshold:
            unmatched.remove(best)
            matches.append({"prediction_index": pred_index, "truth_index": best, "iou": overlap})
        else:
            false_positives.append(pred_index)
    return {"tp": len(matches), "fp": len(false_positives), "fn": len(unmatched),
            "matches": matches, "false_positive_indices": false_positives,
            "false_negative_indices": sorted(unmatched)}


def scores(tp, fp, fn):
    # 无预测/无真值时，以0处理对应分母为0的指标，并在原始计数中保留事实。
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    args = parse_args()
    data = args.data.resolve()
    paths = sorted((data / "images" / "val").glob("*.jpg"))
    if len(paths) != 4:
        raise SystemExit("本检查要求COCO8的4张val图，实际找到{}张：{}".format(len(paths), data))
    weight_paths = args.weights or [ROOT.parent / "yolov8" / "weights" / name for name in ("yolov8n.pt", "yolov8s.pt")]
    weight_paths = [path.resolve() for path in weight_paths]
    for path in weight_paths:
        if not path.is_file():
            raise SystemExit("缺少本地权重，不会自动下载：{}".format(path))
    torch.set_num_threads(args.threads)
    settings.update({"sync": False})
    reports = []
    for path in weight_paths:
        model = YOLO(str(path), task="detect")
        if len(model.names) != 80 or model.names.get(0) != "person" or model.names.get(16) != "dog":
            raise SystemExit("本检查仅接受使用COCO 0–79类别映射的检测权重")
        images = []
        for image_path in paths:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            labels = read_labels(data / "labels" / "val" / (image_path.stem + ".txt"), *image.size)
            result = model.predict(image, device="cpu", imgsz=args.imgsz, conf=args.conf,
                                   iou=args.nms_iou, max_det=args.max_det, verbose=False, save=False)[0]
            predictions = [{"class_id": int(c), "class_name": model.names[int(c)], "score": float(s), "box_xyxy": b}
                           for b, c, s in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            matching = match_detections(predictions, labels, args.match_iou)
            images.append({"image": str(image_path), "width": image.width, "height": image.height,
                           "ground_truth_count": len(labels), "prediction_count": len(predictions),
                           "ground_truth": labels, "predictions": predictions, "matching": matching})
        tp = sum(x["matching"]["tp"] for x in images)
        fp = sum(x["matching"]["fp"] for x in images)
        fn = sum(x["matching"]["fn"] for x in images)
        report = {"weights": str(path), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "image_count": len(images), "tp": tp, "fp": fp, "fn": fn, **scores(tp, fp, fn), "images": images}
        reports.append(report)
        print("{}: TP={} FP={} FN={} P={:.4f} R={:.4f} F1={:.4f}".format(path.name, tp, fp, fn, report["precision"], report["recall"], report["f1"]), flush=True)
    output = (args.output or ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "coco8_sanity.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"purpose": "small labeled sanity/regression check; not mAP or independent generalization evaluation",
               "limitation_zh": WARNING, "dataset": "COCO8 validation subset, 4 images sourced from COCO train2017",
               "dataset_url": DATA_URL, "dataset_root": str(data),
               "conditions": {"imgsz": args.imgsz, "confidence": args.conf, "nms_iou": args.nms_iou,
                              "matching_iou": args.match_iou, "max_detections": args.max_det, "device": "cpu", "threads": args.threads},
               "matching": "confidence-descending greedy, same-class best-IoU unmatched ground truth, one-to-one",
               "aggregation": "micro counts pooled across four images; zero denominators reported as 0",
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__},
               "models": reports}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(WARNING, flush=True)
    print("Output:", output, flush=True)


if __name__ == "__main__":
    main()

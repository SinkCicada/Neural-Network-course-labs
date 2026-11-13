"""Score project predictions on the fixed COCO val2017 vehicle subset."""
from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from evaluate_sanity import match_detections, scores
sys.path.insert(0, str(ROOT.parents[1]))
from inference_utils import suppress_vehicle_duplicates

MODELS = {'faster_rcnn': 'Faster R-CNN', 'retinanet': 'RetinaNet', 'yolov8': 'YOLOv8s'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    subset = read(ROOT / 'annotations_subset.json')
    category_ids = {category['name']: category['id'] for category in subset['categories']}
    images = {image['file_name']: image for image in subset['images']}
    truth_by_id = {image['id']: [] for image in images.values()}
    for ann in subset['annotations']:
        assert not ann['iscrowd']
        x, y, width, height = ann['bbox']
        truth_by_id[ann['image_id']].append({'class_id': ann['category_id'],
                                           'box_xyxy': [x, y, x + width, y + height],
                                           'annotation_id': ann['id']})
    output = {}
    for key, name in MODELS.items():
        predictions = read(ROOT / 'runs' / key / 'summary.json')['results']
        assert len(predictions) == len(images)
        totals = Counter()
        vehicle_nms_totals = Counter()
        records = []
        for record in predictions:
            filename = Path(record['source']).name
            image = images[filename]
            assert (record['image_width'], record['image_height']) == (image['width'], image['height'])
            assert record['threshold'] == .25 and not record['vehicle_nms']['enabled']
            objects = [dict(det, class_id=category_ids[det['class_name']]) for det in record['detections']]
            truth = truth_by_id[image['id']]
            result = match_detections(objects, truth, .5)
            totals.update({k: result[k] for k in ('tp', 'fp', 'fn')})
            deduplicated, _ = suppress_vehicle_duplicates(objects, iou_threshold=.60)
            deduplicated_result = match_detections(deduplicated, truth, .5)
            vehicle_nms_totals.update({k: deduplicated_result[k] for k in ('tp', 'fp', 'fn')})
            records.append({'image': filename, 'truth': truth, 'detections': objects, 'matching': result,
                            'vehicle_nms_060': {'detections': deduplicated, 'matching': deduplicated_result}})
        assert {row['image'] for row in records} == set(images)
        output[key] = {'name': name, **totals, **scores(totals['tp'], totals['fp'], totals['fn']),
                       'vehicle_nms_060': {**vehicle_nms_totals, **scores(vehicle_nms_totals['tp'], vehicle_nms_totals['fp'], vehicle_nms_totals['fn'])},
                       'images': records}
        print(name, dict(totals), f"F1={output[key]['f1']:.3f}", flush=True)
        print('  existing vehicle NMS at .60:', dict(vehicle_nms_totals),
              f"F1={output[key]['vehicle_nms_060']['f1']:.3f}", flush=True)
    payload = {'dataset': 'COCO val2017 fixed 6-image vehicle subset', 'selection': subset['selection'],
               'conditions': {'conf': .25, 'match_iou': .5, 'vehicle_nms': False, 'mode': 'whole image',
                              'vehicle_nms_060': 'Existing vehicle NMS applied to identical saved candidates',
                              'match': 'same category, one-to-one, descending confidence'},
               'truth_count': len(subset['annotations']), 'models': output}
    (ROOT / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

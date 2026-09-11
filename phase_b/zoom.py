"""RGB-only sliced detection geometry; no simulator queries at inference."""
import numpy as np


def tile_windows(width, height, fraction=0.6):
    if not 0.5 <= fraction < 1:
        raise ValueError("tile fraction must be in [0.5, 1)")
    w, h = max(1, round(width*fraction)), max(1, round(height*fraction))
    return list(dict.fromkeys((x, y, x+w, y+h)
                             for y in (0, height-h) for x in (0, width-w)))


def crop_annotations(boxes, classes, window):
    x, y, x2, y2 = window
    out, labels = [], []
    for box, label in zip(boxes, classes):
        a, b, c, d = box
        a, b, c, d = max(a, x), max(b, y), min(c, x2), min(d, y2)
        if c-a >= 2 and d-b >= 2:
            out.append([a-x, b-y, c-x, d-y])
            labels.append(label)
    return out, labels


def training_crop(rgb, boxes, classes, rng):
    h, w = rgb.shape[:2]
    fraction = float(rng.uniform(0.4, 0.7))
    cw, ch = max(2, round(w*fraction)), max(2, round(h*fraction))
    small = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1])/(w*h) < 0.02]
    if small and rng.rand() < 0.75:
        b = small[int(rng.randint(len(small)))]
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        x = int(np.clip(cx-cw*rng.uniform(0.25, 0.75), 0, w-cw))
        y = int(np.clip(cy-ch*rng.uniform(0.25, 0.75), 0, h-ch))
    else:
        x, y = int(rng.randint(w-cw+1)), int(rng.randint(h-ch+1))
    boxes, classes = crop_annotations(boxes, classes, (x, y, x+cw, y+ch))
    return rgb[y:y+ch, x:x+cw], boxes, classes


def box_iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
        0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter / max(union, 1e-9)


def merge_detections(detections, threshold=0.5, max_det=100):
    kept = []
    for d in sorted(detections, key=lambda d: -d['score']):
        if all(d['type'] != k['type'] or box_iou(d['bbox'], k['bbox']) < threshold
               for k in kept):
            kept.append(d)
            if len(kept) >= max_det:
                break
    return kept

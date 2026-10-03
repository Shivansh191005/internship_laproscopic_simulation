"""
Make the YOLO model better at INSTRUMENTS. Run on the training machine
(the one with the dataset and train_yolo.py), in three steps:

  1. split-labels   fix the training labels: one instance per tool
  2. mine           pull hard frames from your own surgery videos, with
                    pre-filled labels, to correct and add to the dataset
  3. train          fine-tune best.pt with settings suited to instruments

    python retrain_instruments.py split-labels --data path/to/data.yaml --dry-run
    python retrain_instruments.py split-labels --data path/to/data.yaml
    python retrain_instruments.py mine  --video data/case01.avi --out hard_frames
    python retrain_instruments.py train --data path/to/data.yaml

Why each step (see the analysis in the chat / README):

  * The labels were made from SEMANTIC colour masks (train_yolo.py:
    rgb_to_label -> label_to_yolo_seg): every instrument pixel had one colour
    and each connected patch became one "instance". Wherever two tools touch
    or cross, they were labelled as ONE object, so the model learned to output
    one merged mask for touching tools. split-labels re-cuts every instrument
    polygon into straight shafts with the same RANSAC splitter the robot uses
    at run time (perception.split_tools). Your old labels are backed up first.

  * The model misses bloody / silver tools in dissection scenes (no
    instrument found in about a third of sampled case01 frames). Those scenes
    are under-represented. mine saves the frames where the model is unsure,
    with its best-guess labels, so correcting them takes minutes per frame
    instead of drawing from scratch.

  * Training used fliplr=0.0 (tools never seen entering from the mirrored
    side), imgsz 512 (thin jaws are a few pixels wide there) and mask_ratio 4
    (masks predicted at 1/4 resolution -> blobby, merged outlines). train fixes
    those and fine-tunes from your current best.pt, so it takes hours, not the
    original ~5 h from scratch.
"""
import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _instrument_id(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for i, n in items:
        if str(n).lower().startswith("instrument"):
            return int(i)
    sys.exit(f"no 'instruments' class in {names}")


def _label_dirs(data_yaml):
    d = _load_yaml(data_yaml)
    root = d.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
    out = []
    for split in ("train", "val", "test"):
        v = d.get(split)
        if not v:
            continue
        for p in (v if isinstance(v, list) else [v]):
            img_dir = p if os.path.isabs(p) else os.path.join(root, p)
            lab_dir = img_dir.replace(f"{os.sep}images", f"{os.sep}labels") \
                             .replace("/images", "/labels")
            if os.path.isdir(lab_dir):
                out.append((split, img_dir, lab_dir))
    return d, out


def _image_size(img_dir, stem):
    from PIL import Image
    for ext in (".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            with Image.open(p) as im:
                return im.size        # (w, h)
    return None


# --------------------------------------------------------------- split-labels
def split_polygon(norm_xy, w, h, work_w=640):
    """One normalised instrument polygon -> list of normalised polygons."""
    from perception import split_tools
    s = work_w / float(w)
    ww, hh = work_w, max(8, int(round(h * s)))
    pts = (np.asarray(norm_xy, np.float32).reshape(-1, 2) *
           np.array([ww, hh], np.float32))
    m = np.zeros((hh, ww), np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    if m.sum() < 50:
        return [norm_xy]
    parts = split_tools(m)
    if len(parts) < 2:
        return [norm_xy]
    out = []
    for part in parts:
        cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < 30:
                continue
            ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
            ap = ap.reshape(-1, 2).astype(np.float32)
            if len(ap) < 3:
                continue
            ap /= np.array([ww, hh], np.float32)
            out.append(np.clip(ap, 0, 1).reshape(-1).tolist())
    return out or [norm_xy]


def cmd_split_labels(a):
    d, dirs = _label_dirs(a.data)
    iid = _instrument_id(d["names"])
    print(f"instrument class id = {iid}")
    tot_files = tot_polys = tot_split = 0
    for split, img_dir, lab_dir in dirs:
        files = sorted(glob.glob(os.path.join(lab_dir, "*.txt")))
        if not a.dry_run:
            bak = lab_dir.rstrip("/\\") + "_semantic_backup"
            if not os.path.isdir(bak):
                shutil.copytree(lab_dir, bak)
                print(f"  backup: {bak}")
        n_split = n_poly = 0
        t0 = time.time()
        for k, f in enumerate(files):
            stem = os.path.splitext(os.path.basename(f))[0]
            with open(f) as fh:
                rows = [r.split() for r in fh.read().splitlines() if r.strip()]
            if not any(int(r[0]) == iid for r in rows):
                continue
            size = _image_size(img_dir, stem)
            if size is None:
                continue
            new_rows, changed = [], False
            for r in rows:
                if int(r[0]) != iid or len(r) < 7:
                    new_rows.append(" ".join(r))
                    continue
                n_poly += 1
                parts = split_polygon([float(v) for v in r[1:]], *size)
                if len(parts) > 1:
                    changed = True
                    n_split += 1
                for p in parts:
                    new_rows.append(f"{iid} " + " ".join(f"{v:.6f}" for v in p))
            if changed and not a.dry_run:
                with open(f, "w") as fh:
                    fh.write("\n".join(new_rows) + "\n")
            if k % 500 == 0 and k:
                print(f"  {split}: {k}/{len(files)}  ({time.time()-t0:.0f}s)")
        print(f"{split}: {len(files)} files, {n_poly} instrument polygons, "
              f"{n_split} split into separate tools"
              + ("  (dry run - nothing written)" if a.dry_run else ""))
        tot_files += len(files)
        tot_polys += n_poly
        tot_split += n_split
    print(f"TOTAL: {tot_split}/{tot_polys} instrument polygons were merged tools")
    if not a.dry_run:
        print("Delete the labels *.cache files next to the label folders so "
              "Ultralytics re-reads them.")


# ---------------------------------------------------------------------- mine
def cmd_mine(a):
    """
    Save frames worth labelling: the model finds NO instrument, or only
    low-confidence ones, or a mask that had to be split. Pre-labels (YOLO seg
    format, all classes, merged tools already split) go next to each image,
    ready to import into CVAT / Roboflow / Label Studio for correction.
    """
    import config as cfg
    from perception import YoloPerception, estimate_field
    src = YoloPerception(video_path=a.video, loop=False)
    cap = src.cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(a.every * fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    names = src.names
    saved = 0
    idx = 0
    print(f"sampling every {a.every}s ({step} frames) of {total}, "
          f"keeping up to {a.max}")
    while saved < a.max:
        if idx % step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if src.field is None:
            src.field = estimate_field(frame)
        res = src.model.predict(frame, imgsz=cfg.IMGSZ, conf=src.min_conf,
                                device=cfg.DEVICE, verbose=False)[0]
        raw = src._extract(res, frame.shape)
        tools = raw["instruments"]
        confs = [t.conf for t in tools]
        reason = None
        if not tools:
            reason = "no-tool"
        elif min(confs) < a.low_conf:
            reason = "low-conf"
        elif res.boxes is not None and len(tools) > int(
                (res.boxes.cls.cpu().numpy() == src.idx_instrument).sum()):
            reason = "split"
        if reason is None:
            continue
        h, w = frame.shape[:2]
        stem = f"{os.path.splitext(os.path.basename(a.video))[0]}_{idx:07d}"
        cv2.imwrite(os.path.join(a.out, "images", stem + ".jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows = []
        lower = {n.lower(): i for i, n in names.items()}
        for name, det in raw.items():
            if name == "instruments" or det is None:
                continue
            c = det.contour.reshape(-1, 2) / np.array([w, h], np.float32)
            rows.append(f"{lower[name.lower()]} " +
                        " ".join(f"{v:.6f}" for v in c.reshape(-1)))
        for t in tools:
            c = t.contour.reshape(-1, 2) / np.array([w, h], np.float32)
            rows.append(f"{src.idx_instrument} " +
                        " ".join(f"{v:.6f}" for v in c.reshape(-1)))
        with open(os.path.join(a.out, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))
        saved += 1
        print(f"  {stem}  {reason}  tools={len(tools)}")
    with open(os.path.join(a.out, "classes.txt"), "w") as f:
        f.write("\n".join(names[i] for i in sorted(names)) + "\n")
    print(f"saved {saved} frames to {a.out}. Correct the labels (fix tool "
          f"outlines, add missed tools, ONE polygon PER TOOL), then copy "
          f"images/ and labels/ into the train split.")


# --------------------------------------------------------------------- train
def cmd_train(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, workers=a.workers, patience=15, seed=42, project=a.project,
        name="instruments_ft", exist_ok=True,
        # fine-tune: lower LR, short warmup
        lr0=0.002, lrf=0.1, warmup_epochs=1.0, cos_lr=True,
        # sharper masks: predict them at 1/2 instead of 1/4 resolution
        mask_ratio=2, overlap_mask=True,
        # augmentation suited to instruments
        fliplr=0.5,        # tools enter from both sides (was 0.0)
        degrees=15.0, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,   # glare / dark bloody scenes
        mosaic=1.0, close_mosaic=10, copy_paste=0.4, copy_paste_mode="flip",
        mixup=0.0, erasing=0.3, cache="disk",
    )
    print("done. Copy runs/.../instruments_ft/weights/best.pt to the robot "
          "project's weights/best.pt, set IMGSZ in config.py to the training "
          "size.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split-labels")
    s.add_argument("--data", required=True, help="dataset data.yaml")
    s.add_argument("--dry-run", action="store_true",
                   help="only count what would change")
    s.set_defaults(fn=cmd_split_labels)

    m = sub.add_parser("mine")
    m.add_argument("--video", required=True)
    m.add_argument("--out", default="hard_frames")
    m.add_argument("--every", type=float, default=2.0, help="seconds")
    m.add_argument("--max", type=int, default=300)
    m.add_argument("--low-conf", type=float, default=0.4)
    m.set_defaults(fn=cmd_mine)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--weights", default=os.path.join(HERE, "weights", "best.pt"))
    t.add_argument("--imgsz", type=int, default=768)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=8,
                   help="8 fits 6 GB at 768; 16 needs ~10 GB")
    t.add_argument("--device", default="0", help='"0" = first GPU, "cpu"')
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--project", default=os.path.join(HERE, "runs", "train"))
    t.set_defaults(fn=cmd_train)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

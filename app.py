import csv
import json
import math
import random
import re
import shutil
from io import BytesIO, StringIO
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageEnhance

app = Flask(__name__, static_folder="static")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── label helpers ─────────────────────────────────────────────────────────────

def read_det_label(path: Path) -> list:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append({"class_id": int(parts[0]),
                          "cx": float(parts[1]), "cy": float(parts[2]),
                          "w":  float(parts[3]), "h":  float(parts[4])})
    return boxes


def write_det_label(path: Path, boxes: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n"
        for b in boxes
    ))


def clf_label_path(project: str, stem: str) -> Path:
    return project_path(project) / "labels" / (stem + ".json")


def read_clf_label(path: Path) -> list:
    """Returns list of selected class indices."""
    if not path.exists():
        return []
    return json.loads(path.read_text())


def write_clf_label(path: Path, indices: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(int(i) for i in indices))))


# ── augmentation ──────────────────────────────────────────────────────────────

def box_corners(b, w, h):
    bx, by = b["cx"]*w, b["cy"]*h
    bw2, bh2 = b["w"]*w/2, b["h"]*h/2
    return [(bx-bw2,by-bh2),(bx+bw2,by-bh2),(bx+bw2,by+bh2),(bx-bw2,by+bh2)]


def corners_to_box(class_id, corners, iw, ih):
    xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
    x1,x2,y1,y2=min(xs),max(xs),min(ys),max(ys)
    cx,cy=(x1+x2)/(2*iw),(y1+y2)/(2*ih)
    bw,bh=(x2-x1)/iw,(y2-y1)/ih
    if 0<cx<1 and 0<cy<1 and bw>0 and bh>0:
        return {"class_id":class_id,
                "cx":max(0.,min(1.,cx)),"cy":max(0.,min(1.,cy)),
                "w":max(0.,min(1.,bw)),"h":max(0.,min(1.,bh))}
    return None


def aug_brightness(img, boxes, v):
    return ImageEnhance.Brightness(img).enhance(max(0.05, 1.0+v)), boxes


def aug_hue(img, boxes, v):
    return ImageEnhance.Color(img).enhance(max(0.0, 1.0+v)), boxes


def aug_rotate(img, boxes, angle_deg):
    if not angle_deg:
        return img, boxes
    w, h = img.size
    img_r = img.rotate(angle_deg, expand=True, resample=Image.BILINEAR,
                       fillcolor=(114,114,114))
    nw, nh = img_r.size
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    new_boxes = []
    for b in boxes:
        pts = box_corners(b, w, h)
        rot = [(cos_a*(px-w/2)-sin_a*(py-h/2)+nw/2,
                sin_a*(px-w/2)+cos_a*(py-h/2)+nh/2) for px,py in pts]
        nb = corners_to_box(b["class_id"], rot, nw, nh)
        if nb: new_boxes.append(nb)
    return img_r, new_boxes


def aug_tilt(img, boxes, tilt_deg):
    if not tilt_deg:
        return img, boxes
    w, h = img.size
    shear = math.tan(math.radians(tilt_deg))
    offset_fwd = (w + abs(shear)*h)/2 - w/2   # keep image centred
    new_w = int(w + abs(shear)*h)
    # PIL inverse: ix = ox - shear*(oy - h/2) - offset_fwd
    coeffs = (1, -shear, shear*h/2 - offset_fwd, 0, 1, 0)
    img_t = img.transform((new_w,h), Image.AFFINE, coeffs,
                           resample=Image.BILINEAR, fillcolor=(114,114,114))
    new_boxes = []
    for b in boxes:
        pts = box_corners(b, w, h)
        fwd = [(px + shear*(py-h/2) + offset_fwd, py) for px,py in pts]
        nb = corners_to_box(b["class_id"], fwd, new_w, h)
        if nb: new_boxes.append(nb)
    return img_t, new_boxes


def apply_random_aug(img: Image.Image, boxes: list, params: dict, rng: random.Random):
    img = img.copy(); boxes = list(boxes)
    if params.get("brightness"):
        img, boxes = aug_brightness(img, boxes, rng.uniform(-params["brightness"], params["brightness"]))
    if params.get("hue"):
        img, boxes = aug_hue(img, boxes, rng.uniform(-params["hue"], params["hue"]))
    if params.get("rotate"):
        img, boxes = aug_rotate(img, boxes, rng.uniform(-params["rotate"], params["rotate"]))
    if params.get("tilt"):
        img, boxes = aug_tilt(img, boxes, rng.uniform(-params["tilt"], params["tilt"]))
    return img, boxes


# ── project management ────────────────────────────────────────────────────────

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)


def project_path(name: str) -> Path:
    return PROJECTS_DIR / name


def load_meta(name: str) -> dict:
    p = project_path(name) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_meta(name: str, meta: dict):
    p = project_path(name) / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))


# ── routes: core ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/projects", methods=["GET"])
def list_projects():
    return jsonify([p.name for p in sorted(PROJECTS_DIR.iterdir()) if p.is_dir()])


@app.route("/api/projects", methods=["POST"])
def create_project():
    data = request.json
    name = data.get("name","").strip()
    if not name:
        return jsonify({"error":"name required"}), 400
    p = project_path(name)
    if p.exists():
        return jsonify({"error":"exists"}), 409
    (p/"images").mkdir(parents=True)
    (p/"labels").mkdir(parents=True)
    save_meta(name, {
        "classes": data.get("classes", ["object"]),
        "task":    data.get("task", "detection"),
    })
    return jsonify({"ok": True})


@app.route("/api/projects/<name>", methods=["DELETE"])
def delete_project(name):
    p = project_path(name)
    if p.exists(): shutil.rmtree(p)
    return jsonify({"ok": True})


@app.route("/api/projects/<name>/meta", methods=["GET"])
def get_meta(name):
    return jsonify(load_meta(name))


@app.route("/api/projects/<name>/meta", methods=["PUT"])
def put_meta(name):
    meta = load_meta(name)
    meta.update(request.json)
    save_meta(name, meta)
    return jsonify({"ok": True})


# ── routes: images ────────────────────────────────────────────────────────────

@app.route("/api/projects/<name>/images", methods=["GET"])
def list_images(name):
    d = project_path(name) / "images"
    if not d.exists(): return jsonify([])
    return jsonify([p.name for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS])


@app.route("/api/projects/<name>/images/upload", methods=["POST"])
def upload_images(name):
    d = project_path(name) / "images"
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in request.files.getlist("files"):
        f.save(str(d / f.filename))
        saved.append(f.filename)
    return jsonify({"saved": saved})


@app.route("/api/projects/<name>/images/<filename>")
def get_image(name, filename):
    return send_file(str(project_path(name) / "images" / filename))


@app.route("/api/projects/<name>/images/<filename>", methods=["DELETE"])
def delete_image(name, filename):
    stem = Path(filename).stem
    for p in [project_path(name)/"images"/filename,
              project_path(name)/"labels"/(stem+".txt"),
              project_path(name)/"labels"/(stem+".json")]:
        if p.exists(): p.unlink()
    return jsonify({"ok": True})


# ── routes: detection labels ──────────────────────────────────────────────────

@app.route("/api/projects/<name>/labels/<filename>", methods=["GET"])
def get_det_label(name, filename):
    return jsonify(read_det_label(project_path(name)/"labels"/(Path(filename).stem+".txt")))


@app.route("/api/projects/<name>/labels/<filename>", methods=["PUT"])
def put_det_label(name, filename):
    write_det_label(project_path(name)/"labels"/(Path(filename).stem+".txt"), request.json)
    return jsonify({"ok": True})


# ── routes: classification labels ─────────────────────────────────────────────

@app.route("/api/projects/<name>/clf_labels/<filename>", methods=["GET"])
def get_clf_label_route(name, filename):
    return jsonify(read_clf_label(clf_label_path(name, Path(filename).stem)))


@app.route("/api/projects/<name>/clf_labels/<filename>", methods=["PUT"])
def put_clf_label_route(name, filename):
    write_clf_label(clf_label_path(name, Path(filename).stem), request.json)
    return jsonify({"ok": True})


# ── routes: import ────────────────────────────────────────────────────────────

@app.route("/api/projects/<name>/import", methods=["POST"])
def import_dataset(name):
    src = Path(request.json.get("path",""))
    if not src.exists():
        return jsonify({"error":"path not found"}), 400

    meta = load_meta(name)
    task = meta.get("task","detection")
    dst_imgs = project_path(name)/"images"
    dst_lbls = project_path(name)/"labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    copied = 0
    if task == "classification":
        # Import from example2 format: train/valid/test flat dirs + _classes.csv
        class_map = {}  # filename → list of class indices
        classes   = []

        for split in ("train","valid","test"):
            split_dir = src / split
            if not split_dir.exists(): continue
            csv_path = split_dir / "_classes.csv"
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    if not classes:
                        classes = [c for c in reader.fieldnames if c != "filename"]
                    for row in reader:
                        fname = row["filename"]
                        selected = [i for i,c in enumerate(classes) if row.get(c,"0")=="1"]
                        class_map[fname] = selected
            for f in split_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(f, dst_imgs/f.name)
                    copied += 1
                    if f.name in class_map:
                        write_clf_label(dst_lbls/(f.stem+".json"), class_map[f.name])

        if classes:
            meta["classes"] = classes
            save_meta(name, meta)
    else:
        # Detection: split structure or flat
        if any((src/s/"images").exists() for s in ("train","valid","test")):
            for split in ("train","valid","test"):
                for f in (src/split/"images").glob("*") if (src/split/"images").exists() else []:
                    if f.suffix.lower() in IMAGE_EXTS:
                        shutil.copy2(f, dst_imgs/f.name); copied += 1
                for f in (src/split/"labels").glob("*.txt") if (src/split/"labels").exists() else []:
                    shutil.copy2(f, dst_lbls/f.name)
        else:
            src_imgs = src/"images" if (src/"images").exists() else src
            for f in src_imgs.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(f, dst_imgs/f.name); copied += 1
            if (src/"labels").exists():
                for f in (src/"labels").glob("*.txt"):
                    shutil.copy2(f, dst_lbls/f.name)

        yaml_path = src/"data.yaml"
        if yaml_path.exists():
            m = re.search(r"names:\s*\[([^\]]+)\]", yaml_path.read_text())
            if m:
                meta["classes"] = [c.strip().strip("'\"") for c in m.group(1).split(",")]
                save_meta(name, meta)

    return jsonify({"copied": copied})


# ── routes: export ────────────────────────────────────────────────────────────

def _save_img_to_zip(zf, zip_path, img, orig_ext, fmt):
    buf = BytesIO()
    img.save(buf, format=fmt)
    zf.writestr(zip_path, buf.getvalue())


@app.route("/api/projects/<name>/export", methods=["POST"])
def export_dataset(name):
    """
    Body: {
      "train": 0.7, "valid": 0.2, "test": 0.1, "seed": 42,
      "augmentation": { "enabled": bool, "copies": N,
                        "brightness": 0.2, "hue": 0.1, "rotate": 15, "tilt": 10 }
    }
    """
    data    = request.json or {}
    train_r = float(data.get("train", 0.7))
    valid_r = float(data.get("valid", 0.2))
    test_r  = float(data.get("test",  0.1))
    seed    = int(data.get("seed", 42))
    aug_cfg = data.get("augmentation", {})
    aug_on  = bool(aug_cfg.get("enabled", False))
    aug_copies = max(1, int(aug_cfg.get("copies", 1)))
    aug_params = {k: float(aug_cfg.get(k, 0)) for k in ("brightness","hue","rotate","tilt")}

    total = train_r + valid_r + test_r or 1
    train_r /= total; valid_r /= total; test_r /= total

    meta    = load_meta(name)
    classes = meta.get("classes", ["object"])
    task    = meta.get("task", "detection")
    img_dir = project_path(name) / "images"
    lbl_dir = project_path(name) / "labels"

    all_imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng = random.Random(seed)
    rng.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = round(n * train_r)
    n_valid = round(n * valid_r)
    splits = {
        "train": all_imgs[:n_train],
        "valid": all_imgs[n_train:n_train+n_valid],
        "test":  all_imgs[n_train+n_valid:],
    }

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if task == "detection":
            _export_detection(zf, name, splits, classes, lbl_dir, aug_on, aug_copies, aug_params, rng)
        else:
            _export_classification(zf, name, splits, classes, lbl_dir, aug_on, aug_copies, aug_params, rng)

    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{name}.zip",
                     mimetype="application/zip")


def _export_detection(zf, name, splits, classes, lbl_dir, aug_on, aug_copies, aug_params, rng):
    import zipfile as _zf
    yaml_content = "\n".join([
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {len(classes)}",
        f"names: {json.dumps(classes)}",
    ])
    zf.writestr(f"{name}/data.yaml", yaml_content)

    for split, files in splits.items():
        for img_path in files:
            stem = img_path.stem
            ext  = img_path.suffix
            lbl_path = lbl_dir / (stem + ".txt")
            boxes = read_det_label(lbl_path)
            fmt = "JPEG" if ext.lower() in (".jpg",".jpeg") else "PNG"

            zf.write(img_path, f"{name}/{split}/images/{img_path.name}")
            if lbl_path.exists():
                zf.write(lbl_path, f"{name}/{split}/labels/{stem}.txt")

            if aug_on and split == "train":
                img = Image.open(img_path).convert("RGB")
                for i in range(aug_copies):
                    aug_img, aug_boxes = apply_random_aug(img, boxes, aug_params, rng)
                    aug_stem = f"{stem}_aug{i+1}"
                    _save_img_to_zip(zf, f"{name}/{split}/images/{aug_stem}{ext}", aug_img, ext, fmt)
                    lbl_lines = "".join(
                        f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n"
                        for b in aug_boxes
                    )
                    zf.writestr(f"{name}/{split}/labels/{aug_stem}.txt", lbl_lines)


def _export_classification(zf, name, splits, classes, lbl_dir, aug_on, aug_copies, aug_params, rng):
    """
    Export format matching example2:
      <name>/
        train/
          img.jpg
          _classes.csv   (filename, class1, class2, ...)
        valid/
          ...
        test/
          ...
    """
    for split, files in splits.items():
        if not files:
            continue

        rows = []          # list of dicts for CSV
        header = ["filename"] + classes

        for img_path in files:
            stem = img_path.stem
            ext  = img_path.suffix
            lbl_path = lbl_dir / (stem + ".json")
            selected = set(read_clf_label(lbl_path))
            fmt = "JPEG" if ext.lower() in (".jpg",".jpeg") else "PNG"

            # Original
            zf.write(img_path, f"{name}/{split}/{img_path.name}")
            row = {"filename": img_path.name}
            for ci, _ in enumerate(classes):
                row[classes[ci]] = "1" if ci in selected else "0"
            rows.append(row)

            # Augmented copies (training split only)
            if aug_on and split == "train":
                img = Image.open(img_path).convert("RGB")
                for i in range(aug_copies):
                    aug_img, _ = apply_random_aug(img, [], aug_params, rng)
                    aug_fname = f"{stem}_aug{i+1}{ext}"
                    _save_img_to_zip(zf, f"{name}/{split}/{aug_fname}", aug_img, ext, fmt)
                    aug_row = {"filename": aug_fname}
                    aug_row.update({classes[ci]: ("1" if ci in selected else "0") for ci in range(len(classes))})
                    rows.append(aug_row)

        # Write _classes.csv for this split
        csv_buf = StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
        zf.writestr(f"{name}/{split}/_classes.csv", csv_buf.getvalue())


# deferred import
import zipfile

if __name__ == "__main__":
    app.run(debug=True, port=5000)

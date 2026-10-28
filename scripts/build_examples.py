"""Pre-render the example strip shown on the landing page.

Every card in the strip is real model output: each test image is run through
the fine-tuned detector, cropped to 4:3 around its detections, annotated in the
design system's palette, and priced with the same table the scanner uses.

Run from the repo root:  python scripts/build_examples.py [sources.json]
Outputs:                 web/examples/ex_XX.jpg and web/examples/examples.json

sources.json, if given, is a list of {"path", "credit", "license", "url", "conf"} for
photos from outside the repo. Licences must allow modification, since the
boxes are drawn onto the photo; the credit is shown on the card.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app  # noqa: E402  (reuses the model, price table and labels)

# Chosen for variety of categories and a confident, multi-item read.
SOURCES = ["test_3.jpg", "test_8.jpg", "test_10.jpg", "test_17.jpg",
           "test_7.jpg", "test_9.jpg", "test_19.png", "test_11.jpeg"]
CONF = 0.45
OUT_W, OUT_H = 960, 720            # 4:3, ~2x the card's rendered width
OUT_DIR = ROOT / "web" / "examples"

PERSIMMON = (217, 95, 59)
FOREST_DEEP = (20, 58, 45)
CREAM = (255, 244, 239)


def font(size):
    for name in ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def crop_window(w, h, boxes):
    """Largest-useful 4:3 window that keeps the detections in frame."""
    ratio = OUT_W / OUT_H
    if boxes:
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        pad = 0.06 * max(w, h)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    cw, ch = x2 - x1, y2 - y1
    if cw / ch < ratio:
        cw = ch * ratio
    else:
        ch = cw / ratio
    if cw > w:
        cw, ch = w, w / ratio
    if ch > h:
        ch, cw = h, h * ratio
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0, cx - cw / 2), w - cw)
    top = min(max(0, cy - ch / 2), h - ch)
    return left, top, cw, ch


def annotate(img, dets):
    """Persimmon boxes with a faint fill and a dark halo, plus pill labels."""
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(22)
    for x1, y1, x2, y2, label in dets:
        r = 10
        d.rounded_rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], r + 1,
                            outline=FOREST_DEEP + (90,), width=7)
        d.rounded_rectangle([x1, y1, x2, y2], r,
                            fill=PERSIMMON + (26,), outline=PERSIMMON + (255,), width=4)
    for x1, y1, x2, y2, label in dets:
        tw = d.textlength(label, font=f)
        ph, px = 34, 14
        ly = y1 - ph - 6 if y1 - ph - 6 > 4 else y1 + 6
        lx = min(max(4, x1), OUT_W - tw - px * 2 - 4)
        d.rounded_rectangle([lx, ly, lx + tw + px * 2, ly + ph], ph // 2,
                            fill=PERSIMMON + (255,))
        d.text((lx + px, ly + ph / 2), label, font=f, fill=CREAM + (255,), anchor="lm")
    return Image.alpha_composite(base, layer).convert("RGB")


def load_sources():
    if len(sys.argv) > 1:
        return json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return [{"path": str(app.TEST_IMAGES_DIR / n)} for n in SOURCES]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("ex_*.jpg"):      # never leave a stale card behind
        old.unlink()
    out = []
    for i, srcinfo in enumerate(load_sources(), start=1):
        src = Path(srcinfo["path"])
        name = src.name
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        res = app.model.predict(img, conf=srcinfo.get("conf", CONF), device="cpu", verbose=False)[0]
        raw = [([float(v) for v in b.xyxy[0]], app.model.names[int(b.cls)])
               for b in res.boxes]

        left, top, cw, ch = crop_window(img.width, img.height, [r[0] for r in raw])
        crop = img.crop((left, top, left + cw, top + ch)).resize((OUT_W, OUT_H), Image.LANCZOS)
        sx, sy = OUT_W / cw, OUT_H / ch

        dets = []
        for (x1, y1, x2, y2), cls in raw:
            nx1, ny1 = (x1 - left) * sx, (y1 - top) * sy
            nx2, ny2 = (x2 - left) * sx, (y2 - top) * sy
            if nx2 < 0 or ny2 < 0 or nx1 > OUT_W or ny1 > OUT_H:
                continue                      # fell outside the crop
            info = app.CATEGORY_PRICES.get(cls, {"en": cls})
            dets.append((max(0, nx1), max(0, ny1), min(OUT_W, nx2), min(OUT_H, ny2), info["en"]))

        fname = f"ex_{i:02d}.jpg"
        annotate(crop, dets).save(OUT_DIR / fname, quality=80, optimize=True, progressive=True)

        counts = Counter(cls for _, cls in raw)
        items = []
        for cls, n in counts.most_common():
            info = app.CATEGORY_PRICES.get(cls, {"en": cls, "ar": cls, "price": 0.0})
            items.append({"key": cls, "en": info["en"], "ar": info["ar"],
                          "count": n, "price": info["price"]})
        subtotal = round(sum(it["count"] * it["price"] for it in items), 2)
        vat = round(subtotal * 0.15, 2)
        out.append({
            "file": fname, "source": name, "invoice": f"VC-{3000 + i * 17}",
            "n": sum(counts.values()), "categories": len(counts), "items": items,
            "subtotal": subtotal, "vat": vat, "total": round(subtotal + vat, 2),
            "credit": srcinfo.get("credit"), "license": srcinfo.get("license"),
            "url": srcinfo.get("url"),
        })
        print(f"{fname}  {name:13s} n={out[-1]['n']:2d}  total={out[-1]['total']:.2f}")

    (OUT_DIR / "examples.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"wrote {len(out)} examples to {OUT_DIR}")


if __name__ == "__main__":
    main()

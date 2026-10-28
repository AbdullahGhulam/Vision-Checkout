"""Vision Checkout — Gradio app for Hugging Face Spaces.

Single-file Gradio app: upload a tray photo, run YOLO detection, generate a receipt.
"""
import json
import random
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# Blush Aisle palette, as RGB, for drawing onto the annotated image.
C_PERSIMMON = (217, 95, 59)
C_PERSIMMON_DEEP = (183, 70, 42)
C_MUSTARD = (224, 169, 59)
C_CREAM = (255, 244, 239)
C_FOREST_DEEP = (20, 58, 45)

# Detection runs at this floor; the slider then splits results into confident
# detections and low-confidence ones, so the threshold is live rather than
# something you have to re-scan to change.
DETECT_FLOOR = 0.15

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Fine_Tuned_SAUDI_model" / "weights" / "best.pt"
TEST_IMAGES_DIR = BASE_DIR / "models" / "Fine_Tuned_SAUDI_model" / "test_images"

CATEGORY_PRICES = {
    "puffed_food":      {"ar": "وجبات خفيفة",   "en": "Puffed snacks",   "price": 6.50,  "emoji": "🍿"},
    "dried_fruit":      {"ar": "فواكه مجففة",   "en": "Dried fruit",     "price": 12.00, "emoji": "🍇"},
    "dried_food":       {"ar": "أغذية مجففة",   "en": "Dried food",      "price": 8.00,  "emoji": "🥜"},
    "instant_drink":    {"ar": "مشروب سريع",    "en": "Instant drink",   "price": 5.00,  "emoji": "☕"},
    "instant_noodles":  {"ar": "نودلز سريع",    "en": "Instant noodles", "price": 4.50,  "emoji": "🍜"},
    "dessert":          {"ar": "حلويات",         "en": "Dessert",         "price": 4.00,  "emoji": "🍰"},
    "drink":            {"ar": "مشروبات",        "en": "Drinks",          "price": 3.50,  "emoji": "🥤"},
    "alcohol":          {"ar": "مشروب كحولي",   "en": "Alcohol",         "price": 0.00,  "emoji": "🚫"},
    "milk":             {"ar": "حليب",           "en": "Milk",            "price": 9.00,  "emoji": "🥛"},
    "canned_food":      {"ar": "معلبات",         "en": "Canned food",     "price": 5.00,  "emoji": "🥫"},
    "chocolate":        {"ar": "شوكولاتة",       "en": "Chocolate",       "price": 7.00,  "emoji": "🍫"},
    "gum":              {"ar": "علكة",           "en": "Gum",             "price": 2.50,  "emoji": "🫧"},
    "candy":            {"ar": "حلوى",           "en": "Candy",           "price": 3.00,  "emoji": "🍬"},
    "seasoner":         {"ar": "بهارات وتوابل", "en": "Seasoning",       "price": 6.00,  "emoji": "🧂"},
    "personal_hygiene": {"ar": "عناية شخصية",   "en": "Personal care",   "price": 15.00, "emoji": "🧴"},
    "tissue":           {"ar": "مناديل",         "en": "Tissues",         "price": 8.00,  "emoji": "🧻"},
    "stationery":       {"ar": "قرطاسية",        "en": "Stationery",      "price": 10.00, "emoji": "✏️"},
}

T = {
    "en": {
        "title": "Scan a cart",
        "tagline": "Upload a photo or pick a sample. Every product is found, counted and priced.",
        "image_label": "Upload image",
        "confidence": "Confidence threshold",
        "store": "Store name",
        "vat": "Add VAT (15%)",
        "scan": "Scan & generate receipt",
        "detections": "Detections",
        "samples": "Try a sample image",
        "shuffle": "Shuffle samples",
        "prices_label": "Price list",
        "no_items": "No products detected. Try a lower confidence threshold.",
        "detected_count": "Detected {n} item(s) across {k} category(ies).",
        "subtotal": "Subtotal", "vat_label": "VAT (15%)", "total": "TOTAL",
        "currency": "SAR", "invoice": "Invoice",
        "thank_you": "Thank you for shopping.",
        "tagline_receipt": "AI-Powered Smart Checkout",
        "print": "Print receipt",
    },
    "ar": {
        "title": "افحص السلة",
        "tagline": "ارفع صورة أو اختر عيّنة، وكل منتج ينكشف وينحسب ويتسعّر.",
        "image_label": "ارفع صورة",
        "confidence": "حد الثقة",
        "store": "اسم المتجر",
        "vat": "إضافة ضريبة القيمة المضافة (15٪)",
        "scan": "ابدأ الفحص وأنشئ الفاتورة",
        "detections": "الاكتشافات",
        "samples": "جرّب عيّنة عشوائية",
        "shuffle": "خلط العيّنات",
        "prices_label": "قائمة الأسعار",
        "no_items": "لم يتم اكتشاف أي منتج. جرّب تخفيض حد الثقة.",
        "detected_count": "تم اكتشاف {n} منتج عبر {k} فئة.",
        "subtotal": "المجموع الفرعي", "vat_label": "ضريبة القيمة المضافة (15٪)", "total": "الإجمالي",
        "currency": "ر.س", "invoice": "فاتورة",
        "thank_you": "شكراً لتسوقكم.",
        "tagline_receipt": "نظام كاشير ذكي بالذكاء الاصطناعي",
        "print": "طباعة الفاتورة",
    },
}

# ---------------------------------------------------------------------------
# Model & helpers
# ---------------------------------------------------------------------------
model = YOLO(str(MODEL_PATH))


def list_test_images():
    if not TEST_IMAGES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(str(p) for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


ALL_SAMPLES = list_test_images()


def pick_random_samples(n=3):
    return random.sample(ALL_SAMPLES, min(n, len(ALL_SAMPLES))) if ALL_SAMPLES else []


def build_cart(detections):
    cart = {}
    for cls in detections:
        info = CATEGORY_PRICES.get(cls, {"ar": cls, "en": cls, "price": 0.0, "emoji": "📦"})
        if cls not in cart:
            cart[cls] = {"info": info, "count": 0}
        cart[cls]["count"] += 1
    return cart


# Printing the live page never worked: the receipt sits deep inside a very long
# document, so `visibility:hidden` on everything plus `position:absolute` pushed
# it outside the printable flow and produced blank pages. Instead, clone the
# receipt into an isolated iframe and print that.
PRINT_JS = (
    "var inv=document.getElementById('vc-invoice');"
    "if(inv){"
    "var w=document.getElementById('vc-receipt-wrap');"
    "var dir=(w&&w.getAttribute('dir'))||'ltr';"
    "var s=document.getElementById('vc-print-style');"
    "var f=document.createElement('iframe');"
    # A real page-sized layout box, parked off-screen. A 1px iframe makes the
    # invoice lay out at 1px wide and print squashed.
    "f.setAttribute('style','position:fixed;left:-10000px;top:0;width:794px;"
    "height:1123px;border:0;');"
    "document.body.appendChild(f);"
    "var d=f.contentWindow.document;d.open();"
    "d.write('<!doctype html><html><head><meta charset=utf-8>'"
    "+(s?s.outerHTML:'')"
    "+'<style>@page{margin:12mm}html,body{margin:0;background:#fff;"
    "-webkit-print-color-adjust:exact;print-color-adjust:exact}"
    "#vc-invoice{display:block}</style></head><body dir='+dir+'>'"
    "+inv.outerHTML+'</body></html>');"
    "d.close();"
    "var L=d.createElement('link');L.rel='stylesheet';"
    "L.href='https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:"
    "opsz,wght@12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap';"
    "d.head.appendChild(L);"
    "setTimeout(function(){"
    "try{f.contentWindow.focus();f.contentWindow.print();}catch(e){window.print();}"
    "setTimeout(function(){if(f.parentNode){f.parentNode.removeChild(f);}},1500);"
    "},700);"
    "}"
)


def render_receipt(cart, store_name, show_vat, lang):
    txt = T[lang]
    direction = "rtl" if lang == "ar" else "ltr"
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])

    subtotal = sum(it["info"]["price"] * it["count"] for it in cart.values())
    vat = subtotal * 0.15 if show_vat else 0
    total = subtotal + vat
    now = datetime.now()
    invoice_no = f"VC-{random.randint(10000, 99999)}"

    n_lines = len(cart)
    rows = "".join(
        f'<div class="row"><span class="rl">{label(it["info"])}'
        f'{" &times;" + str(it["count"]) if it["count"] > 1 else ""}</span>'
        f'<span class="rv">{it["info"]["price"] * it["count"]:.2f}</span></div>'
        for it in cart.values()
    )
    vat_row = (
        f'<div class="row muted"><span class="rl">{txt["vat_label"]}</span>'
        f'<span class="rv">{vat:.2f}</span></div>'
        if show_vat else ""
    )

    inv_rows = "".join(
        f'<tr><td>{label(it["info"])}</td>'
        f'<td class="n">{it["count"]}</td>'
        f'<td class="n">{it["info"]["price"]:.2f}</td>'
        f'<td class="n">{it["info"]["price"] * it["count"]:.2f}</td></tr>'
        for it in cart.values()
    )
    inv_vat = (
        f'<div class="trow"><span>{txt["vat_label"]}</span>'
        f'<span>{vat:.2f}</span></div>' if show_vat else ""
    )

    return f"""
<div id="vc-receipt-wrap" dir="{direction}">
<style id="vc-rstyle">
#vc-receipt-wrap {{ margin:0; font-family:'Instrument Sans',system-ui,sans-serif; }}
#vc-receipt {{ background:#FFF4EF; border:1px solid #EBCFC6; border-radius:28px;
    padding:28px 30px; width:100%; color:#143A2D; }}
#vc-receipt .rhead {{ display:flex; align-items:baseline; justify-content:space-between;
    margin-bottom:18px; }}
#vc-receipt .rlabel {{ font-weight:600; font-size:14px; }}
#vc-receipt .rmeta {{ font-size:13px; color:#4C7A66; font-variant-numeric:tabular-nums; }}
#vc-receipt .rstore {{ font-family:'Bricolage Grotesque','Instrument Sans',sans-serif;
    font-size:26px; font-weight:800; letter-spacing:-.03em; line-height:1.15; }}
#vc-receipt .rsub {{ font-size:13px; color:#4C7A66; margin:2px 0 18px;
    font-variant-numeric:tabular-nums; }}
#vc-receipt .row {{ display:flex; justify-content:space-between; align-items:baseline;
    gap:16px; padding:9px 0; font-size:15px; border-bottom:1px solid #EBCFC6;
    font-variant-numeric:tabular-nums; }}
#vc-receipt .row.muted {{ color:#4C7A66; font-size:14px; }}
#vc-receipt .rv {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
#vc-receipt .total {{ display:flex; justify-content:space-between; align-items:baseline;
    padding-top:18px; }}
#vc-receipt .total .rl {{ font-weight:600; font-size:14px; }}
#vc-receipt .total .rv {{ font-size:26px; font-weight:600; color:#B7462A;
    font-variant-numeric:tabular-nums; letter-spacing:-.01em; }}
#vc-print-btn {{ margin-top:16px; padding:13px 26px; background:transparent;
    color:#143A2D; border:1px solid #EBCFC6; border-radius:999px; font-weight:500;
    font-size:15px; cursor:pointer; font-family:inherit; }}
#vc-print-btn:hover {{ background:#FBE3D8; }}
</style>

<style id="vc-print-style">
/* The printed artefact is a proper invoice, not the on-screen card. Every rule
   is scoped to #vc-invoice, which is hidden on screen and only ever rendered
   inside the print iframe. print-color-adjust keeps the fills — without it
   browsers drop every background and the invoice prints as bare text. */
#vc-invoice {{ display:none; }}
#vc-invoice, #vc-invoice * {{ box-sizing:border-box;
    -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
#vc-invoice {{ font-family:'Instrument Sans',system-ui,sans-serif; color:#143A2D;
    background:#FFF4EF; border:1px solid #E4CFC5; padding:34px 38px 26px;
    max-width:640px; margin:0 auto; }}
#vc-invoice .ihead {{ display:flex; justify-content:space-between;
    align-items:flex-start; gap:20px; border-bottom:2px solid #143A2D;
    padding-bottom:16px; }}
#vc-invoice .istore {{ font-family:'Bricolage Grotesque','Instrument Sans',sans-serif;
    font-size:30px; font-weight:800; letter-spacing:-.03em; line-height:1.05; }}
#vc-invoice .itag {{ font-size:12px; color:#4C7A66; margin-top:4px; }}
#vc-invoice .ititle {{ font-size:13px; font-weight:600; letter-spacing:.16em;
    text-transform:uppercase; color:#B7462A; white-space:nowrap; padding-top:6px; }}
#vc-invoice .imeta {{ display:flex; gap:34px; margin:16px 0 22px; font-size:12.5px; }}
#vc-invoice .imeta span {{ color:#4C7A66; display:block; margin-bottom:2px; }}
#vc-invoice .imeta b {{ font-weight:600; font-variant-numeric:tabular-nums; }}
#vc-invoice table {{ width:100%; border-collapse:collapse; font-size:14px; }}
#vc-invoice thead th {{ text-align:left; font-size:11px; font-weight:600;
    letter-spacing:.12em; text-transform:uppercase; color:#4C7A66;
    border-bottom:1px solid #143A2D; padding:0 0 8px; }}
#vc-invoice th.n, #vc-invoice td.n {{ text-align:right; white-space:nowrap;
    font-variant-numeric:tabular-nums; }}
#vc-invoice th.n, #vc-invoice td.n {{ padding-left:14px; }}
#vc-invoice tbody td {{ padding:9px 0; border-bottom:1px solid #E4CFC5;
    vertical-align:top; }}
#vc-invoice .itotals {{ margin-left:auto; width:58%; margin-top:14px; }}
#vc-invoice .trow {{ display:flex; justify-content:space-between; padding:6px 0;
    font-size:14px; color:#4C7A66; font-variant-numeric:tabular-nums; }}
#vc-invoice .trow.grand {{ color:#FFF4EF; background:#B7462A; margin-top:8px;
    padding:12px 14px; font-size:17px; font-weight:600; }}
#vc-invoice .ifoot {{ margin-top:26px; padding-top:16px;
    border-top:1px solid #E4CFC5; text-align:center; }}
#vc-invoice .barcode {{ height:42px; margin:0 auto 8px; max-width:280px;
    background:repeating-linear-gradient(90deg,#143A2D 0 2px,transparent 2px 5px,
    #143A2D 5px 6px,transparent 6px 10px,#143A2D 10px 13px,transparent 13px 15px); }}
#vc-invoice .ifoot .code {{ font-size:12px; letter-spacing:.22em;
    font-variant-numeric:tabular-nums; }}
#vc-invoice .ifoot .ty {{ font-size:12.5px; color:#4C7A66; margin-top:8px; }}
</style>

<div id="vc-receipt">
  <div class="rhead">
    <span class="rlabel">{txt['invoice']}</span>
    <span class="rmeta">{n_lines} {"line" if n_lines == 1 else "lines"}</span>
  </div>
  <div class="rstore">{store_name}</div>
  <div class="rsub">{now.strftime('%Y-%m-%d %H:%M')} &middot; {invoice_no}</div>
  {rows}
  <div class="row muted"><span class="rl">{txt['subtotal']}</span>
    <span class="rv">{subtotal:.2f}</span></div>
  {vat_row}
  <div class="total"><span class="rl">{txt['total']}</span>
    <span class="rv">{total:.2f} {txt['currency']}</span></div>
</div>
<button id="vc-print-btn" onclick="{PRINT_JS}">{txt['print']}</button>

<div id="vc-invoice">
  <div class="ihead">
    <div>
      <div class="istore">{store_name}</div>
      <div class="itag">{txt['tagline_receipt']}</div>
    </div>
    <div class="ititle">{txt['invoice']}</div>
  </div>
  <div class="imeta">
    <div><span>{txt['invoice']}</span><b>{invoice_no}</b></div>
    <div><span>Date</span><b>{now.strftime('%Y-%m-%d')}</b></div>
    <div><span>Time</span><b>{now.strftime('%H:%M')}</b></div>
  </div>
  <table>
    <thead><tr><th>Item</th><th class="n">Qty</th><th class="n">Unit</th>
      <th class="n">Amount</th></tr></thead>
    <tbody>{inv_rows}</tbody>
  </table>
  <div class="itotals">
    <div class="trow"><span>{txt['subtotal']}</span><span>{subtotal:.2f}</span></div>
    {inv_vat}
    <div class="trow grand"><span>{txt['total']}</span>
      <span>{total:.2f} {txt['currency']}</span></div>
  </div>
  <div class="ifoot">
    <div class="barcode"></div>
    <div class="code">{invoice_no}</div>
    <div class="ty">{txt['thank_you']}</div>
  </div>
</div>
</div>
"""


def _label_font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # Pillow < 10 has no size argument
        return ImageFont.load_default()


def draw_detections(img, dets, threshold):
    """Draw every box in the accent colour, per the design system.

    One colour for all classes — the model's output is one kind of assertion,
    and the class is carried by the label. Low-confidence boxes take the
    scorch/mustard tone and a thinner stroke.
    """
    out = img.copy()
    d = ImageDraw.Draw(out)
    scale = max(out.width, out.height) / 900
    font = _label_font(max(13, int(16 * scale)))
    pad = max(4, int(6 * scale))

    for det in dets:
        low = det["conf"] < threshold
        colour = C_MUSTARD if low else C_PERSIMMON
        width = max(2, int((2 if low else 3) * scale))
        x1, y1, x2, y2 = det["xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=colour, width=width)

        text = f"{det['label']} {int(det['conf'] * 100)}"
        try:
            tw = d.textlength(text, font=font)
            th = font.size
        except AttributeError:                     # very old Pillow
            tw, th = d.textsize(text, font=font)
        cy = max(0, y1 - th - pad * 2)
        d.rectangle([x1, cy, x1 + tw + pad * 2, cy + th + pad * 2], fill=colour)
        d.text((x1 + pad, cy + pad), text,
               fill=C_FOREST_DEEP if low else C_CREAM, font=font)
    return out


def _chip(label, count):
    return (f'<span class="vc-chip">{label}'
            f'<span class="vc-chip-n">&times;{count}</span></span>')


def _badge(label, conf):
    return (f'<span class="vc-badge">{label}'
            f'<span class="vc-badge-n">{int(conf * 100)}%</span></span>')


def predict(image, conf, store_name, show_vat, lang):
    txt = T[lang]
    if image is None:
        return None, "", ""

    img = ImageOps.exif_transpose(image).convert("RGB")
    result = model.predict(img, conf=DETECT_FLOOR, device="cpu", verbose=False)[0]

    dets = [{"cls": model.names[int(b.cls)],
             "label": model.names[int(b.cls)].replace("_", " "),
             "conf": float(b.conf),
             "xyxy": [float(v) for v in b.xyxy[0]]}
            for b in result.boxes]

    annotated = draw_detections(img, dets, conf)

    above = [d for d in dets if d["conf"] >= conf]
    below = [d for d in dets if d["conf"] < conf]

    if not above and not below:
        return annotated, f'<p class="vc-empty">{txt["no_items"]}</p>', ""

    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])
    cart = build_cart([d["cls"] for d in above])
    n_items = sum(it["count"] for it in cart.values())

    chips = "".join(_chip(label(it["info"]), it["count"]) for it in cart.values())
    badges = "".join(_badge(d["label"], d["conf"]) for d in below)

    note = ""
    if below:
        note = (f'<p class="vc-note">{len(below)} '
                f'{"item is" if len(below) == 1 else "items are"} below the '
                f'threshold. Lower it to include '
                f'{"it" if len(below) == 1 else "them"}.</p>')

    if not above:
        summary = f'<div class="vc-chips">{badges}</div>{note}'
        return annotated, summary, ""

    summary = (f'<p class="vc-count">'
               f'{txt["detected_count"].format(n=n_items, k=len(cart))}</p>'
               f'<div class="vc-chips">{chips}{badges}</div>{note}')
    receipt = render_receipt(cart, store_name, show_vat, lang)
    return annotated, summary, receipt


def price_list_markdown(lang):
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])
    cur = T[lang]["currency"]
    rows = [f"| | | |", "|---|---|---|"]
    for info in CATEGORY_PRICES.values():
        rows.append(f"| {info['emoji']} | {label(info)} | **{info['price']:.2f} {cur}** |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Landing page (Blush Aisle design system — see DESIGN.md / web/tokens)
# ---------------------------------------------------------------------------
WEB_DIR = BASE_DIR / "web"


def _asset_url(name, folder="assets"):
    """Serve landing images as static files.

    These were inlined as base64 data URIs, which pushed ~740KB into the Gradio
    config payload and stopped event handlers binding once deployed. Serving
    them through Gradio's file route keeps the config small.
    """
    if not (WEB_DIR / folder / name).exists():
        return ""
    return f"/gradio_api/file=web/{folder}/{name}"


SCAN_ICON = ('<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" aria-hidden="true">'
             '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3'
             'M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M8 12h8"/></svg>')


def render_examples():
    """Cards for the example strip, built from scripts/build_examples.py output."""
    path = WEB_DIR / "examples" / "examples.json"
    if not path.exists():
        return ""
    cards = []
    for i, ex in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        rows = "".join(
            f'<div class="ex-row"><span>{it["en"]}'
            f'{" &times;" + str(it["count"]) if it["count"] > 1 else ""}</span>'
            f'<span>{it["count"] * it["price"]:.2f}</span></div>'
            for it in ex["items"][:3]
        )
        extra = len(ex["items"]) - 3
        more = f'<div class="ex-more">+{extra} more</div>' if extra > 0 else ""
        cats = f'{ex["categories"]} {"category" if ex["categories"] == 1 else "categories"}'
        credit = ""
        if ex.get("credit"):
            credit = (f'<a class="ex-credit" href="{ex["url"]}" target="_blank" rel="noopener">'
                      f'Photo: {ex["credit"]} &middot; {ex["license"]} &middot; boxes added</a>')
        cards.append(
            f'<article class="ex" tabindex="0" aria-label="{ex["n"]} items, total SAR {ex["total"]:.2f}">'
            f'<div class="ex-media"><img src="{_asset_url(ex["file"], "examples")}" '
            f'alt="Model output: {ex["n"]} products detected" width="960" height="720" '
            f'loading="{"eager" if i < 3 else "lazy"}" decoding="async" draggable="false">'
            f'<span class="ex-pill">{SCAN_ICON}<span>{ex["n"]} items detected</span></span></div>'
            f'<div class="ex-receipt"><div class="ex-head"><span>{ex["invoice"]}</span>'
            f'<span>{cats}</span></div>{rows}{more}'
            f'<div class="ex-total"><span>Total incl. VAT</span>'
            f'<span>SAR {ex["total"]:.2f}</span></div>{credit}</div></article>'
        )
    return "".join(cards)


def render_categories():
    chips = []
    for info in CATEGORY_PRICES.values():
        if info["price"] > 0:
            chips.append(f'<span class="cat">{info["en"]}<i>{info["price"]:.2f}</i></span>')
        else:
            chips.append(f'<span class="cat off">{info["en"]}<i>not sold</i></span>')
    return "".join(chips)


def load_landing_html():
    html_path = WEB_DIR / "landing.html"
    if not html_path.exists():
        return ""
    return (
        html_path.read_text(encoding="utf-8")
        .replace("__CART_IMG__", _asset_url("illustration-cart.png"))
        .replace("__EXAMPLES__", render_examples())
        .replace("__CATEGORIES__", render_categories())
    )


# Autoplay and mouse-drag for the example strip. gr.HTML does not run <script>
# tags, so this goes in <head>; it polls until Gradio has rendered the strip.
# Without it the strip still works: native swipe/scroll and the arrow buttons.
STRIP_JS = """
<script>
(function () {
  function bind() {
    var t = document.getElementById('ex-track');
    if (!t || t.dataset.bound) return;
    t.dataset.bound = '1';
    var hold = false, drag = null;
    ['mouseenter', 'focusin'].forEach(function (e) { t.addEventListener(e, function () { hold = true; }); });
    ['mouseleave', 'focusout'].forEach(function (e) { t.addEventListener(e, function () { hold = false; }); });
    t.addEventListener('touchstart', function () { hold = true; }, { passive: true });
    t.addEventListener('touchend', function () { setTimeout(function () { hold = false; }, 5000); }, { passive: true });
    t.addEventListener('pointerdown', function (e) {
      if (e.pointerType !== 'mouse' || e.button !== 0) return;
      drag = { x: e.clientX, left: t.scrollLeft };
      t.classList.add('dragging');
    });
    window.addEventListener('pointermove', function (e) {
      if (drag) t.scrollLeft = drag.left - (e.clientX - drag.x);
    });
    window.addEventListener('pointerup', function () {
      if (!drag) return;
      drag = null;
      t.classList.remove('dragging');
    });
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    setInterval(function () {
      if (hold || drag || document.hidden) return;
      var c = t.querySelector('.ex');
      if (!c) return;
      if (t.scrollLeft + t.clientWidth >= t.scrollWidth - 8) t.scrollTo({ left: 0, behavior: 'smooth' });
      else t.scrollBy({ left: c.offsetWidth + 22, behavior: 'smooth' });
    }, 3600);
  }
  setInterval(bind, 700);
})();
</script>
"""


LANDING_HTML = load_landing_html()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@400;500;700&display=swap");

:root, .gradio-container {
  --vc-blush:#F7DFD9; --vc-cream:#FFF4EF; --vc-forest:#1E5140; --vc-forest-deep:#143A2D;
  --vc-persimmon:#D95F3B; --vc-persimmon-deep:#B7462A; --vc-persimmon-wash:#FBE3D8;
  --vc-mustard-wash:#F6E4BE; --vc-sage:#A9C9B8; --vc-moss:#4C7A66; --vc-hairline:#EBCFC6;
}

/* Blush ground everywhere, including the area behind the app. */
body, gradio-app, .gradio-container { background: var(--vc-blush) !important; }
/* Gradio's container is flex:1 1 auto inside a flex parent, so it collapses to
   content width. Force it to fill before max-width can do anything. */
.gradio-container { max-width: 1840px !important; width: 100% !important;
  flex: 1 1 100% !important; margin: 0 auto !important;
  font-family: "Instrument Sans", system-ui, sans-serif !important; }
.fillable, main.app { max-width: none !important; width: 100% !important; }
.gradio-container .row .column { min-width: 0 !important; }

/* The scanner reads as one cream card sitting on the blush page. */
#scanner { scroll-margin-top: 16px; background: var(--vc-cream); border-radius: 36px;
  padding: 40px 44px; margin-top: 8px; }
@media (max-width: 900px) { #scanner { padding: 32px 24px; border-radius: 32px; } }
#scanner h2 { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800;
  letter-spacing: -.03em; font-size: clamp(30px,3.6vw,44px); line-height: 1.05;
  color: var(--vc-forest-deep); margin: 0 0 8px; }
#scanner .vc-lede { color: var(--vc-moss); margin: 0 0 24px; }

/* Nothing in this system has a square corner. */
#scanner .block, #scanner .form, #scanner .gr-box, #scanner .gr-panel,
#scanner .image-container, #scanner img, #scanner .wrap { border-radius: 32px !important; }
#scanner button, #scanner input[type="text"], #scanner .gr-check-radio { border-radius: 999px !important; }
#scanner .block { background: #fff !important; border-color: var(--vc-hairline) !important; }

/* Primary action = persimmon. Exactly one per view. */
#scanner button.primary, #scanner .primary > button {
  background: var(--vc-persimmon) !important; color: var(--vc-cream) !important;
  border: none !important; font-weight: 500 !important; font-size: 17px !important;
  min-height: 58px !important; transition: background 160ms cubic-bezier(.22,.61,.36,1); }
#scanner button.primary:hover, #scanner .primary > button:hover {
  background: var(--vc-persimmon-deep) !important; }
#scanner button.secondary, #scanner .secondary > button {
  background: var(--vc-persimmon-wash) !important; color: var(--vc-persimmon-deep) !important;
  border: none !important; font-weight: 500 !important; }

#scanner label, #scanner .gr-input-label, #scanner span[data-testid="block-info"] {
  color: var(--vc-forest-deep) !important; font-weight: 500 !important; }
#scanner .prose, #scanner .md { color: var(--vc-forest-deep) !important; }
#scanner a { color: var(--vc-persimmon-deep); }

/* ── Scanner chrome, per ui_kits/scanner ───────────────────────────────── */
#vc-band { background: var(--vc-forest) !important; border-radius: 999px !important;
  padding: 6px 10px 6px 22px !important; margin-bottom: 28px !important;
  align-items: center !important; }
#vc-band .vc-mark { color: var(--vc-cream); font-family: "Bricolage Grotesque", sans-serif;
  font-weight: 800; font-size: 18px; letter-spacing: -.035em; }
#vc-band .block, #vc-band .form { background: transparent !important; border: none !important; }
.lang-toggle button { min-width: 84px; background: transparent !important;
  color: var(--vc-cream) !important; border: 1px solid rgba(255,244,239,.4) !important; }
.lang-toggle button:hover { background: rgba(255,244,239,.12) !important; }

.vc-eyebrow { display: block; color: var(--vc-moss); font-size: 13px; font-weight: 500;
  margin: 0 0 10px; }

/* Text blocks are not cards. The rounded-white treatment is for real controls
   only — applied to gr.HTML/gr.Markdown it turned the Input/Results labels and
   the heading into empty-looking pill boxes. */
#scanner .block:has(> .html-container), #scanner .block:has(> .prose),
#scanner .block:has(> .md) {
  background: transparent !important; border: none !important;
  box-shadow: none !important; border-radius: 0 !important; padding: 0 !important; }
#scanner .html-container, #scanner .prose, #scanner .md {
  background: transparent !important; border: none !important;
  padding: 0 !important; overflow: visible !important; }
#scanner h2 { overflow: visible !important; }


/* Threshold sits in its own cool panel, as in the kit. */
#vc-threshold { background: var(--vc-sage) !important; border: none !important;
  border-radius: 24px !important; padding: 16px 20px !important; }
#vc-threshold .block { background: transparent !important; border: none !important; }
#vc-threshold label > span, #vc-threshold span[data-testid="block-info"] {
  color: var(--vc-forest-deep) !important; }

/* Detections: chips for confident reads, scorch badges for low-confidence ones. */
.vc-count { color: var(--vc-forest-deep); font-weight: 500; margin: 4px 0 12px; font-size: 15px; }
.vc-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }
.vc-chip, .vc-badge { display: inline-flex; align-items: center; gap: 8px;
  border-radius: 999px; padding: 8px 16px; font-size: 14px; font-weight: 500; }
.vc-chip { background: var(--vc-persimmon-wash); color: var(--vc-persimmon-deep); }
.vc-chip-n, .vc-badge-n { font-variant-numeric: tabular-nums; opacity: .75; }
.vc-badge { background: var(--vc-mustard-wash); color: #7A5310; }
.vc-note { color: var(--vc-moss); font-size: 13px; margin: 10px 0 0; }
.vc-empty { color: var(--vc-moss); font-size: 16px; margin: 8px 0 0; }

.vc-foot { margin-top: 32px; background: var(--vc-forest); color: var(--vc-cream);
  border-radius: 999px; padding: 14px 26px; font-size: 13px; font-weight: 500; }

@media (max-width: 860px) { #vc-band { border-radius: 28px !important; } }

/* The design system is light-only. Neutralise Gradio dark mode in CSS rather
   than redirecting the page — a redirect loops and kills every event. */
.dark, .dark body, .dark .gradio-container {
  --body-background-fill: var(--vc-blush) !important;
  --background-fill-primary: #fff !important;
  --background-fill-secondary: var(--vc-cream) !important;
  --block-background-fill: #fff !important;
  --border-color-primary: var(--vc-hairline) !important;
  --body-text-color: var(--vc-forest-deep) !important;
  --body-text-color-subdued: var(--vc-moss) !important;
  --input-background-fill: #fff !important;
  --block-label-text-color: var(--vc-forest-deep) !important;
  --block-title-text-color: var(--vc-forest-deep) !important;
  color-scheme: light !important;
}
.dark #scanner input, .dark #scanner textarea {
  background: #fff !important; color: var(--vc-forest-deep) !important; }
.dark #scanner .block { background: #fff !important; }
.dark #scanner, .dark #scanner .prose, .dark #scanner label,
.dark #scanner .md { color: var(--vc-forest-deep) !important; }

#scanner label > span, #scanner .block > label > span,
#scanner span[data-testid="block-info"], #scanner .gr-input-label {
  background: transparent !important; color: var(--vc-forest-deep) !important;
  font-weight: 500 !important; padding-left: 0 !important; }
#scanner input[type="text"], #scanner textarea, #scanner .gr-text-input {
  background: #fff !important; color: var(--vc-forest-deep) !important;
  border: 1px solid var(--vc-hairline) !important; }
#scanner input[type="number"] { background:#fff !important; color: var(--vc-forest-deep) !important; }
#scanner .wrap.default, #scanner .empty { background: #fff !important; color: var(--vc-moss) !important; }
#scanner input[type="range"]::-webkit-slider-thumb { background: var(--vc-persimmon) !important; }
#scanner .head, #scanner .label-wrap span { color: var(--vc-forest-deep) !important; }
#scanner .label, #scanner .label span { color: var(--vc-forest-deep) !important; }
#scanner .gr-accordion, #scanner details summary { color: var(--vc-forest-deep) !important; }
#scanner .table-wrap, #scanner table { background:#fff !important; color: var(--vc-forest-deep) !important; }


/* RTL mode */
.rtl-mode, .rtl-mode * { direction: rtl !important; }
.rtl-mode .gradio-container,
.rtl-mode label,
.rtl-mode .markdown,
.rtl-mode .prose { text-align: right !important; }

/* In RTL the labels reorder via direction:rtl (max on left, min on right).
   Flip the range track horizontally so the orange fill grows from the "0" side
   toward the thumb, and dragging LEFT increases the value. */
.rtl-mode .slider_input_container input[type="range"] { transform: scaleX(-1); }

/* Last word on text blocks: the dark-mode neutralisation above also forces a
   white card background, so these have to come last and cover .dark too. */
#scanner .block:has(> .html-container), .dark #scanner .block:has(> .html-container),
#scanner .block:has(> .prose), .dark #scanner .block:has(> .prose),
#scanner .block:has(> .md), .dark #scanner .block:has(> .md) {
  background: transparent !important; border: none !important;
  box-shadow: none !important; border-radius: 0 !important; padding: 0 !important; }
#vc-heading, .dark #vc-heading, #scanner #vc-heading {
  background: transparent !important; border: none !important;
  box-shadow: none !important; border-radius: 0 !important;
  padding: 0 !important; margin-bottom: 22px !important; }

/* Component labels are captions, not badges. Gradio tints them with the
   primary hue, which turned "Upload image" and "Detections" into loud pills. */
#scanner [data-testid="block-label"], .dark #scanner [data-testid="block-label"] {
  background: transparent !important; color: var(--vc-moss) !important;
  border: none !important; box-shadow: none !important;
  font-weight: 500 !important; padding-left: 0 !important; }
#scanner [data-testid="block-label"] svg {
  color: var(--vc-moss) !important; opacity: .6; }

/* ── Scanner polish ──────────────────────────────────────────────────────── */
/* Sample gallery: rounded tiles in a tidy row, not floating circles. The
   gallery shrink-wraps by default (it measured 104px wide), so every level
   has to be told to fill before the tiles can size off it. */
#scanner .gallery { width: 100% !important; display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important; gap: 10px !important; }
#scanner .gallery-item { width: 100% !important; display: block !important;
  padding: 0 !important; border: none !important; background: none !important; }
#scanner .gallery-item .container { display: block !important; width: 100% !important;
  height: auto !important; aspect-ratio: 1 / 1; overflow: hidden;
  border-radius: 18px !important; border: none !important; }
#scanner .gallery-item img { width: 100% !important; height: 100% !important;
  max-width: none !important; object-fit: cover; border-radius: 0 !important;
  transition: transform 300ms cubic-bezier(.22,.61,.36,1); }
#scanner .gallery-item:hover img { transform: scale(1.05); }

/* Gradio paints grouped fields' gaps with a .styler background (#EBCFC6),
   which showed up as pink boxes in the threshold panel and between fields. */
#vc-input .styler, #vc-threshold .styler { background: transparent !important; }
#vc-input .styler { gap: 12px !important; }
#scanner .form, .dark #scanner .form { background: transparent !important;
  border: none !important; box-shadow: none !important; gap: 12px !important; }

/* Threshold: the slider sits directly on the sage panel, no white card inside. */
#vc-threshold .block, .dark #vc-threshold .block, #vc-threshold .form {
  background: transparent !important; box-shadow: none !important; border: none !important; }

/* Language toggle stays a compact pill instead of stretching across the band. */
#vc-band .lang-toggle { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }

/* Empty results read as a deliberate drop zone, not a blank white slab. */
#vc-annotated:not(:has(img)), .dark #vc-annotated:not(:has(img)) {
  background: #FBEFEA !important; border: 1.5px dashed #EBCFC6 !important; }
#vc-annotated:not(:has(img))::after {
  content: "Your scan will appear here"; position: absolute; left: 50%; top: 56%;
  transform: translateX(-50%); color: #4C7A66; font-size: 15px; white-space: nowrap; }

/* Gradio's own footer (API / built-with links) is noise in a product demo. */
footer { display: none !important; }
"""


def toggle_lang(lang):
    new_lang = "ar" if lang == "en" else "en"
    t = T[new_lang]
    btn_label = "AR 🌐" if new_lang == "en" else "EN 🌐"
    rtl_class = "rtl-mode" if new_lang == "ar" else ""
    return (
        new_lang,
        gr.update(value=btn_label),
        gr.update(value=f"<h2>{t['title']}</h2><p class='vc-lede'>{t['tagline']}</p>"),
        gr.update(label=t["image_label"]),
        gr.update(label=t["confidence"]),
        gr.update(label=t["store"]),
        gr.update(label=t["vat"]),
        gr.update(value=t["scan"]),
        gr.update(label=t["detections"]),
        gr.update(label=t["samples"]),
        gr.update(value=t["shuffle"]),
        gr.update(label=t["prices_label"], value=price_list_markdown(new_lang)),
        gr.update(elem_classes=[rtl_class] if rtl_class else []),
    )


def open_sample(path):
    """Return a decoded image, not a path.

    gr.Image(type="pil") can be handed a filepath, but Gradio then has to serve
    that absolute path back to the browser, which fails silently on Spaces.
    Decoding here sidesteps the whole allowed-paths question.
    """
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except (OSError, ValueError):
        return gr.update()


def load_random_sample():
    """One button, one handler — the demo path that cannot break."""
    picks = pick_random_samples(1)
    return open_sample(picks[0]) if picks else gr.update()


def shuffle_samples():
    picks = pick_random_samples(3)
    return gr.update(samples=[[p] for p in picks]), picks


def choose_sample(samples, evt: gr.SelectData):
    """gr.Dataset reports the selected row index via SelectData, not the row itself."""
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    try:
        return open_sample(samples[int(idx)])
    except (TypeError, ValueError, IndexError):
        return gr.update()


VC_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="emerald",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"],
)

with gr.Blocks(css=CUSTOM_CSS, theme=VC_THEME, head=STRIP_JS,
               title="Vision Checkout", fill_width=True) as demo:
    lang_state = gr.State("en")
    samples_state = gr.State(pick_random_samples(3))

    if LANDING_HTML:
        gr.HTML(LANDING_HTML)

    root = gr.Column(elem_classes=[], elem_id="scanner")

    with root:
        # Header band — the forest strip that ties the tool to the poster.
        with gr.Row(elem_id="vc-band"):
            with gr.Column(scale=6, min_width=200):
                gr.HTML("<span class='vc-mark'>Vision Checkout</span>")
            with gr.Column(scale=1, min_width=110, elem_classes=["lang-toggle"]):
                lang_btn = gr.Button("AR 🌐", size="sm")

        header_md = gr.Markdown(
            f"<h2>{T['en']['title']}</h2>"
            f"<p class='vc-lede'>{T['en']['tagline']}</p>",
            elem_id="vc-heading",
        )

        with gr.Row(equal_height=False):
            # ── LEFT: input, narrow ──
            with gr.Column(scale=1, min_width=320, elem_id="vc-input"):
                gr.HTML("<span class='vc-eyebrow'>Input</span>")
                image_in = gr.Image(label=T["en"]["image_label"], type="pil",
                                    sources=["upload", "clipboard"], height=250)
                with gr.Row():
                    random_btn = gr.Button("Load a random sample", size="sm",
                                           variant="secondary")
                    shuffle_btn = gr.Button(T["en"]["shuffle"], size="sm")

                samples_gallery = gr.Dataset(
                    components=[image_in],
                    samples=[[p] for p in samples_state.value],
                    label=T["en"]["samples"],
                    samples_per_page=3,
                )

                with gr.Group(elem_id="vc-threshold"):
                    conf_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05,
                                            label=T["en"]["confidence"])

                store_in = gr.Textbox(label=T["en"]["store"], value="VISION MART")
                vat_in = gr.Checkbox(label=T["en"]["vat"], value=True)
                scan_btn = gr.Button(T["en"]["scan"], variant="primary")

                with gr.Accordion(T["en"]["prices_label"], open=False) as prices_acc:
                    prices_md = gr.Markdown(price_list_markdown("en"))

            # ── RIGHT: results, wide ──
            with gr.Column(scale=2, elem_id="vc-results"):
                gr.HTML("<span class='vc-eyebrow'>Results</span>")
                annotated_out = gr.Image(label=T["en"]["detections"], height=430,
                                         interactive=False, elem_id="vc-annotated")
                summary_out = gr.HTML()
                receipt_out = gr.HTML()

        gr.HTML("<div class='vc-foot'>Vision Checkout &mdash; alpha</div>")

    # ── wiring ──
    scan_btn.click(
        predict,
        inputs=[image_in, conf_slider, store_in, vat_in, lang_state],
        outputs=[annotated_out, summary_out, receipt_out],
    )

    lang_btn.click(
        toggle_lang,
        inputs=[lang_state],
        outputs=[lang_state, lang_btn, header_md, image_in, conf_slider, store_in,
                 vat_in, scan_btn, annotated_out, samples_gallery, shuffle_btn,
                 prices_md, root],
    )

    shuffle_btn.click(shuffle_samples, outputs=[samples_gallery, samples_state])

    samples_gallery.select(choose_sample, inputs=[samples_state], outputs=[image_in])

    random_btn.click(load_random_sample, outputs=[image_in])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,
                allowed_paths=[str(TEST_IMAGES_DIR), str(WEB_DIR)])

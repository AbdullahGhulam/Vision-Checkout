"""Vision Checkout — Gradio app for Hugging Face Spaces.

Single-file Gradio app: upload a tray photo, run YOLO detection, generate a receipt.
"""
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
        "title": "Vision Checkout",
        "tagline": "Smart AI cashier — upload a tray photo and get a receipt.",
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
        "title": "الكاشير الذكي",
        "tagline": "ارفع صورة المنتجات واحصل على الفاتورة فوراً.",
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
    "var r=document.getElementById('vc-receipt');"
    "if(r){"
    "var w=document.getElementById('vc-receipt-wrap');"
    "var dir=(w&&w.getAttribute('dir'))||'ltr';"
    "var s=document.getElementById('vc-rstyle');"
    "var f=document.createElement('iframe');"
    # A real page-sized layout box, parked off-screen. A 1px iframe makes the
    # receipt lay out at 1px wide and print squashed.
    "f.setAttribute('style','position:fixed;left:-10000px;top:0;width:794px;"
    "height:1123px;border:0;');"
    "document.body.appendChild(f);"
    "var d=f.contentWindow.document;d.open();"
    "d.write('<!doctype html><html><head><meta charset=utf-8>'"
    "+(s?s.outerHTML:'')"
    "+'<style>@page{margin:14mm}body{margin:0;background:#fff}"
    "#vc-receipt{max-width:520px;margin:0 auto;border:none}"
    "#vc-print-btn{display:none}</style></head><body dir='+dir+'>'"
    "+r.outerHTML+'</body></html>');"
    "d.close();"
    "var L=d.createElement('link');L.rel='stylesheet';"
    "L.href='https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:"
    "opsz,wght@12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap';"
    "d.head.appendChild(L);"
    "setTimeout(function(){"
    "try{f.contentWindow.focus();f.contentWindow.print();}catch(e){window.print();}"
    "setTimeout(function(){if(f.parentNode){f.parentNode.removeChild(f);}},1500);"
    "},600);"
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
    font-size:15px; cursor:pointer; font-family:inherit;
    transition:background 160ms cubic-bezier(.22,.61,.36,1); }}
#vc-print-btn:hover {{ background:#FBE3D8; }}
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


def _asset_url(name):
    """Serve the illustration as a static file.

    These were inlined as base64 data URIs, which pushed ~740KB into the Gradio
    config payload and stopped event handlers binding once deployed. Serving
    them through Gradio's file route keeps the config small.
    """
    if not (WEB_DIR / "assets" / name).exists():
        return ""
    return "/gradio_api/file=web/assets/" + name


def load_landing_html():
    html_path = WEB_DIR / "landing.html"
    if not html_path.exists():
        return ""
    return (
        html_path.read_text(encoding="utf-8")
        .replace("__CART_IMG__", _asset_url("illustration-cart.png"))
        .replace("__PRODUCE_IMG__", _asset_url("illustration-trolley-produce.png"))
    )


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

with gr.Blocks(css=CUSTOM_CSS, theme=VC_THEME,
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
            f"<p class='vc-lede'>{T['en']['tagline']}</p>"
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
                                         interactive=False)
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

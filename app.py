"""Vision Checkout — Gradio app for Hugging Face Spaces.

Single-file Gradio app: upload a tray photo, run YOLO detection, generate a receipt.
"""
import base64
import random
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps
from ultralytics import YOLO

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


def render_receipt(cart, store_name, show_vat, lang):
    txt = T[lang]
    direction = "rtl" if lang == "ar" else "ltr"
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])

    subtotal = sum(it["info"]["price"] * it["count"] for it in cart.values())
    vat = subtotal * 0.15 if show_vat else 0
    total = subtotal + vat
    now = datetime.now()
    invoice_no = f"VC-{random.randint(10000, 99999)}"

    rows = "".join(
        f'<div class="row"><span>{it["info"]["emoji"]} {label(it["info"])} ×{it["count"]}</span>'
        f'<span>{it["info"]["price"] * it["count"]:.2f} {txt["currency"]}</span></div>'
        for it in cart.values()
    )
    vat_row = (
        f'<div class="row vat"><span>{txt["vat_label"]}</span>'
        f'<span>{vat:.2f} {txt["currency"]}</span></div>'
        if show_vat else ""
    )

    return f"""
<div id="vc-receipt-wrap" dir="{direction}">
<style>
#vc-receipt-wrap {{ display:flex; flex-direction:column; align-items:center; margin:8px 0;
    font-family:'Instrument Sans',system-ui,sans-serif; }}
#vc-receipt {{ background:#FFF4EF; border:1px solid #EBCFC6; border-radius:32px;
    padding:32px 28px; max-width:400px; width:100%; color:#143A2D; }}
#vc-receipt .hd {{ text-align:center; border-bottom:1px solid #EBCFC6;
    padding-bottom:16px; margin-bottom:16px; }}
#vc-receipt .store {{ font-family:'Bricolage Grotesque','Instrument Sans',sans-serif;
    font-size:1.5rem; font-weight:800; letter-spacing:-.03em; color:#143A2D; }}
#vc-receipt .sub {{ font-size:0.78rem; color:#4C7A66; margin-top:4px; }}
#vc-receipt .row {{ display:flex; justify-content:space-between; padding:9px 0;
    font-size:0.92rem; border-bottom:1px solid #F6E4BE; font-variant-numeric:tabular-nums; }}
#vc-receipt .row.vat {{ color:#4C7A66; font-size:0.85rem; }}
#vc-receipt .total {{ display:flex; justify-content:space-between;
    border-top:1px solid #EBCFC6; padding-top:14px; margin-top:12px;
    font-weight:600; font-size:1.15rem; color:#B7462A; font-variant-numeric:tabular-nums; }}
#vc-receipt .ft {{ text-align:center; margin-top:18px; padding-top:16px;
    border-top:1px solid #EBCFC6; font-size:0.78rem; color:#4C7A66; }}
#vc-receipt .barcode {{ font-size:1.6rem; letter-spacing:3px; color:#A9C9B8; margin:8px 0 4px; }}
#vc-print-btn {{ margin-top:16px; padding:14px 30px; background:#D95F3B; color:#FFF4EF;
    border:none; border-radius:999px; font-weight:500; font-size:15px; cursor:pointer;
    transition:background 160ms cubic-bezier(.22,.61,.36,1); }}
#vc-print-btn:hover {{ background:#B7462A; }}
@media print {{
    body * {{ visibility:hidden !important; }}
    #vc-receipt, #vc-receipt * {{ visibility:visible !important; }}
    #vc-receipt {{ position:absolute; left:0; top:0; border:none; box-shadow:none; }}
    #vc-print-btn {{ display:none !important; }}
}}
</style>
<div id="vc-receipt">
  <div class="hd">
    <div class="store">🛒 {store_name}</div>
    <div class="sub">{txt['tagline_receipt']}</div>
    <div class="sub">{now.strftime('%Y-%m-%d')} &nbsp; {now.strftime('%H:%M:%S')}</div>
    <div class="sub">{txt['invoice']}: {invoice_no}</div>
  </div>
  {rows}
  <div class="row" style="margin-top:6px;"><span>{txt['subtotal']}</span>
    <span>{subtotal:.2f} {txt['currency']}</span></div>
  {vat_row}
  <div class="total"><span>{txt['total']}</span>
    <span>{total:.2f} {txt['currency']}</span></div>
  <div class="ft">
    <div class="barcode">||||| |||| ||||| ||||</div>
    <div>{invoice_no}</div>
    <div style="margin-top:8px;">{txt['thank_you']}</div>
  </div>
</div>
<button id="vc-print-btn" onclick="window.print()">{txt['print']}</button>
</div>
"""


def predict(image, conf, store_name, show_vat, lang):
    txt = T[lang]
    if image is None:
        return None, "", ""

    img = ImageOps.exif_transpose(image).convert("RGB")
    result = model.predict(img, conf=conf, device="cpu", verbose=False)[0]
    # result.plot() returns a BGR ndarray; flip channels and copy() so PIL gets a contiguous RGB array.
    annotated = Image.fromarray(result.plot()[:, :, ::-1].copy())

    classes = [model.names[int(b.cls)] for b in result.boxes]
    if not classes:
        return annotated, f"**{txt['no_items']}**", ""

    cart = build_cart(classes)
    n_items = sum(it["count"] for it in cart.values())
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])

    summary_lines = [f"**{txt['detected_count'].format(n=n_items, k=len(cart))}**", ""]
    for it in cart.values():
        info = it["info"]
        summary_lines.append(
            f"- {info['emoji']} **{label(info)}** ×{it['count']} — "
            f"`{info['price'] * it['count']:.2f} {txt['currency']}`"
        )
    summary = "\n".join(summary_lines)
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


def _data_uri(path):
    """Inline an image so the page needs no static-file routing on Spaces.

    Returns "" if the file is missing or is an unresolved Git LFS pointer, so the
    page degrades to an empty frame instead of embedding garbage.
    """
    if not path.exists():
        return ""
    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def load_landing_html():
    html_path = WEB_DIR / "landing.html"
    if not html_path.exists():
        return ""
    return (
        html_path.read_text(encoding="utf-8")
        .replace("__CART_IMG__", _data_uri(WEB_DIR / "assets" / "illustration-cart.png"))
        .replace("__PRODUCE_IMG__", _data_uri(WEB_DIR / "assets" / "illustration-trolley-produce.png"))
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
.gradio-container { max-width: 1200px !important; margin: 0 auto !important;
  font-family: "Instrument Sans", system-ui, sans-serif !important; }
.fillable { max-width: none !important; width: 100% !important; }
.gradio-container .row .column { min-width: 0 !important; }

/* The scanner reads as one cream card sitting on the blush page. */
#scanner { scroll-margin-top: 16px; background: var(--vc-cream); border-radius: 44px;
  padding: 48px; margin-top: 24px; }
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
.lang-toggle button { min-width: 90px; }
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


def shuffle_samples():
    picks = pick_random_samples(3)
    return gr.update(samples=[[p] for p in picks]), picks


def choose_sample(samples, evt: gr.SelectData):
    """gr.Dataset reports the selected row index via SelectData, not the row itself."""
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    try:
        return samples[int(idx)]
    except (TypeError, ValueError, IndexError):
        return gr.update()


VC_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="emerald",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"],
)

FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=VC_THEME, js=FORCE_LIGHT_JS,
               title="Vision Checkout", fill_width=True) as demo:
    lang_state = gr.State("en")
    samples_state = gr.State(pick_random_samples(3))

    if LANDING_HTML:
        gr.HTML(LANDING_HTML)

    root = gr.Column(elem_classes=[], elem_id="scanner")

    with root:
        with gr.Row():
            with gr.Column(scale=6):
                header_md = gr.Markdown(
                    f"<h2>{T['en']['title']}</h2>"
                    f"<p class='vc-lede'>{T['en']['tagline']}</p>"
                )
            with gr.Column(scale=1, min_width=120, elem_classes=["lang-toggle"]):
                lang_btn = gr.Button("AR 🌐", size="sm")

        with gr.Row():
            # ── LEFT column: results (detections image + receipt) ──
            with gr.Column(scale=1):
                annotated_out = gr.Image(label=T["en"]["detections"], height=380,
                                         interactive=False)
                summary_out = gr.Markdown()
                receipt_out = gr.HTML()

            # ── RIGHT column: upload, parameters, samples ──
            with gr.Column(scale=1):
                image_in = gr.Image(label=T["en"]["image_label"], type="pil",
                                    sources=["upload", "clipboard"], height=380)

                with gr.Group():
                    conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.05,
                                            label=T["en"]["confidence"])
                    store_in = gr.Textbox(label=T["en"]["store"], value="VISION MART")
                    vat_in = gr.Checkbox(label=T["en"]["vat"], value=True)

                scan_btn = gr.Button(T["en"]["scan"], variant="primary")

                with gr.Accordion(T["en"]["prices_label"], open=False) as prices_acc:
                    prices_md = gr.Markdown(price_list_markdown("en"))

                samples_gallery = gr.Dataset(
                    components=[image_in],
                    samples=[[p] for p in samples_state.value],
                    label=T["en"]["samples"],
                    samples_per_page=3,
                )
                shuffle_btn = gr.Button(T["en"]["shuffle"], size="sm")

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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

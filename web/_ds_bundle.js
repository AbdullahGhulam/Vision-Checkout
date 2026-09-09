/* @ds-bundle: {"format":4,"namespace":"VisionCheckoutDesignSystem_a34a69","components":[{"name":"Button","sourcePath":"components/core/Button.jsx"},{"name":"Card","sourcePath":"components/core/Card.jsx"},{"name":"Divider","sourcePath":"components/core/Divider.jsx"},{"name":"Input","sourcePath":"components/core/Input.jsx"},{"name":"Panel","sourcePath":"components/core/Panel.jsx"},{"name":"ThresholdSlider","sourcePath":"components/core/ThresholdSlider.jsx"},{"name":"Tooltip","sourcePath":"components/core/Tooltip.jsx"},{"name":"Price","sourcePath":"components/data/Price.jsx"},{"name":"BoundingBox","sourcePath":"components/detection/BoundingBox.jsx"},{"name":"ConfidenceBadge","sourcePath":"components/detection/ConfidenceBadge.jsx"},{"name":"DetectionChip","sourcePath":"components/detection/DetectionChip.jsx"},{"name":"HeaderBand","sourcePath":"components/layout/HeaderBand.jsx"},{"name":"SectionHeading","sourcePath":"components/layout/SectionHeading.jsx"}],"sourceHashes":{"components/core/Button.jsx":"8ccc63501b9d","components/core/Card.jsx":"13935b3cf17d","components/core/Divider.jsx":"d5724d72c998","components/core/Input.jsx":"13f1a8553717","components/core/Panel.jsx":"5af84e0d6f06","components/core/ThresholdSlider.jsx":"4f6904cd4eb3","components/core/Tooltip.jsx":"5f3424eafb7f","components/data/Price.jsx":"48ceed936fd4","components/detection/BoundingBox.jsx":"7fd4f656a2c8","components/detection/ConfidenceBadge.jsx":"2c36135e0424","components/detection/DetectionChip.jsx":"1a7dc659fdf0","components/layout/HeaderBand.jsx":"06ac6c45020d","components/layout/SectionHeading.jsx":"2bc4cf9b678d","ui_kits/landing/CartPlate.jsx":"e340795f5652","ui_kits/landing/Hero.jsx":"89c622356f70","ui_kits/landing/LandingPoster.jsx":"f9ecd76b207a","ui_kits/landing/Nav.jsx":"ac27d471f4cd","ui_kits/landing/Sections.jsx":"ad3cc32922e7","ui_kits/scanner/InputColumn.jsx":"44774f6310d0","ui_kits/scanner/ResultsColumn.jsx":"3390490ef42d","ui_kits/scanner/samples.js":"01a0c8e49cd7"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.VisionCheckoutDesignSystem_a34a69 = window.VisionCheckoutDesignSystem_a34a69 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/core/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const base = {
  fontFamily: 'var(--font-sans)',
  borderRadius: 'var(--radius-sm)',
  border: 'none',
  boxShadow: 'none',
  cursor: 'pointer',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 'var(--space-sm)',
  transition: 'background-color var(--transition-state), color var(--transition-state)',
  textDecoration: 'none',
  whiteSpace: 'nowrap'
};
const variants = {
  primary: {
    background: 'var(--color-primary-strong)',
    color: 'var(--text-on-primary)',
    height: 'var(--control-height-lg)',
    padding: '0 var(--space-lg)',
    fontSize: 'var(--type-headline-md-size)',
    lineHeight: 'var(--type-headline-md-lh)',
    fontWeight: 'var(--weight-bold)'
  },
  secondary: {
    background: 'var(--surface-cool)',
    color: 'var(--text-body)',
    height: 'var(--control-height-md)',
    padding: '0 var(--pad-button-md)',
    fontSize: 'var(--type-label-size)',
    lineHeight: 'var(--type-label-lh)',
    letterSpacing: 'var(--type-label-ls)',
    textTransform: 'uppercase',
    fontFamily: 'var(--font-mono)',
    fontWeight: 'var(--weight-bold)'
  }
};
function Button({
  variant = 'primary',
  disabled = false,
  fullWidth = false,
  type = 'button',
  href,
  children,
  style,
  ...rest
}) {
  const [hover, setHover] = React.useState(false);
  const v = variants[variant] || variants.primary;
  const hoverBg = variant === 'primary' ? 'var(--color-primary-hover)' : '#AFC9D4';
  const s = {
    ...base,
    ...v,
    width: fullWidth ? '100%' : undefined,
    background: hover && !disabled ? hoverBg : v.background,
    opacity: disabled ? 0.45 : 1,
    cursor: disabled ? 'not-allowed' : 'pointer',
    ...style
  };
  const handlers = {
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false)
  };
  if (href) return /*#__PURE__*/React.createElement("a", _extends({
    href: href,
    style: s
  }, handlers, rest), children);
  return /*#__PURE__*/React.createElement("button", _extends({
    type: type,
    disabled: disabled,
    style: s
  }, handlers, rest), children);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Button.jsx", error: String((e && e.message) || e) }); }

// components/core/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Card({
  padded = true,
  children,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      background: 'var(--surface-card)',
      color: 'var(--text-body)',
      borderRadius: 'var(--radius-lg)',
      padding: padded ? 'var(--pad-card)' : 0,
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-body-md-size)',
      lineHeight: 'var(--type-body-md-lh)',
      boxShadow: 'none',
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Card.jsx", error: String((e && e.message) || e) }); }

// components/core/Divider.jsx
try { (() => {
function Divider({
  vertical = false,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    role: "separator",
    style: {
      background: 'var(--border-hairline)',
      borderRadius: 'var(--radius-none)',
      width: vertical ? 'var(--border-width-hairline)' : '100%',
      height: vertical ? '100%' : 'var(--border-width-hairline)',
      flex: 'none',
      ...style
    }
  });
}
Object.assign(__ds_scope, { Divider });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Divider.jsx", error: String((e && e.message) || e) }); }

// components/core/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Input({
  error = false,
  label,
  hint,
  id,
  style,
  ...rest
}) {
  const [focus, setFocus] = React.useState(false);
  const inputId = id || React.useId();
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-xs)'
    }
  }, label && /*#__PURE__*/React.createElement("label", {
    htmlFor: inputId,
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--type-label-size)',
      lineHeight: 'var(--type-label-lh)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      color: 'var(--text-body)'
    }
  }, label), /*#__PURE__*/React.createElement("input", _extends({
    id: inputId,
    onFocus: () => setFocus(true),
    onBlur: () => setFocus(false),
    style: {
      height: 'var(--control-height-md)',
      padding: '0 var(--pad-button-md)',
      background: 'var(--surface-card)',
      color: error ? 'var(--text-error)' : 'var(--text-body)',
      border: `var(--border-width-hairline) solid ${error ? 'var(--color-error)' : 'var(--border-hairline)'}`,
      borderRadius: 'var(--radius-sm)',
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-body-md-size)',
      lineHeight: 'var(--type-body-md-lh)',
      boxShadow: 'none',
      outline: focus ? 'var(--focus-ring-width) solid var(--border-focus)' : 'none',
      outlineOffset: 'var(--focus-ring-offset)',
      transition: 'border-color var(--transition-state), color var(--transition-state)',
      ...style
    }
  }, rest)), hint && /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-caption-size)',
      lineHeight: 'var(--type-caption-lh)',
      color: error ? 'var(--text-error)' : 'var(--text-muted)'
    }
  }, error && /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinecap: "round",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "9"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M12 7.5v6M12 16.5h.01"
  })), hint));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Input.jsx", error: String((e && e.message) || e) }); }

// components/core/Panel.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Panel({
  tone = 'warm',
  children,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      background: tone === 'cool' ? 'var(--surface-cool)' : 'var(--surface-warm)',
      color: 'var(--text-body)',
      borderRadius: 'var(--radius-none)',
      padding: 'var(--pad-panel)',
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-body-md-size)',
      lineHeight: 'var(--type-body-md-lh)',
      boxShadow: 'none',
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Panel });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Panel.jsx", error: String((e && e.message) || e) }); }

// components/core/ThresholdSlider.jsx
try { (() => {
function ThresholdSlider({
  value = 0.45,
  min = 0,
  max = 1,
  step = 0.01,
  label = 'Threshold',
  onChange,
  rtl = false
}) {
  const pct = (value - min) / (max - min) * 100;
  const fill = rtl ? `linear-gradient(to left, var(--color-primary) 0 ${pct}%, var(--border-hairline) ${pct}% 100%)` : `linear-gradient(to right, var(--color-primary) 0 ${pct}%, var(--border-hairline) ${pct}% 100%)`;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-sm)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      gap: 'var(--space-md)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--type-label-size)',
      lineHeight: 'var(--type-label-lh)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      color: 'var(--text-body)'
    }
  }, label), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-numeric-md-size)',
      lineHeight: 'var(--type-numeric-md-lh)',
      fontFeatureSettings: 'var(--font-feature-tabular)',
      color: 'var(--text-body)'
    }
  }, Number(value).toFixed(2))), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: min,
    max: max,
    step: step,
    value: value,
    onChange: e => onChange && onChange(parseFloat(e.target.value)),
    style: {
      WebkitAppearance: 'none',
      appearance: 'none',
      width: '100%',
      height: '6px',
      background: fill,
      borderRadius: 'var(--radius-none)',
      outline: 'none',
      cursor: 'pointer',
      direction: rtl ? 'rtl' : 'ltr'
    }
  }));
}
Object.assign(__ds_scope, { ThresholdSlider });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/ThresholdSlider.jsx", error: String((e && e.message) || e) }); }

// components/core/Tooltip.jsx
try { (() => {
function Tooltip({
  label,
  placement = 'top',
  children
}) {
  const [open, setOpen] = React.useState(false);
  const pos = placement === 'bottom' ? {
    top: 'calc(100% + var(--space-xs))'
  } : {
    bottom: 'calc(100% + var(--space-xs))'
  };
  return /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'relative',
      display: 'inline-flex'
    },
    onMouseEnter: () => setOpen(true),
    onMouseLeave: () => setOpen(false),
    onFocus: () => setOpen(true),
    onBlur: () => setOpen(false)
  }, children, open && /*#__PURE__*/React.createElement("span", {
    role: "tooltip",
    style: {
      position: 'absolute',
      left: '50%',
      transform: 'translateX(-50%)',
      ...pos,
      background: 'var(--slate-deep)',
      color: 'var(--text-on-primary)',
      borderRadius: 'var(--radius-sm)',
      padding: 'var(--pad-chip)',
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-caption-size)',
      lineHeight: 'var(--type-caption-lh)',
      whiteSpace: 'nowrap',
      boxShadow: 'none',
      zIndex: 10
    }
  }, label));
}
Object.assign(__ds_scope, { Tooltip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Tooltip.jsx", error: String((e && e.message) || e) }); }

// components/data/Price.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Price({
  value,
  currency = 'SAR',
  size = 'md',
  style,
  ...rest
}) {
  const lg = size === 'lg';
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: lg ? 'var(--weight-bold)' : 'var(--weight-regular)',
      fontSize: lg ? 'var(--type-numeric-lg-size)' : 'var(--type-numeric-md-size)',
      lineHeight: lg ? 'var(--type-numeric-lg-lh)' : 'var(--type-numeric-md-lh)',
      fontFeatureSettings: 'var(--font-feature-tabular)',
      fontVariantNumeric: 'tabular-nums',
      color: 'var(--text-body)',
      display: 'inline-flex',
      alignItems: 'baseline',
      gap: '6px',
      transition: 'var(--transition-readout)',
      ...style
    }
  }, rest), currency && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.7em',
      letterSpacing: '0.06em'
    }
  }, currency), /*#__PURE__*/React.createElement("span", null, typeof value === 'number' ? value.toFixed(2) : value));
}
Object.assign(__ds_scope, { Price });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data/Price.jsx", error: String((e && e.message) || e) }); }

// components/detection/BoundingBox.jsx
try { (() => {
function BoundingBox({
  label,
  confidence,
  x,
  y,
  width,
  height,
  lowConfidence = false
}) {
  // Keep the label chip inside the plate: flip to the inner side near the right
  // edge, and drop it inside the box when there is no room above.
  const nearRight = x + width > 70;
  const nearTop = y < 8;
  const horizontal = nearRight ? {
    right: 0,
    left: 'auto'
  } : {
    left: 0,
    right: 'auto'
  };
  const vertical = nearTop ? {
    top: 0
  } : {
    top: '-2px',
    transform: 'translateY(-100%)'
  };
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: `${x}%`,
      top: `${y}%`,
      width: `${width}%`,
      height: `${height}%`,
      border: 'var(--border-width-box) solid var(--detection-box)',
      borderRadius: 'var(--radius-none)',
      opacity: lowConfidence ? 0.55 : 1,
      transition: 'var(--transition-readout)',
      boxShadow: 'none'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'absolute',
      ...horizontal,
      ...vertical,
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      background: 'var(--detection-box)',
      color: 'var(--text-on-hero)',
      borderRadius: 'var(--radius-none)',
      padding: '4px 6px',
      fontFamily: 'var(--font-mono)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--type-label-size)',
      lineHeight: 'var(--type-label-lh)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      fontFeatureSettings: 'var(--font-feature-tabular)',
      whiteSpace: 'nowrap'
    }
  }, label, confidence != null && /*#__PURE__*/React.createElement("span", null, Number(confidence).toFixed(2))));
}
Object.assign(__ds_scope, { BoundingBox });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/detection/BoundingBox.jsx", error: String((e && e.message) || e) }); }

// components/detection/ConfidenceBadge.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ConfidenceBadge({
  label,
  confidence,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 'var(--space-sm)',
      background: 'var(--surface-warm)',
      color: 'var(--text-body)',
      borderRadius: 'var(--radius-sm)',
      padding: 'var(--pad-chip)',
      fontFamily: 'var(--font-mono)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--type-label-size)',
      lineHeight: 'var(--type-label-lh)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      fontFeatureSettings: 'var(--font-feature-tabular)',
      boxShadow: 'none',
      transition: 'var(--transition-readout)',
      ...style
    }
  }, rest), label && /*#__PURE__*/React.createElement("span", null, label), /*#__PURE__*/React.createElement("span", null, Number(confidence).toFixed(2)));
}
Object.assign(__ds_scope, { ConfidenceBadge });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/detection/ConfidenceBadge.jsx", error: String((e && e.message) || e) }); }

// components/detection/DetectionChip.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function DetectionChip({
  label,
  count,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 'var(--space-sm)',
      background: 'var(--color-primary-subtle)',
      color: 'var(--text-chip)',
      borderRadius: 'var(--radius-sm)',
      padding: 'var(--pad-chip)',
      fontFamily: 'var(--font-mono)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--type-label-size)',
      lineHeight: 'var(--type-label-lh)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      fontFeatureSettings: 'var(--font-feature-tabular)',
      boxShadow: 'none',
      transition: 'var(--transition-readout)',
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", null, label), count != null && /*#__PURE__*/React.createElement("span", {
    style: {
      opacity: 0.75
    }
  }, "\xD7", count));
}
Object.assign(__ds_scope, { DetectionChip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/detection/DetectionChip.jsx", error: String((e && e.message) || e) }); }

// components/layout/HeaderBand.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function HeaderBand({
  brand = 'Vision Checkout',
  children,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("header", _extends({
    style: {
      background: 'var(--surface-band)',
      color: 'var(--text-on-band)',
      borderRadius: 'var(--radius-none)',
      padding: 'var(--pad-band) var(--page-margin)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 'var(--space-md)',
      fontFamily: 'var(--font-mono)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--type-label-size)',
      lineHeight: 'var(--type-label-lh)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      boxShadow: 'none',
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", null, brand), children && /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 'var(--space-md)'
    }
  }, children));
}
Object.assign(__ds_scope, { HeaderBand });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/layout/HeaderBand.jsx", error: String((e && e.message) || e) }); }

// components/layout/SectionHeading.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const levels = {
  hero: {
    fontFamily: 'var(--font-display)',
    fontSize: 'var(--type-display-hero-size)',
    lineHeight: 'var(--type-display-hero-lh)',
    letterSpacing: 'var(--type-display-hero-ls)',
    fontWeight: 400
  },
  section: {
    fontFamily: 'var(--font-display)',
    fontSize: 'var(--type-display-size)',
    lineHeight: 'var(--type-display-lh)',
    letterSpacing: 'var(--type-display-ls)',
    fontWeight: 400
  },
  sub: {
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--type-headline-lg-size)',
    lineHeight: 'var(--type-headline-lg-lh)',
    letterSpacing: 'var(--type-headline-lg-ls)',
    fontWeight: 700
  },
  label: {
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--type-headline-md-size)',
    lineHeight: 'var(--type-headline-md-lh)',
    fontWeight: 700
  }
};
function SectionHeading({
  level = 'section',
  as,
  lang,
  children,
  style,
  ...rest
}) {
  const arabic = lang === 'ar';
  const v = levels[level] || levels.section;
  const Tag = as || (level === 'hero' ? 'h1' : level === 'section' ? 'h2' : level === 'sub' ? 'h3' : 'h4');
  const arabicOverride = arabic && (level === 'hero' || level === 'section') ? {
    fontFamily: 'var(--font-arabic)',
    fontWeight: 700,
    letterSpacing: 'normal',
    fontSize: level === 'hero' ? '60px' : '38px',
    lineHeight: 1.25
  } : arabic ? {
    fontFamily: 'var(--font-arabic)',
    lineHeight: 'var(--type-body-ar-lh)'
  } : null;
  return /*#__PURE__*/React.createElement(Tag, _extends({
    dir: arabic ? 'rtl' : undefined,
    style: {
      margin: 0,
      color: 'var(--text-heading)',
      textWrap: 'pretty',
      ...v,
      ...arabicOverride,
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { SectionHeading });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/layout/SectionHeading.jsx", error: String((e && e.message) || e) }); }

// ui_kits/landing/CartPlate.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const {
  BoundingBox
} = window.VisionCheckoutDesignSystem_a34a69;

// Placeholder plate standing in for the cart photograph. No imagery was supplied.
function CartPlate({
  boxes
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      width: '100%',
      height: '100%',
      background: 'var(--beige)',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 11,
      letterSpacing: '.1em',
      textTransform: 'uppercase',
      color: 'var(--text-muted)'
    }
  }, "Cart photograph"), boxes.map(b => /*#__PURE__*/React.createElement(BoundingBox, _extends({
    key: b.label
  }, b))));
}
window.CartPlate = CartPlate;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/landing/CartPlate.jsx", error: String((e && e.message) || e) }); }

// ui_kits/landing/Hero.jsx
try { (() => {
function Hero() {
  // Tags sit on the illustration's own warm ground, so they take the dark fill —
  // a cream pill would vanish into the artwork.
  const tag = {
    position: 'absolute',
    background: 'var(--forest-deep)',
    color: 'var(--cream)',
    borderRadius: 'var(--radius-pill)',
    padding: '10px 18px',
    fontSize: 14,
    fontWeight: 500,
    display: 'flex',
    gap: 10,
    alignItems: 'center',
    whiteSpace: 'nowrap',
    boxShadow: '0 0 0 1px rgba(255,244,239,.22)'
  };
  const num = {
    color: 'var(--sage)',
    fontWeight: 600,
    fontVariantNumeric: 'tabular-nums'
  };
  // Detection outlines drawn over the artwork, in the model's own visual language.
  // Positioned as percentages of the image frame so they track the artwork at any width.
  const box = {
    position: 'absolute',
    border: '2px dotted var(--persimmon)',
    borderRadius: 23,
    pointerEvents: 'none'
  };
  return /*#__PURE__*/React.createElement("section", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))',
      gap: 'var(--gutter)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--cream)',
      borderRadius: 'var(--radius-xl)',
      padding: 'var(--pad-hero)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      gap: 'var(--space-lg)',
      position: 'relative',
      overflow: 'hidden',
      minHeight: 460
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      width: 300,
      height: 300,
      background: 'var(--mustard)',
      opacity: .45,
      borderRadius: '58% 42% 47% 53% / 45% 55% 45% 55%',
      right: -120,
      top: -100
    }
  }), /*#__PURE__*/React.createElement("h1", {
    style: {
      fontSize: 'clamp(38px, 5.2vw, var(--type-display-hero-size))',
      lineHeight: 'var(--type-display-hero-lh)',
      letterSpacing: 'var(--type-display-hero-ls)',
      position: 'relative',
      textWrap: 'balance'
    }
  }, "Picture the", /*#__PURE__*/React.createElement("br", null), "cart", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      color: 'var(--persimmon)'
    }
  }, "We do the rest")), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: 0,
      maxWidth: '33ch',
      color: 'var(--text-muted)',
      position: 'relative'
    }
  }, "Vision Checkout reads the whole cart at once. No scanning one item at a time, no queue, no unpacking at the till."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 'var(--space-sm)',
      flexWrap: 'wrap',
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "../scanner/index.html",
    style: {
      background: 'var(--forest)',
      color: 'var(--text-on-hero)',
      borderRadius: 'var(--radius-pill)',
      padding: '18px 32px',
      fontWeight: 500,
      fontSize: 17,
      textDecoration: 'none'
    }
  }, "Try a scan"), /*#__PURE__*/React.createElement("a", {
    href: "#how",
    style: {
      color: 'var(--text-body)',
      borderRadius: 'var(--radius-pill)',
      padding: '18px 26px',
      fontWeight: 500,
      fontSize: 17,
      textDecoration: 'none',
      border: '1px solid var(--border-hairline)'
    }
  }, "See how it works"))), /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--forest)',
      borderRadius: 'var(--radius-xl)',
      minHeight: 460,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 'var(--space-lg)',
      padding: 'var(--space-xl)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width: '100%',
      maxWidth: 380,
      aspectRatio: '1 / 1',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden',
      background: 'var(--persimmon-wash)',
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: "../../assets/illustration-cart.png",
    alt: "Illustration of a shopping cart filled with groceries",
    style: {
      width: '100%',
      height: '100%',
      objectFit: 'cover',
      display: 'block'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      ...box,
      left: '14%',
      top: '23%',
      width: '17%',
      height: '18%'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      ...box,
      left: '56%',
      top: '12%',
      width: '11%',
      height: '22%'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      ...box,
      left: '35%',
      top: '40%',
      width: '14%',
      height: '23%'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      ...box,
      left: '33%',
      top: '19%',
      width: '14%',
      height: '23%'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      ...box,
      left: '46%',
      top: '47%',
      width: '11%',
      height: '14%'
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      ...tag,
      left: '2%',
      top: '13%'
    }
  }, "Carrots ", /*#__PURE__*/React.createElement("span", {
    style: num
  }, "\xD72")), /*#__PURE__*/React.createElement("span", {
    style: {
      ...tag,
      left: '4%',
      top: '43%'
    }
  }, "Bread ", /*#__PURE__*/React.createElement("span", {
    style: num
  }, "\xD71")), /*#__PURE__*/React.createElement("span", {
    style: {
      ...tag,
      right: '4%',
      top: '36%'
    }
  }, "Olive oil ", /*#__PURE__*/React.createElement("span", {
    style: num
  }, "\xD71")), /*#__PURE__*/React.createElement("span", {
    style: {
      ...tag,
      left: '28%',
      top: '65%'
    }
  }, "Milk ", /*#__PURE__*/React.createElement("span", {
    style: num
  }, "\xD71")), /*#__PURE__*/React.createElement("span", {
    style: {
      ...tag,
      right: '4%',
      top: '63%'
    }
  }, "Jam ", /*#__PURE__*/React.createElement("span", {
    style: num
  }, "\xD71"))), /*#__PURE__*/React.createElement("span", {
    style: {
      background: 'var(--persimmon)',
      color: 'var(--cream)',
      borderRadius: 'var(--radius-pill)',
      padding: '12px 22px',
      fontSize: 15,
      fontWeight: 500,
      display: 'flex',
      gap: 12,
      alignItems: 'center'
    }
  }, "Total ", /*#__PURE__*/React.createElement("span", {
    style: {
      ...num,
      color: 'var(--cream)'
    }
  }, "SAR 84.20"))));
}
window.Hero = Hero;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/landing/Hero.jsx", error: String((e && e.message) || e) }); }

// ui_kits/landing/LandingPoster.jsx
try { (() => {
const {
  HeaderBand,
  SectionHeading,
  Button
} = window.VisionCheckoutDesignSystem_a34a69;
const COPY = {
  en: {
    band: 'Vision Checkout',
    toggle: 'العربية',
    headline: ['Put the cart down.', "We'll ring it up."],
    sub: 'One photograph. Every product, priced.',
    cta: 'Start scanning',
    boxes: [{
      label: 'Milk',
      confidence: 0.94,
      x: 6,
      y: 26,
      width: 24,
      height: 54
    }, {
      label: 'Bananas',
      confidence: 0.91,
      x: 36,
      y: 16,
      width: 28,
      height: 42
    }, {
      label: 'Tomatoes',
      confidence: 0.87,
      x: 62,
      y: 46,
      width: 30,
      height: 44
    }]
  },
  ar: {
    band: 'فيجن تشيك آوت',
    toggle: 'English',
    headline: ['ضع العربة.', 'ونحن نحسبها.'],
    sub: 'صورة واحدة. كل منتج، بسعره.',
    cta: 'ابدأ المسح',
    boxes: [{
      label: 'حليب',
      confidence: 0.94,
      x: 70,
      y: 26,
      width: 24,
      height: 54
    }, {
      label: 'موز',
      confidence: 0.91,
      x: 36,
      y: 16,
      width: 28,
      height: 42
    }, {
      label: 'طماطم',
      confidence: 0.87,
      x: 8,
      y: 46,
      width: 30,
      height: 44
    }]
  }
};
function LandingPoster({
  lang,
  onToggleLang,
  onStart
}) {
  const t = COPY[lang];
  const ar = lang === 'ar';
  return /*#__PURE__*/React.createElement("div", {
    dir: ar ? 'rtl' : 'ltr',
    style: {
      minHeight: '100%',
      background: 'var(--surface-page)',
      color: 'var(--text-body)'
    }
  }, /*#__PURE__*/React.createElement(HeaderBand, {
    brand: t.band
  }, /*#__PURE__*/React.createElement("button", {
    onClick: onToggleLang,
    style: {
      background: 'none',
      border: 'none',
      color: 'inherit',
      font: 'inherit',
      letterSpacing: 'inherit',
      textTransform: 'inherit',
      cursor: 'pointer',
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: 0
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "13",
    height: "13",
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.5",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "9"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M3 12h18M12 3c2.5 2.7 2.5 15.3 0 18M12 3c-2.5 2.7-2.5 15.3 0 18"
  })), t.toggle)), /*#__PURE__*/React.createElement("section", {
    style: {
      background: 'var(--surface-hero)',
      color: 'var(--text-on-hero)',
      padding: 'var(--pad-hero)',
      display: 'grid',
      gridTemplateColumns: 'minmax(0,1fr) minmax(0,1.05fr)',
      gap: 'var(--space-xl)',
      alignItems: 'center',
      overflow: 'hidden'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      paddingInlineEnd: 'var(--space-xl)'
    }
  }, /*#__PURE__*/React.createElement(SectionHeading, {
    level: "hero",
    lang: lang,
    style: {
      textWrap: 'balance'
    }
  }, t.headline[0], /*#__PURE__*/React.createElement("br", null), t.headline[1]), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: '32px 0 0',
      maxWidth: '46ch',
      fontFamily: ar ? 'var(--font-arabic)' : 'var(--font-sans)',
      fontSize: ar ? 'var(--type-body-ar-size)' : 'var(--type-body-lg-size)',
      lineHeight: ar ? 'var(--type-body-ar-lh)' : 'var(--type-body-lg-lh)'
    }
  }, t.sub), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 'var(--space-lg)'
    }
  }, /*#__PURE__*/React.createElement(Button, {
    onClick: onStart
  }, t.cta))), /*#__PURE__*/React.createElement("div", {
    style: {
      height: 420,
      marginInlineEnd: -160,
      marginBlock: -24
    }
  }, /*#__PURE__*/React.createElement(CartPlate, {
    boxes: t.boxes
  }))), /*#__PURE__*/React.createElement("footer", {
    style: {
      background: 'var(--surface-band)',
      color: 'var(--text-on-band)',
      padding: 'var(--pad-band) var(--page-margin)',
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 'var(--type-label-size)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase'
    }
  }, ar ? 'فيجن تشيك آوت — ألفا' : 'Vision Checkout — Alpha'));
}
window.LandingPoster = LandingPoster;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/landing/LandingPoster.jsx", error: String((e && e.message) || e) }); }

// ui_kits/landing/Nav.jsx
try { (() => {
function Nav() {
  const link = {
    color: 'var(--text-body)',
    textDecoration: 'none',
    fontSize: 15,
    fontWeight: 500,
    padding: '10px 18px',
    borderRadius: 'var(--radius-pill)',
    transition: 'background var(--transition-state)'
  };
  return /*#__PURE__*/React.createElement("nav", {
    style: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 'var(--space-lg)',
      padding: 'var(--space-lg) 0'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-display)',
      fontWeight: 800,
      fontSize: 23,
      letterSpacing: '-.035em',
      color: 'var(--text-heading)'
    }
  }, "Vision Checkout"), /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 4
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "#how",
    style: link
  }, "How it works"), /*#__PURE__*/React.createElement("a", {
    href: "#trolley",
    style: link
  }, "What it reads"), /*#__PURE__*/React.createElement("a", {
    href: "../scanner/index.html",
    style: {
      ...link,
      background: 'var(--cream)'
    }
  }, "Open the scanner")));
}
window.Nav = Nav;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/landing/Nav.jsx", error: String((e && e.message) || e) }); }

// ui_kits/landing/Sections.jsx
try { (() => {
const STEPS = [{
  n: '01',
  t: 'Fill the cart',
  d: 'Shop the way you already do. Nothing needs to be arranged, faced up or unpacked.'
}, {
  n: '02',
  t: 'Take one photo',
  d: 'At the till, or from your phone. One frame is enough for the whole cart.'
}, {
  n: '03',
  t: 'Pay and go',
  d: 'Every product is found, counted and priced. Check the list, tap pay, walk out.'
}];
function HowItWorks() {
  return /*#__PURE__*/React.createElement("section", {
    id: "how",
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-xl)'
    }
  }, /*#__PURE__*/React.createElement("h2", {
    style: {
      fontSize: 'clamp(30px, 3.6vw, var(--type-display-size))',
      lineHeight: 'var(--type-display-lh)',
      maxWidth: '18ch'
    }
  }, "Three steps, and one of them is shopping."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))',
      gap: 'var(--gutter)'
    }
  }, STEPS.map(s => /*#__PURE__*/React.createElement("div", {
    key: s.n,
    style: {
      background: 'var(--cream)',
      borderRadius: 'var(--radius-lg)',
      padding: 'var(--pad-card)',
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-sm)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      width: 44,
      height: 44,
      borderRadius: 'var(--radius-pill)',
      background: 'var(--persimmon-wash)',
      color: 'var(--persimmon-deep)',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: 600,
      fontVariantNumeric: 'tabular-nums',
      marginBottom: 'var(--space-sm)'
    }
  }, s.n), /*#__PURE__*/React.createElement("h3", {
    style: {
      fontSize: 'var(--type-headline-lg-size)',
      lineHeight: 'var(--type-headline-lg-lh)'
    }
  }, s.t), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: 0,
      fontSize: 'var(--type-body-md-size)',
      lineHeight: 'var(--type-body-md-lh)',
      color: 'var(--text-muted)'
    }
  }, s.d)))));
}
function Trolley() {
  const chip = {
    background: 'var(--persimmon-wash)',
    color: 'var(--persimmon-deep)',
    borderRadius: 'var(--radius-pill)',
    padding: '10px 18px',
    fontSize: 14,
    fontWeight: 500
  };
  return /*#__PURE__*/React.createElement("section", {
    id: "trolley",
    style: {
      background: 'var(--mustard-wash)',
      borderRadius: 'var(--radius-xl)',
      padding: 'var(--pad-hero)',
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))',
      gap: 'var(--space-xl)',
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-lg)'
    }
  }, /*#__PURE__*/React.createElement("h2", {
    style: {
      fontSize: 'clamp(30px, 3.6vw, var(--type-display-size))',
      lineHeight: 'var(--type-display-lh)',
      maxWidth: '16ch'
    }
  }, "Loose produce included."), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: 0,
      maxWidth: '34ch',
      color: 'var(--text-muted)'
    }
  }, "Barcodes are the easy part. The model reads the things that usually hold up a queue \u2014 bagged fruit, bunched herbs, bakery items, anything without a label."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 'var(--space-sm)'
    }
  }, ['Lettuce', 'Broccoli', 'Aubergine', 'Carrot', 'Tomatoes', 'Leek'].map(c => /*#__PURE__*/React.createElement("span", {
    key: c,
    style: chip
  }, c)))), /*#__PURE__*/React.createElement("div", {
    style: {
      width: '100%',
      aspectRatio: '5 / 4',
      maxHeight: 420,
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden',
      background: 'var(--cream)'
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: "../../assets/illustration-trolley-produce.png",
    alt: "Flat illustration of fresh produce around a shopping cart",
    style: {
      width: '100%',
      height: '100%',
      objectFit: 'cover',
      objectPosition: '72% 42%',
      display: 'block'
    }
  })));
}
function Closer() {
  return /*#__PURE__*/React.createElement("section", {
    style: {
      background: 'var(--forest)',
      color: 'var(--text-on-hero)',
      borderRadius: 'var(--radius-xl)',
      padding: 'var(--pad-hero)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 'var(--space-lg)',
      textAlign: 'center'
    }
  }, /*#__PURE__*/React.createElement("h2", {
    style: {
      fontSize: 'clamp(30px, 3.6vw, var(--type-display-size))',
      lineHeight: 'var(--type-display-lh)',
      maxWidth: '20ch',
      color: 'var(--text-on-hero)'
    }
  }, "Put the cart down. We will ring it up."), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: 0,
      maxWidth: '42ch',
      color: 'var(--text-on-hero-muted)'
    }
  }, "Point a camera at a cart and see what comes back."), /*#__PURE__*/React.createElement("a", {
    href: "../scanner/index.html",
    style: {
      background: 'var(--persimmon)',
      color: 'var(--cream)',
      borderRadius: 'var(--radius-pill)',
      padding: '18px 34px',
      fontWeight: 500,
      fontSize: 17,
      textDecoration: 'none'
    }
  }, "Scan a cart"));
}
function Footer() {
  return /*#__PURE__*/React.createElement("footer", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 'var(--space-md)',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 'var(--space-xl) 0',
      color: 'var(--text-muted)',
      fontSize: 'var(--type-caption-size)'
    }
  }, /*#__PURE__*/React.createElement("span", null, "Vision Checkout \u2014 alpha"), /*#__PURE__*/React.createElement("span", null, "Illustrations via ", /*#__PURE__*/React.createElement("a", {
    href: "https://www.vecteezy.com",
    style: {
      color: 'inherit'
    }
  }, "Vecteezy"), " and iconicbestiary / ", /*#__PURE__*/React.createElement("a", {
    href: "https://www.freepik.com",
    style: {
      color: 'inherit'
    }
  }, "Freepik")));
}
Object.assign(window, {
  HowItWorks,
  Trolley,
  Closer,
  Footer
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/landing/Sections.jsx", error: String((e && e.message) || e) }); }

// ui_kits/scanner/InputColumn.jsx
try { (() => {
const {
  Panel,
  Button,
  Input,
  ThresholdSlider
} = window.VisionCheckoutDesignSystem_a34a69;
function DropTarget({
  sampleName,
  onPick
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--surface-card)',
      border: '1px solid var(--border-hairline)',
      borderRadius: 'var(--radius-lg)',
      padding: 'var(--space-lg)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 'var(--space-sm)',
      textAlign: 'center'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "22",
    height: "22",
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "var(--text-muted)",
    strokeWidth: "1.5",
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M12 16V4M12 4 7 9M12 4l5 5"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 'var(--type-body-md-size)',
      lineHeight: 'var(--type-body-md-lh)'
    }
  }, "Drop an image, or pick a sample."), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 'var(--type-label-size)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      color: 'var(--text-muted)'
    }
  }, sampleName), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 'var(--space-sm)'
    }
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    onClick: onPick
  }, "Shuffle sample")));
}
function InputColumn({
  sample,
  threshold,
  onThreshold,
  onShuffle,
  onScan,
  scanning,
  hasRun,
  onReset,
  url,
  onUrl,
  urlError
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-md)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 'var(--type-label-size)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      color: 'var(--text-muted)'
    }
  }, "Input"), /*#__PURE__*/React.createElement(DropTarget, {
    sampleName: sample.name,
    onPick: onShuffle
  }), /*#__PURE__*/React.createElement(Input, {
    label: "Image URL",
    placeholder: "https://",
    value: url,
    onChange: e => onUrl(e.target.value),
    error: !!urlError,
    hint: urlError || undefined
  }), /*#__PURE__*/React.createElement(Panel, {
    tone: "cool",
    style: {
      padding: 'var(--space-md)'
    }
  }, /*#__PURE__*/React.createElement(ThresholdSlider, {
    value: threshold,
    onChange: onThreshold,
    label: "Threshold"
  })), /*#__PURE__*/React.createElement(Button, {
    onClick: onScan,
    disabled: scanning,
    fullWidth: true
  }, scanning ? 'Scanning…' : 'Scan cart'), hasRun && /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    onClick: onReset,
    fullWidth: true
  }, "Reset"));
}
window.InputColumn = InputColumn;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/scanner/InputColumn.jsx", error: String((e && e.message) || e) }); }

// ui_kits/scanner/ResultsColumn.jsx
try { (() => {
const {
  SectionHeading,
  DetectionChip,
  ConfidenceBadge,
  BoundingBox,
  Card,
  Divider,
  Price
} = window.VisionCheckoutDesignSystem_a34a69;
function AnnotatedPlate({
  detections,
  scanning
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      aspectRatio: '16 / 9',
      background: 'var(--beige)',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 'var(--type-label-size)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase',
      color: 'var(--text-muted)'
    }
  }, "Cart image"), scanning && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      insetInline: 0,
      height: 3,
      background: 'var(--peach)',
      animation: 'vc-sweep 1.1s ease-out infinite'
    }
  }), !scanning && detections.map(d => /*#__PURE__*/React.createElement(BoundingBox, {
    key: d.label,
    label: d.category,
    confidence: d.confidence,
    x: d.x,
    y: d.y,
    width: d.width,
    height: d.height,
    lowConfidence: d.low
  })));
}
function Receipt({
  rows,
  total
}) {
  return /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'baseline',
      justifyContent: 'space-between'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 'var(--type-label-size)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase'
    }
  }, "Receipt"), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-caption-size)',
      color: 'var(--text-muted)'
    }
  }, rows.length, " lines")), /*#__PURE__*/React.createElement("div", {
    style: {
      height: 'var(--space-md)'
    }
  }), rows.map(r => /*#__PURE__*/React.createElement(React.Fragment, {
    key: r.label
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      gap: 'var(--space-md)',
      padding: '4px 0'
    }
  }, /*#__PURE__*/React.createElement("span", null, r.label, r.qty > 1 ? ` ×${r.qty}` : ''), /*#__PURE__*/React.createElement(Price, {
    value: r.qty * r.price,
    currency: ""
  })), /*#__PURE__*/React.createElement(Divider, null))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      paddingTop: 'var(--space-md)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      fontSize: 'var(--type-label-size)',
      letterSpacing: 'var(--type-label-ls)',
      textTransform: 'uppercase'
    }
  }, "Total"), /*#__PURE__*/React.createElement(Price, {
    value: total,
    size: "lg"
  })));
}
function ResultsColumn({
  detections,
  threshold,
  scanning,
  hasRun
}) {
  const above = detections.filter(d => d.confidence >= threshold);
  const below = detections.filter(d => d.confidence < threshold);
  const marked = detections.map(d => ({
    ...d,
    low: d.confidence < threshold
  }));
  const total = above.reduce((s, d) => s + d.qty * d.price, 0);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--space-lg)'
    }
  }, /*#__PURE__*/React.createElement(AnnotatedPlate, {
    detections: hasRun ? marked : [],
    scanning: scanning
  }), !hasRun ? /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 'var(--type-body-lg-size)',
      lineHeight: 'var(--type-body-lg-lh)',
      color: 'var(--text-muted)',
      maxWidth: 'var(--measure)'
    }
  }, "Nothing scanned yet. Pick a sample and scan the cart.") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(SectionHeading, {
    level: "sub"
  }, "Detections"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 'var(--space-sm)',
      marginTop: 'var(--space-md)'
    }
  }, above.map(d => /*#__PURE__*/React.createElement(DetectionChip, {
    key: d.label,
    label: d.category,
    count: d.qty
  })), below.map(d => /*#__PURE__*/React.createElement(ConfidenceBadge, {
    key: d.label,
    label: d.category,
    confidence: d.confidence
  }))), below.length > 0 && /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 'var(--space-sm)',
      fontSize: 'var(--type-caption-size)',
      lineHeight: 'var(--type-caption-lh)',
      color: 'var(--text-muted)'
    }
  }, below.length, " ", below.length === 1 ? 'item is' : 'items are', " below the threshold. Lower it to include ", below.length === 1 ? 'it' : 'them', ".")), /*#__PURE__*/React.createElement(Receipt, {
    rows: above,
    total: total
  })));
}
window.ResultsColumn = ResultsColumn;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/scanner/ResultsColumn.jsx", error: String((e && e.message) || e) }); }

// ui_kits/scanner/samples.js
try { (() => {
window.SAMPLES = [{
  id: 'weekly',
  name: 'Weekly shop',
  detections: [{
    label: 'Milk 1L',
    category: 'Milk',
    confidence: 0.94,
    qty: 1,
    price: 8.5,
    x: 6,
    y: 24,
    width: 22,
    height: 52
  }, {
    label: 'Bananas',
    category: 'Bananas',
    confidence: 0.91,
    qty: 3,
    price: 4.2,
    x: 32,
    y: 14,
    width: 26,
    height: 40
  }, {
    label: 'Tomatoes',
    category: 'Tomatoes',
    confidence: 0.86,
    qty: 6,
    price: 1.03,
    x: 60,
    y: 42,
    width: 28,
    height: 44
  }, {
    label: 'Bell pepper',
    category: 'Bell pepper',
    confidence: 0.31,
    qty: 1,
    price: 3.4,
    x: 76,
    y: 12,
    width: 20,
    height: 26
  }]
}, {
  id: 'breakfast',
  name: 'Breakfast run',
  detections: [{
    label: 'Eggs ×12',
    category: 'Eggs',
    confidence: 0.97,
    qty: 1,
    price: 14.0,
    x: 10,
    y: 30,
    width: 30,
    height: 40
  }, {
    label: 'Orange juice',
    category: 'Orange juice',
    confidence: 0.89,
    qty: 2,
    price: 9.75,
    x: 46,
    y: 18,
    width: 22,
    height: 56
  }, {
    label: 'Bread',
    category: 'Bread',
    confidence: 0.42,
    qty: 1,
    price: 5.5,
    x: 70,
    y: 46,
    width: 26,
    height: 36
  }]
}, {
  id: 'produce',
  name: 'Produce only',
  detections: [{
    label: 'Cucumbers',
    category: 'Cucumbers',
    confidence: 0.93,
    qty: 4,
    price: 1.75,
    x: 8,
    y: 34,
    width: 26,
    height: 42
  }, {
    label: 'Lemons',
    category: 'Lemons',
    confidence: 0.9,
    qty: 5,
    price: 0.95,
    x: 38,
    y: 22,
    width: 24,
    height: 36
  }, {
    label: 'Onions',
    category: 'Onions',
    confidence: 0.79,
    qty: 3,
    price: 2.1,
    x: 64,
    y: 40,
    width: 28,
    height: 44
  }, {
    label: 'Aubergine',
    category: 'Aubergine',
    confidence: 0.27,
    qty: 1,
    price: 6.0,
    x: 74,
    y: 10,
    width: 20,
    height: 24
  }]
}];
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/scanner/samples.js", error: String((e && e.message) || e) }); }

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.Divider = __ds_scope.Divider;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Panel = __ds_scope.Panel;

__ds_ns.ThresholdSlider = __ds_scope.ThresholdSlider;

__ds_ns.Tooltip = __ds_scope.Tooltip;

__ds_ns.Price = __ds_scope.Price;

__ds_ns.BoundingBox = __ds_scope.BoundingBox;

__ds_ns.ConfidenceBadge = __ds_scope.ConfidenceBadge;

__ds_ns.DetectionChip = __ds_scope.DetectionChip;

__ds_ns.HeaderBand = __ds_scope.HeaderBand;

__ds_ns.SectionHeading = __ds_scope.SectionHeading;

})();

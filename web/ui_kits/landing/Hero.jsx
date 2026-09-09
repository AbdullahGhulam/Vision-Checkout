function Hero() {
  // Tags sit on the illustration's own warm ground, so they take the dark fill —
  // a cream pill would vanish into the artwork.
  const tag = { position: 'absolute', background: 'var(--forest-deep)', color: 'var(--cream)', borderRadius: 'var(--radius-pill)', padding: '10px 18px', fontSize: 14, fontWeight: 500, display: 'flex', gap: 10, alignItems: 'center', whiteSpace: 'nowrap', boxShadow: '0 0 0 1px rgba(255,244,239,.22)' };
  const num = { color: 'var(--sage)', fontWeight: 600, fontVariantNumeric: 'tabular-nums' };
  // Detection outlines drawn over the artwork, in the model's own visual language.
  // Positioned as percentages of the image frame so they track the artwork at any width.
  const box = { position: 'absolute', border: '2px dotted var(--persimmon)', borderRadius: 23, pointerEvents: 'none' };
  return (
    <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 'var(--gutter)' }}>
      <div style={{ background: 'var(--cream)', borderRadius: 'var(--radius-xl)', padding: 'var(--pad-hero)', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 'var(--space-lg)', position: 'relative', overflow: 'hidden', minHeight: 460 }}>
        <div style={{ position: 'absolute', width: 300, height: 300, background: 'var(--mustard)', opacity: .45, borderRadius: '58% 42% 47% 53% / 45% 55% 45% 55%', right: -120, top: -100 }} />
        <h1 style={{ fontSize: 'clamp(38px, 5.2vw, var(--type-display-hero-size))', lineHeight: 'var(--type-display-hero-lh)', letterSpacing: 'var(--type-display-hero-ls)', position: 'relative', textWrap: 'balance' }}>
          Picture the<br />cart<br /><span style={{ color: 'var(--persimmon)' }}>We do the rest</span>
        </h1>
        <p style={{ margin: 0, maxWidth: '33ch', color: 'var(--text-muted)', position: 'relative' }}>
          Vision Checkout reads the whole cart at once. No scanning one item at a time, no queue, no unpacking at the till.
        </p>
        <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap', position: 'relative' }}>
          <a href="../scanner/index.html" style={{ background: 'var(--forest)', color: 'var(--text-on-hero)', borderRadius: 'var(--radius-pill)', padding: '18px 32px', fontWeight: 500, fontSize: 17, textDecoration: 'none' }}>Try a scan</a>
          <a href="#how" style={{ color: 'var(--text-body)', borderRadius: 'var(--radius-pill)', padding: '18px 26px', fontWeight: 500, fontSize: 17, textDecoration: 'none', border: '1px solid var(--border-hairline)' }}>See how it works</a>
        </div>
      </div>

      <div style={{ background: 'var(--forest)', borderRadius: 'var(--radius-xl)', minHeight: 460, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 'var(--space-lg)', padding: 'var(--space-xl)' }}>
        {/* Every detection lives inside the image frame in percentages, so a label can
            never drift off the product it names. */}
        <div style={{ width: '100%', maxWidth: 380, aspectRatio: '1 / 1', borderRadius: 'var(--radius-lg)', overflow: 'hidden', background: 'var(--persimmon-wash)', position: 'relative' }}>
          <img src="../../assets/illustration-cart.png" alt="Illustration of a shopping cart filled with groceries" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
          <div style={{ ...box, left: '14%', top: '23%', width: '17%', height: '18%' }} />
          <div style={{ ...box, left: '56%', top: '12%', width: '11%', height: '22%' }} />
          <div style={{ ...box, left: '35%', top: '40%', width: '14%', height: '23%' }} />
          <div style={{ ...box, left: '33%', top: '19%', width: '14%', height: '23%' }} />
          <div style={{ ...box, left: '46%', top: '47%', width: '11%', height: '14%' }} />
          <span style={{ ...tag, left: '2%', top: '13%' }}>Carrots <span style={num}>×2</span></span>
          <span style={{ ...tag, left: '4%', top: '43%' }}>Bread <span style={num}>×1</span></span>
          <span style={{ ...tag, right: '4%', top: '36%' }}>Olive oil <span style={num}>×1</span></span>
          <span style={{ ...tag, left: '28%', top: '65%' }}>Milk <span style={num}>×1</span></span>
          <span style={{ ...tag, right: '4%', top: '63%' }}>Jam <span style={num}>×1</span></span>
        </div>
        <span style={{ background: 'var(--persimmon)', color: 'var(--cream)', borderRadius: 'var(--radius-pill)', padding: '12px 22px', fontSize: 15, fontWeight: 500, display: 'flex', gap: 12, alignItems: 'center' }}>Total <span style={{ ...num, color: 'var(--cream)' }}>SAR 84.20</span></span>
      </div>
    </section>
  );
}
window.Hero = Hero;

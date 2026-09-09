const STEPS = [
  { n: '01', t: 'Fill the cart', d: 'Shop the way you already do. Nothing needs to be arranged, faced up or unpacked.' },
  { n: '02', t: 'Take one photo', d: 'At the till, or from your phone. One frame is enough for the whole cart.' },
  { n: '03', t: 'Pay and go', d: 'Every product is found, counted and priced. Check the list, tap pay, walk out.' },
];

function HowItWorks() {
  return (
    <section id="how" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
      <h2 style={{ fontSize: 'clamp(30px, 3.6vw, var(--type-display-size))', lineHeight: 'var(--type-display-lh)', maxWidth: '18ch' }}>Three steps, and one of them is shopping.</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 'var(--gutter)' }}>
        {STEPS.map((s) => (
          <div key={s.n} style={{ background: 'var(--cream)', borderRadius: 'var(--radius-lg)', padding: 'var(--pad-card)', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
            <span style={{ width: 44, height: 44, borderRadius: 'var(--radius-pill)', background: 'var(--persimmon-wash)', color: 'var(--persimmon-deep)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, fontVariantNumeric: 'tabular-nums', marginBottom: 'var(--space-sm)' }}>{s.n}</span>
            <h3 style={{ fontSize: 'var(--type-headline-lg-size)', lineHeight: 'var(--type-headline-lg-lh)' }}>{s.t}</h3>
            <p style={{ margin: 0, fontSize: 'var(--type-body-md-size)', lineHeight: 'var(--type-body-md-lh)', color: 'var(--text-muted)' }}>{s.d}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Trolley() {
  const chip = { background: 'var(--persimmon-wash)', color: 'var(--persimmon-deep)', borderRadius: 'var(--radius-pill)', padding: '10px 18px', fontSize: 14, fontWeight: 500 };
  return (
    <section id="trolley" style={{ background: 'var(--mustard-wash)', borderRadius: 'var(--radius-xl)', padding: 'var(--pad-hero)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 'var(--space-xl)', alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
        <h2 style={{ fontSize: 'clamp(30px, 3.6vw, var(--type-display-size))', lineHeight: 'var(--type-display-lh)', maxWidth: '16ch' }}>Loose produce included.</h2>
        <p style={{ margin: 0, maxWidth: '34ch', color: 'var(--text-muted)' }}>Barcodes are the easy part. The model reads the things that usually hold up a queue — bagged fruit, bunched herbs, bakery items, anything without a label.</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
          {['Lettuce', 'Broccoli', 'Aubergine', 'Carrot', 'Tomatoes', 'Leek'].map((c) => <span key={c} style={chip}>{c}</span>)}
        </div>
      </div>
      {/* The source composition is clipped at its own left edge, so it is framed as a
          deliberate crop on all four sides rather than shown as a cut-out object. */}
      <div style={{ width: '100%', aspectRatio: '5 / 4', maxHeight: 420, borderRadius: 'var(--radius-lg)', overflow: 'hidden', background: 'var(--cream)' }}>
        <img src="../../assets/illustration-trolley-produce.png" alt="Flat illustration of fresh produce around a shopping cart" style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: '72% 42%', display: 'block' }} />
      </div>
    </section>
  );
}

function Closer() {
  return (
    <section style={{ background: 'var(--forest)', color: 'var(--text-on-hero)', borderRadius: 'var(--radius-xl)', padding: 'var(--pad-hero)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-lg)', textAlign: 'center' }}>
      <h2 style={{ fontSize: 'clamp(30px, 3.6vw, var(--type-display-size))', lineHeight: 'var(--type-display-lh)', maxWidth: '20ch', color: 'var(--text-on-hero)' }}>Put the cart down. We will ring it up.</h2>
      <p style={{ margin: 0, maxWidth: '42ch', color: 'var(--text-on-hero-muted)' }}>Point a camera at a cart and see what comes back.</p>
      <a href="../scanner/index.html" style={{ background: 'var(--persimmon)', color: 'var(--cream)', borderRadius: 'var(--radius-pill)', padding: '18px 34px', fontWeight: 500, fontSize: 17, textDecoration: 'none' }}>Scan a cart</a>
    </section>
  );
}

function Footer() {
  return (
    <footer style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-md)', justifyContent: 'space-between', alignItems: 'center', padding: 'var(--space-xl) 0', color: 'var(--text-muted)', fontSize: 'var(--type-caption-size)' }}>
      <span>Vision Checkout — alpha</span>
      <span>Illustrations via <a href="https://www.vecteezy.com" style={{ color: 'inherit' }}>Vecteezy</a> and iconicbestiary / <a href="https://www.freepik.com" style={{ color: 'inherit' }}>Freepik</a></span>
    </footer>
  );
}
Object.assign(window, { HowItWorks, Trolley, Closer, Footer });

function Nav() {
  const link = { color: 'var(--text-body)', textDecoration: 'none', fontSize: 15, fontWeight: 500, padding: '10px 18px', borderRadius: 'var(--radius-pill)', transition: 'background var(--transition-state)' };
  return (
    <nav style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-lg)', padding: 'var(--space-lg) 0' }}>
      <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 23, letterSpacing: '-.035em', color: 'var(--text-heading)' }}>Vision Checkout</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <a href="#how" style={link}>How it works</a>
        <a href="#trolley" style={link}>What it reads</a>
        <a href="../scanner/index.html" style={{ ...link, background: 'var(--cream)' }}>Open the scanner</a>
      </span>
    </nav>
  );
}
window.Nav = Nav;

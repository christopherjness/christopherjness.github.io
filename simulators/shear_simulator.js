  // ─── Configuration ───────────────────────────────────────────────
  const CANVAS_SIZE = 500;
  const N_PARTICLES = 100;
  const R_MEAN = 1.0;         // mean particle radius (absolute units)
  const RHO_P = 1.0;          // 2-D particle area density
  const shearRate = 1.0;      // fixed shear rate γ̇

  // ─── Viscosity & timestep ─────────────────────────────────────────
  const eta0 = 100.0;                                    // solvent viscosity
  let dt    = 1e-4;                                      // timestep (adaptive)
  let stepsPerFrame = 20;                                // steps per animation frame
  // Pre-allocated force buffers — reused every step to avoid GC pressure
  const _fx     = new Float64Array(N_PARTICLES);
  const _fy     = new Float64Array(N_PARTICLES);
  const _torque = new Float64Array(N_PARTICLES);

  // ─── State ───────────────────────────────────────────────────────
  let particles = [];
  let contacts = [];          // populated each force evaluation
  let currentLeShift = 0;    // kept in sync with step() for rendering
  let phi = 0.65;
  let poly = 0.43;            // polydispersity: half-width / mean radius
  const stiffness = 2e6;
  const kt = stiffness * 2 / 7;  // tangential spring stiffness (Mindlin approximation)
  let strain = 0;
  let L = 1.0;   // box side length — computed from phi and radii
  let mu = 0.5;              // friction coefficient
  let contactMap = new Map(); // tangential spring state: (i*N_PARTICLES+j) → delta_t

  // ─── Canvas ──────────────────────────────────────────────────────
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  // ─── Helpers ─────────────────────────────────────────────────────
  function pbc(d) {
    const half = L / 2;
    while (d >  half) d -= L;
    while (d < -half) d += L;
    return d;
  }

  // ─── Initialise: place N particles, compute L from phi ───────────
  function noOverlap(x, y, r) {
    for (const p of particles) {
      const dx = pbc(x - p.x), dy = pbc(y - p.y);
      if (Math.hypot(dx, dy) < r + p.r + 1e-4) return false;
    }
    return true;
  }

  function initParticles() {
    strain = 0;
    contactMap = new Map();

    const r0s = Array.from({ length: N_PARTICLES }, () => Math.random() * 2 - 1);
    const radii = r0s.map(r0 => R_MEAN * (1 + r0 * poly));

    const totalArea = radii.reduce((s, r) => s + Math.PI * r * r, 0);
    L = Math.sqrt(totalArea / phi);

    particles = [];
    for (let i = 0; i < N_PARTICLES; i++) {
      const r = radii[i];
      let x, y, placed = false;
      for (let attempt = 0; attempt < 500; attempt++) {
        x = Math.random() * L;
        y = Math.random() * L;
        if (noOverlap(x, y, r)) { placed = true; break; }
      }
      if (!placed) { x = Math.random() * L; y = Math.random() * L; }
      const m = RHO_P * Math.PI * r * r;
      particles.push({ x, y, vx: 0, vy: 0, ax: 0, ay: 0, omega: 0, alpha: 0, theta: 0,
                        r, r0: r0s[i], mass: m, I: 0.5 * m * r * r });
    }
  }

  // ─── Shared: rescale particle positions to a new box side length ──
  function rescalePositions(newL) {
    const ratio = newL / L;
    for (const p of particles) {
      p.x = (p.x * ratio + newL) % newL;
      p.y = (p.y * ratio + newL) % newL;
      p.vx = 0; p.vy = 0; p.ax = 0; p.ay = 0;
      p.omega = 0; p.alpha = 0;
    }
    contactMap = new Map();
    L = newL;
  }

  // ─── Forces ──────────────────────────────────────────────────────
  function computeForces(fx, fy, torque, leShift) {
    const n = particles.length;
    const newContactMap = new Map();
    contacts = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = particles[j].x - particles[i].x;
        let dy = particles[j].y - particles[i].y;
        const ky = Math.round(dy / L);
        dy -= ky * L;
        dx -= ky * leShift;
        dx = pbc(dx);
        const dist = Math.hypot(dx, dy);
        const sigma = particles[i].r + particles[j].r;
        if (dist < sigma && dist > 1e-10) {
          const fn = stiffness * (sigma - dist);
          const nx = dx / dist, ny = dy / dist;

          // ── Normal force ──────────────────────────────────────────
          fx[i] -= fn * nx;  fy[i] -= fn * ny;
          fx[j] += fn * nx;  fy[j] += fn * ny;

          // ── Tangential friction (Cundall-Strack spring-slider) ────
          const tx = -ny, ty = nx;
          const pi = particles[i], pj = particles[j];
          const vs = (pj.vx - pi.vx - ky * shearRate * L) * tx + (pj.vy - pi.vy) * ty
                   - pi.omega * pi.r - pj.omega * pj.r;
          const key = i * n + j;
          let delta_t = (contactMap.has(key) ? contactMap.get(key) : 0) + vs * dt;
          let ft = kt * delta_t;
          const ft_max = mu * fn;
          if (ft >  ft_max) { ft =  ft_max; delta_t =  ft_max / kt; }
          if (ft < -ft_max) { ft = -ft_max; delta_t = -ft_max / kt; }
          newContactMap.set(key, delta_t);
          fx[i] += ft * tx;  fy[i] += ft * ty;
          fx[j] -= ft * tx;  fy[j] -= ft * ty;
          torque[i] += pi.r * ft;
          torque[j] += pj.r * ft;

          contacts.push({ i, j, fn, ft, nx, ny, dy });
        }
      }
    }
    contactMap = newContactMap;
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const dragI = 6 * Math.PI * eta0 * p.r;
      fx[i]      -= dragI * (p.vx - shearRate * p.y);
      fy[i]      -= dragI * p.vy;
      torque[i]  -= 8 * Math.PI * eta0 * p.r * p.r * p.r * (p.omega + shearRate / 2);
    }
  }

  // ─── Velocity-Verlet step ─────────────────────────────────────────
  function step() {
    const n = particles.length;
    if (n === 0) return;

    strain += shearRate * dt;
    const leShift = (strain % 1) * L;
    currentLeShift = leShift;
    const dt2 = dt * dt;

    for (let i = 0; i < n; i++) {
      const p = particles[i];
      p.x += p.vx * dt + 0.5 * p.ax * dt2;
      p.y += p.vy * dt + 0.5 * p.ay * dt2;
      p.theta += p.omega * dt + 0.5 * p.alpha * dt2;
      if (p.y >= L) { p.y -= L; p.x -= leShift; p.vx -= shearRate * L; }
      else if (p.y < 0) { p.y += L; p.x += leShift; p.vx += shearRate * L; }
      p.x = ((p.x % L) + L) % L;
    }

    _fx.fill(0); _fy.fill(0); _torque.fill(0);
    computeForces(_fx, _fy, _torque, leShift);

    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const ax_new    = _fx[i]     / p.mass;
      const ay_new    = _fy[i]     / p.mass;
      const alpha_new = _torque[i] / p.I;
      p.vx   += 0.5 * (p.ax    + ax_new)    * dt;
      p.vy   += 0.5 * (p.ay    + ay_new)    * dt;
      p.omega += 0.5 * (p.alpha + alpha_new) * dt;
      p.ax    = ax_new;
      p.ay    = ay_new;
      p.alpha = alpha_new;
    }
  }

  // ─── Rendering ────────────────────────────────────────────────────
  function draw() {
    const SCALE = CANVAS_SIZE / L;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // background grid
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 0.5;
    const nGrid = 10, gStep = CANVAS_SIZE / nGrid;
    for (let i = 0; i <= nGrid; i++) {
      ctx.beginPath(); ctx.moveTo(i * gStep, 0); ctx.lineTo(i * gStep, CANVAS_SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * gStep); ctx.lineTo(CANVAS_SIZE, i * gStep); ctx.stroke();
    }

    const shiftPx = ((currentLeShift % L) + L) % L * SCALE;

    for (const p of particles) {
      const px = p.x * SCALE;
      const py = CANVAS_SIZE - p.y * SCALE;
      const pr = p.r * SCALE;

      const draws = [[px, py]];
      if (py - pr < 0)             draws.push([px - shiftPx, py + CANVAS_SIZE]);
      if (py + pr > CANVAS_SIZE)   draws.push([px + shiftPx, py - CANVAS_SIZE]);

      const allDraws = [];
      for (const [gx, gy] of draws) {
        const nx = ((gx % CANVAS_SIZE) + CANVAS_SIZE) % CANVAS_SIZE;
        allDraws.push([nx, gy]);
        if (nx - pr < 0)             allDraws.push([nx + CANVAS_SIZE, gy]);
        if (nx + pr > CANVAS_SIZE)   allDraws.push([nx - CANVAS_SIZE, gy]);
      }

      const cosT = Math.cos(p.theta);
      const sinT = Math.sin(p.theta);

      for (const [gx, gy] of allDraws) {
        const gradient = ctx.createRadialGradient(
          gx - pr * 0.3, gy - pr * 0.3, pr * 0.1,
          gx, gy, pr * 1.1
        );
        const Lv = Math.max(20, Math.min(85, Math.round(78 - p.r0 * 32)));
        const baseColor = `hsl(0,0%,${Lv}%)`;
        const highlightColor = `hsl(0,0%,${Math.min(95, Lv + 15)}%)`;
        const shadowColor = `hsl(0,0%,${Math.max(5, Lv - 20)}%)`;
        gradient.addColorStop(0, highlightColor);
        gradient.addColorStop(0.5, baseColor);
        gradient.addColorStop(1, shadowColor);
        ctx.beginPath();
        ctx.arc(gx, gy, pr, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.7)';
        ctx.lineWidth = 0.8;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(gx - pr * cosT, gy + pr * sinT);
        ctx.lineTo(gx + pr * cosT, gy - pr * sinT);
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.lineWidth = 1.0;
        ctx.stroke();
      }
    }

    // contact force lines drawn on top
    if (contacts.length > 0) {
      const meanF = contacts.reduce((s, c) => s + c.fn, 0) / contacts.length;
      ctx.strokeStyle = 'hsla(0,100%,50%,0.9)';
      for (const c of contacts) {
        const pi = particles[c.i], pj = particles[c.j];
        let dx = pj.x - pi.x;
        let dy = pj.y - pi.y;
        const ny = Math.round(dy / L);
        dy -= ny * L;
        dx -= ny * currentLeShift;
        dx = pbc(dx);
        ctx.beginPath();
        ctx.moveTo(pi.x * SCALE,        CANVAS_SIZE - pi.y * SCALE);
        ctx.lineTo((pi.x + dx) * SCALE, CANVAS_SIZE - (pi.y + dy) * SCALE);
        ctx.lineWidth = Math.max(0.5, Math.min(c.fn / meanF * 2, 8));
        ctx.stroke();
      }
    }
  }

  // ─── Animation loop ───────────────────────────────────────────────
  function loop() {
    for (let s = 0; s < stepsPerFrame; s++) step();
    draw();
    requestAnimationFrame(loop);
  }

  // ─── Adaptive timestep ────────────────────────────────────────────
  function updateTimestep() {
    const rMin = R_MEAN * Math.max(0.1, 1 - poly);
    const m_eff = RHO_P * Math.PI * rMin * rMin / 2;
    const tau_contact  = Math.PI * Math.sqrt(m_eff / stiffness);
    const tau_drag_rot = RHO_P * rMin / (16 * eta0);
    dt = Math.min(tau_contact / 20, tau_drag_rot / 8);
    stepsPerFrame = Math.min(100, Math.max(3, Math.round(0.002 / (shearRate * dt))));
  }

  // ─── Start ────────────────────────────────────────────────────────
  initParticles();
  updateTimestep();
  loop();

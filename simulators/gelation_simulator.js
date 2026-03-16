  // ─── Configuration ───────────────────────────────────────────────
  const CANVAS_SIZE = 500;
  const N_PARTICLES = 100;
  const R_MEAN = 1.0;         // mean particle radius (absolute units)
  const RHO_P = 1.0;          // 2-D particle area density

  // ─── Timestep ────────────────────────────────────────────────────
  const eta0 = 100.0;                                    // solvent viscosity
  const kBT = 1000.0;                                    // thermal energy scale
  let dt    = 1e-4;                                      // timestep (adaptive)
  let stepsPerFrame = 20;                                // steps per animation frame
  // Pre-allocated force buffers — reused every step to avoid GC pressure
  const _fx     = new Float64Array(N_PARTICLES);
  const _fy     = new Float64Array(N_PARTICLES);
  const _torque = new Float64Array(N_PARTICLES);

  // ─── State ───────────────────────────────────────────────────────
  let particles = [];
  let phi = 0.25;
  const poly = 0.43;          // polydispersity: half-width / mean radius
  const stiffness = 2e6;
  const kt = stiffness * 2 / 7;
  let simTime = 0;
  let L = 1.0;   // box side length — computed from phi and radii
  const mu = 0.5;
  let attrStrength = 0.5;
  let gamma = 25.0;           // U0/kBT dimensionless attraction strength
  const attrRange = 0.25;    // attraction range in units of mean radius
  let contactMap = new Map();

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

  // Standard normal RNG using Box-Muller (cached for efficiency)
  let _randnSpare = null;
  function randn() {
    if (_randnSpare !== null) {
      const v = _randnSpare;
      _randnSpare = null;
      return v;
    }
    let u = 0, v = 0, s = 0;
    while (s === 0 || s >= 1) {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    }
    const mul = Math.sqrt(-2 * Math.log(s) / s);
    _randnSpare = v * mul;
    return u * mul;
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
    simTime = 0;
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
      particles.push({ x, y, vx: 0, vy: 0, omega: 0, theta: 0, r, r0: r0s[i] });
    }
  }

  // ─── Rescale box when phi changes ────────────────────────────────
  function rescalePositions(newL) {
    const ratio = newL / L;
    for (const p of particles) {
      p.x = (p.x * ratio + newL) % newL;
      p.y = (p.y * ratio + newL) % newL;
      p.vx = 0; p.vy = 0;
      p.omega = 0;
    }
    contactMap = new Map();
    L = newL;
  }

  // ─── Forces ──────────────────────────────────────────────────────
  function computeForces(fx, fy, torque) {
    const n = particles.length;
    const newContactMap = new Map();
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = pbc(particles[j].x - particles[i].x);
        const dy = pbc(particles[j].y - particles[i].y);
        const dist = Math.hypot(dx, dy);
        const sigma = particles[i].r + particles[j].r;
        if (dist > 1e-10 && dist < sigma + attrRange * R_MEAN) {
          const nx = dx / dist, ny = dy / dist;
          let fn = 0;
          let ft = 0;

          if (dist < sigma) {
            fn = stiffness * (sigma - dist);

            const tx = -ny, ty = nx;
            const pi = particles[i], pj = particles[j];
            const vs = (pj.vx - pi.vx) * tx + (pj.vy - pi.vy) * ty
                     - pi.omega * pi.r - pj.omega * pj.r;
            const key = i * n + j;
            let delta_t = (contactMap.has(key) ? contactMap.get(key) : 0) + vs * dt;
            ft = kt * delta_t;
            const ft_max = mu * fn;
            if (ft >  ft_max) { ft =  ft_max; delta_t =  ft_max / kt; }
            if (ft < -ft_max) { ft = -ft_max; delta_t = -ft_max / kt; }
            newContactMap.set(key, delta_t);
            fx[i] += ft * tx;  fy[i] += ft * ty;
            fx[j] -= ft * tx;  fy[j] -= ft * ty;
            torque[i] += pi.r * ft;
            torque[j] += pj.r * ft;
          } else {
            const sep = dist - sigma;
            fn = -attrStrength * stiffness * (1 - sep / (attrRange * R_MEAN));
          }

          fx[i] -= fn * nx;  fy[i] -= fn * ny;
          fx[j] += fn * nx;  fy[j] += fn * ny;
        }
      }
    }
    contactMap = newContactMap;
  }

  // ─── Overdamped Brownian dynamics step ────────────────────────────
  function step() {
    const n = particles.length;
    if (n === 0) return;

    _fx.fill(0); _fy.fill(0); _torque.fill(0);
    computeForces(_fx, _fy, _torque);

    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const dragI = 6 * Math.PI * eta0 * p.r;
      const dragRot = 8 * Math.PI * eta0 * p.r * p.r * p.r;
      const vxDet = _fx[i] / dragI;
      const vyDet = _fy[i] / dragI;
      const omegaDet = _torque[i] / dragRot;

      const sigmaT = Math.sqrt(2 * kBT * dt / dragI);
      const sigmaR = Math.sqrt(2 * kBT * dt / dragRot);
      const dx = sigmaT * randn();
      const dy = sigmaT * randn();
      const dtheta = sigmaR * randn();

      p.x += vxDet * dt + dx;
      p.y += vyDet * dt + dy;
      p.theta += omegaDet * dt + dtheta;

      p.vx = vxDet + dx / dt;
      p.vy = vyDet + dy / dt;
      p.omega = omegaDet + dtheta / dt;

      p.x = ((p.x % L) + L) % L;
      p.y = ((p.y % L) + L) % L;
    }
    simTime += dt;
  }

  // ─── Rendering ────────────────────────────────────────────────────
  function draw() {
    const SCALE = CANVAS_SIZE / L;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 0.5;
    const nGrid = 10, gStep = CANVAS_SIZE / nGrid;
    for (let i = 0; i <= nGrid; i++) {
      ctx.beginPath(); ctx.moveTo(i * gStep, 0); ctx.lineTo(i * gStep, CANVAS_SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * gStep); ctx.lineTo(CANVAS_SIZE, i * gStep); ctx.stroke();
    }

    for (const p of particles) {
      const px = p.x * SCALE;
      const py = CANVAS_SIZE - p.y * SCALE;
      const pr = p.r * SCALE;

      const draws = [[px, py]];
      if (py - pr < 0)             draws.push([px, py + CANVAS_SIZE]);
      if (py + pr > CANVAS_SIZE)   draws.push([px, py - CANVAS_SIZE]);

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
    stepsPerFrame = Math.min(200, Math.max(5, Math.round(0.002 / dt)));
  }

  function updateAttrStrength() {
    const denom = stiffness * Math.pow(attrRange * R_MEAN, 2);
    attrStrength = denom > 0 ? (2 * gamma * kBT) / denom : 0;
  }

  // ─── Start ────────────────────────────────────────────────────────
  updateAttrStrength();
  initParticles();
  updateTimestep();
  loop();

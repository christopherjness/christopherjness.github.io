  // ─── Configuration ───────────────────────────────────────────────
  const CANVAS_SIZE = 500;
  const N_PARTICLES = 100;
  const R_MEAN = 0.70;        // mean cap radius (absolute units)
  let aspectRatio = 3.0;      // end-to-end length / diameter
  const RHO_P = 1.0;          // 2-D particle area density
  const shearRate = 1.0;      // fixed shear rate γ̇

  // ─── Viscosity & timestep (mutable — updated by sliders) ─────────
  const eta0 = 100.0;                                    // solvent viscosity (St = 0.01, hard-coded)
  // Stokes translational drag uses equivalent radius from particle area
  let dt    = 1e-4;                                      // timestep (adaptive)
  let stepsPerFrame = 20;                                // steps per animation frame
  // Pre-allocated force buffers — reused every step to avoid GC pressure
  const _fx     = new Float64Array(N_PARTICLES);
  const _fy     = new Float64Array(N_PARTICLES);
  const _torque = new Float64Array(N_PARTICLES);

  // ─── Plot dimensions ─────────────────────────────────────────────
  const PLOT_W = 280, PLOT_H = 180;
  const PLOT_PAD = { top: 20, right: 16, bottom: 36, left: 58 };

  // ─── State ───────────────────────────────────────────────────────
  let particles = [];
  let contacts = [];          // populated each force evaluation
  let currentLeShift = 0;    // kept in sync with step() for rendering
  let phi = 0.65;
  let poly = 0.43;            // polydispersity: half-width / mean radius
  const stiffness = 2e6;  // k = γ̇²·ρ·a³/Γ²; Γ=0.001 (hardest) → k=2e6
  const kt = stiffness * 2 / 7;  // tangential spring stiffness (Mindlin approximation)
  const contactDampingRatioN = 0.70;
  const contactDampingRatioT = 0.45;
  let strain = 0;
  let L = 1.0;   // box side length — computed from phi and radii
  let stressHistory = [];    // { strain, stress } pairs for the strain plot
  let phiHistory    = [];    // { phi, stress } — transient scatter dots, cleared by reset
  let phiGroups     = new Map(); // running geometric-mean per (phi,poly,mu) — survives reset
  const strainWindow = 5;    // width of the sliding plot window (hard-coded)
  let mu = 0.5;              // friction coefficient
  let contactMap = new Map(); // tangential spring state: (i*N_PARTICLES+j) → delta_t

  // ─── Canvas ──────────────────────────────────────────────────────
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const plotCanvas = document.getElementById('plotCanvas');
  const plotCtx = plotCanvas.getContext('2d');
  const phiCanvas  = document.getElementById('phiCanvas');
  const phiCtx     = phiCanvas.getContext('2d');

  // ─── Helpers ─────────────────────────────────────────────────────
  function pbc(d) {
    const half = L / 2;
    while (d >  half) d -= L;
    while (d < -half) d += L;
    return d;
  }

  // particle fill color: lightness encodes size (larger = darker gray)
  function particleColor(p) {
    const L = Math.max(20, Math.min(85, Math.round(78 - p.r0 * 32)));
    return `hsl(0,0%,${L}%)`;
  }

  function capsuleHalfCore(r) {
    return Math.max(0, (aspectRatio - 1) * r);
  }

  function capsuleArea(r, a) {
    return Math.PI * r * r + 4 * a * r;
  }

  function capsuleInertia(m, r, a) {
    const areaRect = 4 * a * r;
    const areaCap = Math.PI * r * r;
    const totalArea = Math.max(areaRect + areaCap, 1e-12);
    const mRect = m * areaRect / totalArea;
    const mCap = m * areaCap / totalArea;
    const iRect = mRect * ((2 * a) * (2 * a) + (2 * r) * (2 * r)) / 12;
    const iCap = mCap * (0.5 * r * r + a * a);
    return iRect + iCap;
  }

  function segmentSegmentClosest(p0x, p0y, p1x, p1y, q0x, q0y, q1x, q1y) {
    const EPS = 1e-12;
    const ux = p1x - p0x, uy = p1y - p0y;
    const vx = q1x - q0x, vy = q1y - q0y;
    const wx = p0x - q0x, wy = p0y - q0y;

    const a = ux * ux + uy * uy;
    const b = ux * vx + uy * vy;
    const c = vx * vx + vy * vy;
    const d = ux * wx + uy * wy;
    const e = vx * wx + vy * wy;
    const D = a * c - b * b;

    let sN, sD = D;
    let tN, tD = D;

    if (D < EPS) {
      sN = 0;
      sD = 1;
      tN = e;
      tD = c;
    } else {
      sN = b * e - c * d;
      tN = a * e - b * d;

      if (sN < 0) {
        sN = 0;
        tN = e;
        tD = c;
      } else if (sN > sD) {
        sN = sD;
        tN = e + b;
        tD = c;
      }
    }

    if (tN < 0) {
      tN = 0;
      if (-d < 0) sN = 0;
      else if (-d > a) sN = sD;
      else {
        sN = -d;
        sD = a;
      }
    } else if (tN > tD) {
      tN = tD;
      if (-d + b < 0) sN = 0;
      else if (-d + b > a) sN = sD;
      else {
        sN = -d + b;
        sD = a;
      }
    }

    const sc = Math.abs(sN) < EPS ? 0 : sN / sD;
    const tc = Math.abs(tN) < EPS ? 0 : tN / tD;

    const cpx = p0x + sc * ux;
    const cpy = p0y + sc * uy;
    const cqx = q0x + tc * vx;
    const cqy = q0y + tc * vy;
    return { cpx, cpy, cqx, cqy };
  }

  function capsuleClosestInPair(pi, pj, dx, dy) {
    const cix = Math.cos(pi.theta), ciy = Math.sin(pi.theta);
    const cjx = Math.cos(pj.theta), cjy = Math.sin(pj.theta);
    const p0x = -pi.a * cix, p0y = -pi.a * ciy;
    const p1x =  pi.a * cix, p1y =  pi.a * ciy;
    const q0x = dx - pj.a * cjx, q0y = dy - pj.a * cjy;
    const q1x = dx + pj.a * cjx, q1y = dy + pj.a * cjy;
    const seg = segmentSegmentClosest(p0x, p0y, p1x, p1y, q0x, q0y, q1x, q1y);
    return { cix: seg.cpx, ciy: seg.cpy, cjx: seg.cqx, cjy: seg.cqy };
  }

  // ─── Initialise: place N particles, compute L from phi ───────────
  function noOverlap(x, y, r, a, theta) {
    for (const p of particles) {
      const dx = pbc(x - p.x), dy = pbc(y - p.y);
      const probe = { theta, a };
      const close = capsuleClosestInPair(probe, p, dx, dy);
      const sep = Math.hypot(close.cjx - close.cix, close.cjy - close.ciy);
      if (sep < r + p.r + 1e-4) return false;
    }
    return true;
  }

  function initParticles() {
    strain = 0;
    stressHistory = [];
    phiHistory    = [];
    contactMap    = new Map();

    // r0 ∈ [−1, 1]: normalised size offset, fixed per particle for lifetime of sim
    // cap radius = R_MEAN * (1 + r0 * poly)
    const r0s = Array.from({ length: N_PARTICLES }, () => Math.random() * 2 - 1);
    const radii = r0s.map(r0 => R_MEAN * (1 + r0 * poly));
    const halfCores = radii.map(r => capsuleHalfCore(r));

    const totalArea = radii.reduce((s, r, i) => s + capsuleArea(r, halfCores[i]), 0);
    L = Math.sqrt(totalArea / phi);

    particles = [];
    for (let i = 0; i < N_PARTICLES; i++) {
      const r = radii[i];
      const a = halfCores[i];
      let x, y, placed = false;
      for (let attempt = 0; attempt < 500; attempt++) {
        x = Math.random() * L;
        y = Math.random() * L;
        const theta = Math.random() * Math.PI * 2;
        if (noOverlap(x, y, r, a, theta)) {
          const area = capsuleArea(r, a);
          const m = RHO_P * area;
          particles.push({
            x, y, vx: 0, vy: 0, ax: 0, ay: 0, omega: 0, alpha: 0,
            theta, r, a, r0: r0s[i], mass: m, I: capsuleInertia(m, r, a)
          });
          placed = true;
          break;
        }
      }
      if (!placed) {
        x = Math.random() * L;
        y = Math.random() * L;
        const theta = Math.random() * Math.PI * 2;
        const area = capsuleArea(r, a);
        const m = RHO_P * area;
        particles.push({
          x, y, vx: 0, vy: 0, ax: 0, ay: 0, omega: 0, alpha: 0,
          theta, r, a, r0: r0s[i], mass: m, I: capsuleInertia(m, r, a)
        });
      }
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

  // ─── Rescale polydispersity (keeps positions, updates radii + L) ──
  function rescalePoly(newPoly) {
    for (const p of particles) {
      p.r = R_MEAN * (1 + p.r0 * newPoly);
      p.a = capsuleHalfCore(p.r);
      p.mass = RHO_P * capsuleArea(p.r, p.a);
      p.I = capsuleInertia(p.mass, p.r, p.a);
    }
    const totalArea = particles.reduce((s, p) => s + capsuleArea(p.r, p.a), 0);
    rescalePositions(Math.sqrt(totalArea / phi));
  }

  function rescaleAspectRatio(newAspectRatio) {
    aspectRatio = newAspectRatio;
    for (const p of particles) {
      p.a = capsuleHalfCore(p.r);
      p.mass = RHO_P * capsuleArea(p.r, p.a);
      p.I = capsuleInertia(p.mass, p.r, p.a);
    }
    const totalArea = particles.reduce((s, p) => s + capsuleArea(p.r, p.a), 0);
    rescalePositions(Math.sqrt(totalArea / phi));
  }

  // ─── Rescale box when phi changes (preserves particle structure) ──
  function rescaleBox(newPhi) {
    const totalArea = particles.reduce((s, p) => s + capsuleArea(p.r, p.a), 0);
    rescalePositions(Math.sqrt(totalArea / newPhi));
  }

  // ─── Forces ──────────────────────────────────────────────────────
  function computeForces(fx, fy, torque, leShift) {
    const n = particles.length;
    const newContactMap = new Map();
    contacts = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const pi = particles[i], pj = particles[j];
        const baseDx = pj.x - pi.x;
        const baseDy = pj.y - pi.y;

        let bestKy = 0;
        let bestDx = 0;
        let bestDy = 0;
        let bestClose = null;
        let bestDist = Infinity;

        for (let kyTry = -1; kyTry <= 1; kyTry++) {
          let dxTry = baseDx - kyTry * leShift;
          let dyTry = baseDy - kyTry * L;
          dxTry = pbc(dxTry);
          const closeTry = capsuleClosestInPair(pi, pj, dxTry, dyTry);
          const sepXTry = closeTry.cjx - closeTry.cix;
          const sepYTry = closeTry.cjy - closeTry.ciy;
          const distTry = Math.hypot(sepXTry, sepYTry);
          if (distTry < bestDist) {
            bestDist = distTry;
            bestKy = kyTry;
            bestDx = dxTry;
            bestDy = dyTry;
            bestClose = closeTry;
          }
        }

        const sigma = pi.r + pj.r;
        if (bestDist < sigma) {
          const close = bestClose;
          const dist = bestDist;
          const sepX = close.cjx - close.cix;
          const sepY = close.cjy - close.ciy;
          let nx, ny;
          if (dist > 1e-10) {
            nx = sepX / dist;
            ny = sepY / dist;
          } else {
            const cdist = Math.hypot(bestDx, bestDy);
            if (cdist > 1e-10) {
              nx = bestDx / cdist;
              ny = bestDy / cdist;
            } else {
              nx = Math.cos(pi.theta);
              ny = Math.sin(pi.theta);
            }
          }
          const overlap = sigma - dist;
          const tx = -ny, ty = nx;

          const rciX = close.cix + nx * pi.r;
          const rciY = close.ciy + ny * pi.r;
          const rcjX = close.cjx - nx * pj.r;
          const rcjY = close.cjy - ny * pj.r;

          const vix = pi.vx - pi.omega * rciY;
          const viy = pi.vy + pi.omega * rciX;
          const vjx = (pj.vx - bestKy * shearRate * L) - pj.omega * rcjY;
          const vjy = pj.vy + pj.omega * rcjX;
          const dvx = vjx - vix;
          const dvy = vjy - viy;
          const vn = dvx * nx + dvy * ny;
          const vs = (vjx - vix) * tx + (vjy - viy) * ty;

          const mEff = (pi.mass * pj.mass) / Math.max(pi.mass + pj.mass, 1e-12);
          const gammaN = 2 * contactDampingRatioN * Math.sqrt(stiffness * mEff);
          const gammaT = 2 * contactDampingRatioT * Math.sqrt(kt * mEff);

          const fnElastic = stiffness * overlap;
          let fn = fnElastic - gammaN * vn;
          if (fn < 0) fn = 0;

          const key = i * n + j;
          let delta_t = (contactMap.has(key) ? contactMap.get(key) : 0) + vs * dt;
          let ft = kt * delta_t - gammaT * vs;
          const ft_max = mu * fn;
          if (ft >  ft_max) { ft =  ft_max; delta_t = (ft + gammaT * vs) / kt; }
          if (ft < -ft_max) { ft = -ft_max; delta_t = (ft + gammaT * vs) / kt; }
          if (fn > 1e-3 || Math.abs(delta_t) > 1e-10) newContactMap.set(key, delta_t);

          const fcx = fn * nx + ft * tx;
          const fcy = fn * ny + ft * ty;

          fx[i] -= fcx; fy[i] -= fcy;
          fx[j] += fcx; fy[j] += fcy;
          torque[i] += rciX * (-fcy) - rciY * (-fcx);
          torque[j] += rcjX * fcy - rcjY * fcx;

          contacts.push({
            i, j, fn, ft, nx, ny, dy: bestDy,
            cix: close.cix, ciy: close.ciy, cjx: close.cjx, cjy: close.cjy,
            rciX, rciY, rcjX, rcjY
          });
        }
      }
    }
    contactMap = newContactMap;
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const req = Math.sqrt(capsuleArea(p.r, p.a) / Math.PI);
      const dragI = 6 * Math.PI * eta0 * req;
      fx[i]      -= dragI * (p.vx - shearRate * p.y);
      fy[i]      -= dragI * p.vy;
      torque[i]  -= 8 * Math.PI * eta0 * req * req * req * (p.omega + shearRate / 2);  // Stokes rotational drag (fluid vorticity = −γ̇/2)
    }
  }

  // ─── Relative viscosity  η/η₀ = 1 - τ / (γ̇ · η₀ · L²) ──────────
  // τ = -(1/V) Σ (fn·nₓ + ft·nᵧ)·Δy  (virial, full contact force)
  function computeShearStress() {
    let tau = 0;
    for (const c of contacts) tau += (c.fn * c.nx + c.ft * c.ny) * c.dy;
    return 1 - tau / (L * L * shearRate * eta0);
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

    // LE shift in canvas pixels, normalised to [0, CANVAS_SIZE)
    const shiftPx = ((currentLeShift % L) + L) % L * SCALE;

    for (const p of particles) {
      const px = p.x * SCALE;
      const py = CANVAS_SIZE - p.y * SCALE;
      const pr = p.r * SCALE;
      const pa = p.a * SCALE;
      const extent = pa + pr;

      // Collect draw positions: main + y-boundary ghosts (with LE x-shift)
      const draws = [[px, py]];
      if (py - extent < 0)             draws.push([px - shiftPx, py + CANVAS_SIZE]);
      if (py + extent > CANVAS_SIZE)   draws.push([px + shiftPx, py - CANVAS_SIZE]);

      // For each position, also add x-boundary mirrors (after normalising x)
      const allDraws = [];
      for (const [gx, gy] of draws) {
        const nx = ((gx % CANVAS_SIZE) + CANVAS_SIZE) % CANVAS_SIZE;
        allDraws.push([nx, gy]);
        if (nx - extent < 0)             allDraws.push([nx + CANVAS_SIZE, gy]);
        if (nx + extent > CANVAS_SIZE)   allDraws.push([nx - CANVAS_SIZE, gy]);
      }

      const fill = particleColor(p);
      const cosT = Math.cos(p.theta);
      const sinT = Math.sin(p.theta);

      for (const [gx, gy] of allDraws) {
        // Get base color
        const L = Math.max(20, Math.min(85, Math.round(78 - p.r0 * 32)));
        const baseColor = `hsl(0,0%,${L}%)`;
        const highlightColor = `hsl(0,0%,${Math.min(95, L + 15)}%)`;

        ctx.save();
        ctx.translate(gx, gy);
        ctx.rotate(-p.theta);
        const gradient = ctx.createLinearGradient(-pa - pr, 0, pa + pr, 0);
        gradient.addColorStop(0, highlightColor);
        gradient.addColorStop(0.5, baseColor);
        gradient.addColorStop(1, highlightColor);

        ctx.beginPath();
        ctx.moveTo(-pa, -pr);
        ctx.lineTo(pa, -pr);
        ctx.arc(pa, 0, pr, -Math.PI / 2, Math.PI / 2);
        ctx.lineTo(-pa, pr);
        ctx.arc(-pa, 0, pr, Math.PI / 2, -Math.PI / 2);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.7)';
        ctx.lineWidth = 0.8;
        ctx.stroke();

        ctx.restore();
      }
    }

    // contact force lines drawn on top (width encodes magnitude)
    if (contacts.length > 0) {
      const meanF = contacts.reduce((s, c) => s + c.fn, 0) / contacts.length;
      ctx.strokeStyle = 'hsla(0,100%,50%,0.9)';
      for (const c of contacts) {
        const pi = particles[c.i];
        const cix = (pi.x + c.rciX) * SCALE;
        const ciy = CANVAS_SIZE - (pi.y + c.rciY) * SCALE;
        const cjx = (pi.x + c.rcjX) * SCALE;
        const cjy = CANVAS_SIZE - (pi.y + c.rcjY) * SCALE;
        ctx.beginPath();
        ctx.moveTo(cix, ciy);
        ctx.lineTo(cjx, cjy);
        ctx.lineWidth = Math.max(0.5, Math.min(c.fn / meanF * 2, 8));
        ctx.stroke();
      }
    }
  }

  // ─── Mean-marker shape for φ–viscosity plot ──────────────────────
  // circle=frictionless(<0.1), square=low(<0.4), diamond=moderate(<0.7), triangle=rough(≥0.7)
  function muPath(ctx2d, cx, cy, muVal, s) {
    ctx2d.beginPath();
    if (muVal < 0.1) {
      ctx2d.arc(cx, cy, s, 0, Math.PI * 2);
    } else if (muVal < 0.4) {
      ctx2d.rect(cx - s, cy - s, s * 2, s * 2);
    } else if (muVal < 0.7) {
      const d = s * 1.35;
      ctx2d.moveTo(cx, cy - d); ctx2d.lineTo(cx + d, cy);
      ctx2d.lineTo(cx, cy + d); ctx2d.lineTo(cx - d, cy);
      ctx2d.closePath();
    } else {
      const h = s * 1.6, w = s * 1.4;
      ctx2d.moveTo(cx, cy - h);
      ctx2d.lineTo(cx + w, cy + h * 0.65);
      ctx2d.lineTo(cx - w, cy + h * 0.65);
      ctx2d.closePath();
    }
  }

  // ─── Log-spaced tick generator ───────────────────────────────────
  function logTicks(lo, hi) {
    const ticks = [];
    const dMin = Math.floor(Math.log10(lo));
    const dMax = Math.ceil(Math.log10(hi));
    for (let d = dMin; d <= dMax; d++) {
      for (const m of [1, 2, 5]) {
        const v = m * Math.pow(10, d);
        if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v);
      }
    }
    return ticks;
  }

  // ─── Stress–Strain plot ───────────────────────────────────────────
  function drawStressPlot() {
    const PW = PLOT_W, PH = PLOT_H;
    const pad = PLOT_PAD;
    const areaX = pad.left, areaY = pad.top;
    const areaW = PW - pad.left - pad.right;
    const areaH = PH - pad.top - pad.bottom;

    plotCtx.clearRect(0, 0, PW, PH);

    // X domain: sliding window of the last 2 strain units
    const xMax = strain;
    const xMin = Math.max(0, strain - strainWindow);
    const xRange = Math.max(xMax - xMin, 1e-9);

    // Filter history to the current window (with a tiny look-behind for line continuity)
    const inWindow = stressHistory.filter(d => d.strain >= xMin - 0.01);

    // Y domain: log scale, floor at 1
    let yMax = 2;
    if (inWindow.length > 0) {
      let hi = -Infinity;
      for (const d of inWindow) { if (d.stress > hi) hi = d.stress; }
      yMax = hi > 1.01 ? Math.pow(10, Math.log10(hi) + 0.1) : 2;
    }
    const yMaxLog = Math.log10(yMax);

    // Coordinate transforms
    const toCanvasX = x => areaX + (x - xMin) / xRange * areaW;
    const toCanvasY = y => areaY + areaH - Math.log10(Math.max(y, 1)) / yMaxLog * areaH;

    // Plot area background with subtle gradient
    plotCtx.fillStyle = '#fafafa';
    plotCtx.fillRect(areaX, areaY, areaW, areaH);

    // Horizontal grid at log-spaced ticks (subtle, lighter)
    const yTicks = logTicks(1, yMax);
    plotCtx.strokeStyle = '#d8d8d8';
    plotCtx.lineWidth = 0.5;
    plotCtx.setLineDash([2, 3]);
    for (const tv of yTicks) {
      const gy = toCanvasY(tv);
      plotCtx.beginPath();
      plotCtx.moveTo(areaX, gy);
      plotCtx.lineTo(areaX + areaW, gy);
      plotCtx.stroke();
    }
    plotCtx.setLineDash([]);

    // Vertical grid every 0.5 strain (subtle)
    const strainStep = 0.5;
    for (let s = Math.ceil(xMin / strainStep) * strainStep; s <= xMax + 0.01; s += strainStep) {
      const gx = toCanvasX(s);
      if (gx >= areaX && gx <= areaX + areaW) {
        plotCtx.beginPath();
        plotCtx.strokeStyle = '#d8d8d8';
        plotCtx.setLineDash([2, 3]);
        plotCtx.moveTo(gx, areaY);
        plotCtx.lineTo(gx, areaY + areaH);
        plotCtx.stroke();
        plotCtx.setLineDash([]);
      }
    }

    // Stress curve with improved styling
    if (inWindow.length > 1) {
      plotCtx.beginPath();
      plotCtx.strokeStyle = '#2563eb';
      plotCtx.lineWidth = 2;
      plotCtx.lineCap = 'round';
      plotCtx.lineJoin = 'round';
      let first = true;
      for (const d of inWindow) {
        if (d.strain < xMin || d.stress < 1) { first = true; continue; }
        const cx = toCanvasX(d.strain);
        const cy = toCanvasY(d.stress);
        if (first) { plotCtx.moveTo(cx, cy); first = false; }
        else plotCtx.lineTo(cx, cy);
      }
      plotCtx.stroke();
    }

    // Axes border
    plotCtx.strokeStyle = '#666';
    plotCtx.lineWidth = 1;
    plotCtx.strokeRect(areaX, areaY, areaW, areaH);

    // X-axis tick labels with better styling
    plotCtx.fillStyle = '#666';
    plotCtx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    plotCtx.textAlign = 'center';
    plotCtx.textBaseline = 'top';
    for (let i = 0; i <= 3; i++) {
      const s = xMin + i * xRange / 3;
      const tx = toCanvasX(s);
      plotCtx.fillText(s.toFixed(1), tx, areaY + areaH + 8);
    }

    // Y-axis tick labels at log-spaced values
    plotCtx.textAlign = 'right';
    plotCtx.textBaseline = 'middle';
    for (const tv of yTicks) {
      plotCtx.fillText(tv < 10 ? tv.toFixed(1) : tv.toFixed(0), areaX - 8, toCanvasY(tv));
    }

    // Axis labels with improved styling
    plotCtx.fillStyle = '#1a1a1a';
    plotCtx.font = '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    plotCtx.textAlign = 'center';
    plotCtx.textBaseline = 'bottom';
    plotCtx.fillText('Strain', areaX + areaW / 2, PH - 4);

    plotCtx.save();
    plotCtx.translate(10, areaY + areaH / 2);
    plotCtx.rotate(-Math.PI / 2);
    plotCtx.textBaseline = 'top';
    plotCtx.fillText('Viscosity', 0, 0);
    plotCtx.restore();
  }

  // ─── Viscosity–φ scatter plot ─────────────────────────────────────
  function drawPhiPlot() {
    const PW = PLOT_W, PH = PLOT_H;
    const pad = PLOT_PAD;
    const areaX = pad.left, areaY = pad.top;
    const areaW = PW - pad.left - pad.right;
    const areaH = PH - pad.top - pad.bottom;

    phiCtx.clearRect(0, 0, PW, PH);

    // X domain: 0.05 beyond slider limits
    const xMin = 0.35, xMax = 0.95, xRange = xMax - xMin;

    // Y domain: fixed log scale 1–700
    const yMax = 700;
    const yMaxLog = Math.log10(yMax);

    const toX = x => areaX + (x - xMin) / xRange * areaW;
    const toY = y => areaY + areaH - Math.log10(Math.max(y, 1)) / yMaxLog * areaH;

    // Plot background
    phiCtx.fillStyle = '#fafafa';
    phiCtx.fillRect(areaX, areaY, areaW, areaH);

    // Horizontal grid at log-spaced ticks (subtle)
    const yTicks = logTicks(1, yMax);
    phiCtx.strokeStyle = '#d8d8d8';
    phiCtx.lineWidth = 0.5;
    phiCtx.setLineDash([2, 3]);
    for (const tv of yTicks) {
      const gy = toY(tv);
      phiCtx.beginPath(); phiCtx.moveTo(areaX, gy); phiCtx.lineTo(areaX + areaW, gy); phiCtx.stroke();
    }
    phiCtx.setLineDash([]);

    // Vertical grid every 0.1 in φ (subtle)
    phiCtx.strokeStyle = '#d8d8d8';
    phiCtx.setLineDash([2, 3]);
    for (let x = 0.4; x <= 0.91; x += 0.1) {
      const gx = toX(x);
      phiCtx.beginPath(); phiCtx.moveTo(gx, areaY); phiCtx.lineTo(gx, areaY + areaH); phiCtx.stroke();
    }
    phiCtx.setLineDash([]);

    // Scatter dots — hue encodes polydispersity (blue δ=0 → red δ=0.85)
    for (const d of phiHistory) {
      if (d.stress < 1) continue;
      const cx = toX(d.phi);
      const cy = toY(d.stress);
      if (cy < areaY || cy > areaY + areaH) continue;
      const hue = 220 * (1 - d.poly / 0.85);
      phiCtx.fillStyle = `hsla(${hue},80%,45%,0.4)`;
      phiCtx.beginPath();
      phiCtx.arc(cx, cy, 2, 0, Math.PI * 2);
      phiCtx.fill();
    }

    // Time-averaged markers per parameter set (phi, poly, mu)
    for (const g of phiGroups.values()) {
      const meanStress = Math.pow(10, g.logSum / g.n);
      const cx = toX(g.phi);
      const cy = toY(meanStress);
      if (cy < areaY || cy > areaY + areaH) continue;
      const hue = 220 * (1 - g.poly / 0.85);
      phiCtx.fillStyle = `hsl(${hue},85%,38%)`;
      phiCtx.strokeStyle = 'rgba(0,0,0,0.9)';
      phiCtx.lineWidth = 1.3;
      muPath(phiCtx, cx, cy, g.mu, 5);
      phiCtx.fill();
      phiCtx.stroke();
    }

    // Current-φ vertical marker
    phiCtx.strokeStyle = 'rgba(220,80,80,0.6)';
    phiCtx.lineWidth = 2;
    phiCtx.setLineDash([3, 4]);
    const curX = toX(phi);
    phiCtx.beginPath(); phiCtx.moveTo(curX, areaY); phiCtx.lineTo(curX, areaY + areaH); phiCtx.stroke();
    phiCtx.setLineDash([]);

    // Axes border
    phiCtx.strokeStyle = '#666';
    phiCtx.lineWidth = 1;
    phiCtx.strokeRect(areaX, areaY, areaW, areaH);

    // X-axis tick labels with improved styling
    phiCtx.fillStyle = '#666';
    phiCtx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    phiCtx.textAlign = 'center';
    phiCtx.textBaseline = 'top';
    for (let x = 0.4; x <= 0.91; x += 0.1) {
      phiCtx.fillText(x.toFixed(2), toX(x), areaY + areaH + 8);
    }

    // Y-axis tick labels at log-spaced values
    phiCtx.textAlign = 'right';
    phiCtx.textBaseline = 'middle';
    for (const tv of yTicks) {
      phiCtx.fillText(tv < 10 ? tv.toFixed(1) : tv.toFixed(0), areaX - 8, toY(tv));
    }

    // Axis labels with improved styling
    phiCtx.fillStyle = '#1a1a1a';
    phiCtx.font = '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    phiCtx.textAlign = 'center';
    phiCtx.textBaseline = 'bottom';
    phiCtx.fillText('Area fraction', areaX + areaW / 2, PH - 4);

    phiCtx.save();
    phiCtx.translate(10, areaY + areaH / 2);
    phiCtx.rotate(-Math.PI / 2);
    phiCtx.textBaseline = 'top';
    phiCtx.fillText('Average viscosity', 0, 0);
    phiCtx.restore();
  }

  // ─── Animation loop ───────────────────────────────────────────────
  function loop() {
    for (let s = 0; s < stepsPerFrame; s++) step();

    // Record virial shear stress at current strain; prune entries outside visible window
    const eta = computeShearStress();
    stressHistory.push({ strain, stress: eta });
    while (stressHistory.length > 0 && stressHistory[0].strain < strain - strainWindow - 0.5)
      stressHistory.shift();
    phiHistory.push({ phi, stress: eta, poly, mu });
    if (eta >= 1) {
      const gkey = `${phi.toFixed(2)}_${poly.toFixed(2)}_${mu.toFixed(2)}`;
      if (!phiGroups.has(gkey)) phiGroups.set(gkey, { phi, poly, mu, logSum: 0, n: 0 });
      const g = phiGroups.get(gkey);
      g.logSum += Math.log10(eta);
      g.n++;
    }

    draw();
    drawStressPlot();
    drawPhiPlot();
    requestAnimationFrame(loop);
  }

  // ─── Adaptive timestep ────────────────────────────────────────────
  // DT must resolve both the contact spring oscillation and Stokes drag relaxation.
  // τ_contact = π·√(m_eff/k);  τ_drag_rot = ρ_p·r_min/(16·η₀)  (smallest particle, rotation)
  function updateTimestep() {
    const rMin = R_MEAN * Math.max(0.1, 1 - poly);
    const aMin = capsuleHalfCore(rMin);
    const m_eff = RHO_P * capsuleArea(rMin, aMin) / 2;
    const tau_contact  = Math.PI * Math.sqrt(m_eff / stiffness);
    const reqMin = Math.sqrt(capsuleArea(rMin, aMin) / Math.PI);
    const tau_drag_rot = RHO_P * reqMin / (16 * eta0);
    // Spring oscillations need tighter resolution than drag relaxation
    dt = Math.min(tau_contact / 45, tau_drag_rot / 12);
    // keep ~0.0008 strain per frame for smoother contact evolution
    stepsPerFrame = Math.min(180, Math.max(8, Math.round(0.0008 / (shearRate * dt))));
  }

  // ─── Slider descriptor functions ─────────────────────────────────
  function polyDesc(v)  { return v < 0.05 ? 'monodisperse' : v < 0.35 ? 'low poly' : v < 0.60 ? 'moderate' : 'polydisperse'; }

  // ─── UI wiring ────────────────────────────────────────────────────
  const phiSlider  = document.getElementById('phi-slider');
  const phiVal     = document.getElementById('phi-val');
  const polySlider = document.getElementById('poly-slider');
  const polyVal    = document.getElementById('poly-val');
  const arSlider   = document.getElementById('ar-slider');
  const arVal      = document.getElementById('ar-val');

  phiSlider.addEventListener('input', () => {
    phi = parseFloat(phiSlider.value);
    phiVal.textContent = phi.toFixed(2);
    rescaleBox(phi);
  });

  polySlider.addEventListener('input', () => {
    poly = parseFloat(polySlider.value);
    polyVal.textContent = polyDesc(poly);
    rescalePoly(poly);
    updateTimestep();
  });

  arSlider.addEventListener('input', () => {
    aspectRatio = parseFloat(arSlider.value);
    arVal.textContent = aspectRatio.toFixed(2);
    rescaleAspectRatio(aspectRatio);
    updateTimestep();
  });


  const muSlider = document.getElementById('mu-slider');
  const muVal    = document.getElementById('mu-val');
  muSlider.addEventListener('input', () => {
    mu = parseFloat(muSlider.value);
    muVal.textContent = mu.toFixed(2);
  });


  // ─── Start ────────────────────────────────────────────────────────
  initParticles();
  updateTimestep();   // sets dt, stepsPerFrame, and ip-dt display
  loop();

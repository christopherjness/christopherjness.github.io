const flowCanvas = document.getElementById("flowCanvas");
const flowCtx = flowCanvas.getContext("2d");

const startResetBtn = document.getElementById("startResetBtn");
const uAxisLabel = document.getElementById("uAxisLabel");
const phiAxisLabel = document.getElementById("phiAxisLabel");
const tauAxisLabel = document.getElementById("tauAxisLabel");

const sim = {
  nx: 120,
  ny: 100,
  width: flowCanvas.width,
  height: flowCanvas.height,
  plateSpeed: 2,
  baseViscosity: 1,
  intrinsicViscosity: 2.5,
  phiMaxPacking: 0.63,
  migrationRate: 2.5,
  momentumDiffusionRate: 10,
  migrationDiffusivityFactor: 0.08,
  stepsPerFrame: 30,
  convergenceTolU: 2e-4,
  convergenceTolPhi: 1e-4,
  maxExtraConvergenceSteps: 160,
  isStarted: false,
  isDrawingPhi: false,
  lastDrawJ: null,
  lastDrawPhi: null,
  periodicMixStrength: 0.02,
  dt: 0,
  u: [],
  phi: new Float64Array(100),
  initialPhi: new Float64Array(100),
  muByY: new Float64Array(100),
  time: 0,
};

const plateVisual = {
  stripeOffset: 0,
};

function resizeFlowCanvasToDisplay() {
  const size = Math.max(240, Math.floor(flowCanvas.clientWidth));
  if (flowCanvas.width !== size || flowCanvas.height !== size) {
    flowCanvas.width = size;
    flowCanvas.height = size;
    sim.width = size;
    sim.height = size;
  }
}

function updateBelowAxisLabelPositions() {
  const uCenter = (16 + 132) * 0.5;
  const phiCenter = sim.width * 0.5;
  const tauCenter = sim.width - 74;

  uAxisLabel.style.left = `${(uCenter / sim.width) * 100}%`;
  phiAxisLabel.style.left = `${(phiCenter / sim.width) * 100}%`;
  tauAxisLabel.style.left = `${(tauCenter / sim.width) * 100}%`;
}

function setup() {
  resizeFlowCanvasToDisplay();
  updateBelowAxisLabelPositions();
  updateStartButton();
  initializeField();
  draw();
  requestAnimationFrame(loop);
}

function updateStartButton() {
  startResetBtn.textContent = sim.isStarted ? "RESET" : "START";
}

function sumArray(values) {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i];
  }
  return total;
}

function enforcePhiBoundsConservative(phiValues, targetMass) {
  const minPhi = 0;
  const maxPhi = sim.phiMaxPacking - 1e-4;
  const corrected = new Float64Array(phiValues.length);

  for (let i = 0; i < phiValues.length; i += 1) {
    corrected[i] = Math.min(Math.max(phiValues[i], minPhi), maxPhi);
  }

  for (let iter = 0; iter < 8; iter += 1) {
    const currentMass = sumArray(corrected);
    const massError = targetMass - currentMass;
    if (Math.abs(massError) < 1e-10) {
      break;
    }

    if (massError > 0) {
      let capacity = 0;
      for (let i = 0; i < corrected.length; i += 1) {
        capacity += Math.max(0, maxPhi - corrected[i]);
      }
      if (capacity <= 0) {
        break;
      }
      const scale = Math.min(1, massError / capacity);
      for (let i = 0; i < corrected.length; i += 1) {
        const room = Math.max(0, maxPhi - corrected[i]);
        corrected[i] += scale * room;
      }
    } else {
      let removable = 0;
      for (let i = 0; i < corrected.length; i += 1) {
        removable += Math.max(0, corrected[i] - minPhi);
      }
      if (removable <= 0) {
        break;
      }
      const scale = Math.min(1, (-massError) / removable);
      for (let i = 0; i < corrected.length; i += 1) {
        const excess = Math.max(0, corrected[i] - minPhi);
        corrected[i] -= scale * excess;
      }
    }
  }

  return corrected;
}

function clampPhi(phi) {
  return Math.min(Math.max(phi, 0), sim.phiMaxPacking - 1e-4);
}

function kriegerDoughertyViscosity(phi) {
  const safePhi = clampPhi(phi);
  const exponent = 2;
  const factor = Math.max(1 - safePhi / sim.phiMaxPacking, 1e-6);
  return sim.baseViscosity * Math.pow(factor, -exponent);
}

function buildInitialPhiProfile() {
  sim.phi = new Float64Array(sim.ny);
  sim.phi.fill(0);
}

function updateMaterialFromPhi() {
  sim.muByY = new Float64Array(sim.ny);
  let maxMu = 0;

  for (let j = 0; j < sim.ny; j += 1) {
    const mu = kriegerDoughertyViscosity(sim.phi[j]);
    sim.muByY[j] = mu;
    if (mu > maxMu) {
      maxMu = mu;
    }
  }

  const dy = 1 / (sim.ny - 1);
  const effectiveMomentumRate = Math.max(sim.momentumDiffusionRate, 1e-8);
  sim.dt = 0.28 * dy * dy / Math.max(maxMu * effectiveMomentumRate, 1e-8);
}

function initializeField() {
  sim.u = new Array(sim.nx);
  for (let i = 0; i < sim.nx; i += 1) {
    sim.u[i] = new Float64Array(sim.ny);
  }

  buildInitialPhiProfile();
  sim.initialPhi = new Float64Array(sim.phi);
  updateMaterialFromPhi();

  for (let i = 0; i < sim.nx; i += 1) {
    sim.u[i].fill(0);
  }

  sim.time = 0;
  plateVisual.stripeOffset = 0;
}

function stepSimulation() {
  const next = new Array(sim.nx);
  for (let i = 0; i < sim.nx; i += 1) {
    next[i] = new Float64Array(sim.ny);
  }

  const dy = 1 / (sim.ny - 1);
  let maxDeltaU = 0;

  for (let i = 0; i < sim.nx; i += 1) {
    const left = (i - 1 + sim.nx) % sim.nx;
    const right = (i + 1) % sim.nx;

    next[i][0] = 0;
    next[i][sim.ny - 1] = sim.plateSpeed;

    for (let j = 1; j < sim.ny - 1; j += 1) {
      const here = sim.u[i][j];

      const muUp = 0.5 * (sim.muByY[j] + sim.muByY[j + 1]);
      const muDown = 0.5 * (sim.muByY[j] + sim.muByY[j - 1]);

      const gradUp = sim.u[i][j + 1] - here;
      const gradDown = here - sim.u[i][j - 1];
      const viscousY = (muUp * gradUp - muDown * gradDown) / (dy * dy);

      const lapX = sim.u[left][j] - 2 * here + sim.u[right][j];
      const periodicMix = sim.periodicMixStrength * sim.muByY[j] * lapX / (dy * dy);

      next[i][j] = here + sim.dt * sim.momentumDiffusionRate * (viscousY + periodicMix);
      maxDeltaU = Math.max(maxDeltaU, Math.abs(next[i][j] - here));
    }
  }

  sim.u = next;
  const velocityProfile = averageProfile();
  const stressProfile = stressProfileFromVelocity(velocityProfile);
  const maxDeltaPhi = migratePhiByStressGradient(stressProfile);
  sim.time += sim.dt;

  return {
    maxDeltaU,
    maxDeltaPhi,
  };
}

function averageProfile() {
  const profile = new Float64Array(sim.ny);

  for (let j = 0; j < sim.ny; j += 1) {
    let sum = 0;
    for (let i = 0; i < sim.nx; i += 1) {
      sum += sim.u[i][j];
    }
    profile[j] = sum / sim.nx;
  }

  return profile;
}

function stressProfileFromVelocity(profile) {
  const tau = new Float64Array(sim.ny);
  const dy = 1 / (sim.ny - 1);

  for (let j = 0; j < sim.ny; j += 1) {
    let dudy;

    if (j === 0) {
      dudy = (profile[1] - profile[0]) / dy;
    } else if (j === sim.ny - 1) {
      dudy = (profile[sim.ny - 1] - profile[sim.ny - 2]) / dy;
    } else {
      dudy = (profile[j + 1] - profile[j - 1]) / (2 * dy);
    }

    tau[j] = sim.muByY[j] * dudy;
  }

  return tau;
}

function computeMigrationFluxFaces(phiState, stressProfile, subDt) {
  const dy = 1 / (sim.ny - 1);
  const phiCap = sim.phiMaxPacking - 1e-4;
  const flux = new Float64Array(sim.ny + 1);
  flux[0] = 0;
  flux[sim.ny] = 0;

  for (let j = 1; j < sim.ny; j += 1) {
    const stressHere = Math.abs(stressProfile[j]);
    const stressBelow = Math.abs(stressProfile[j - 1]);
    const dTauDy = (stressHere - stressBelow) / dy;
    const phiFace = 0.5 * (phiState[j] + phiState[j - 1]);
    const mobility = sim.migrationRate * phiFace * Math.max(0, 1 - phiFace / sim.phiMaxPacking);
    const migrationFlux = -mobility * dTauDy;

    const dPhiDy = (phiState[j] - phiState[j - 1]) / dy;
    const diffusionFlux = -sim.migrationRate * sim.migrationDiffusivityFactor * dPhiDy;

    let faceFlux = migrationFlux + diffusionFlux;

    if (faceFlux > 0) {
      const donor = j - 1;
      const receiver = j;
      const maxFromDonor = (phiState[donor] * dy) / Math.max(subDt, 1e-12);
      const maxToReceiver = ((phiCap - phiState[receiver]) * dy) / Math.max(subDt, 1e-12);
      faceFlux = Math.min(faceFlux, Math.max(0, maxFromDonor), Math.max(0, maxToReceiver));
    } else if (faceFlux < 0) {
      const donor = j;
      const receiver = j - 1;
      const maxFromDonor = (phiState[donor] * dy) / Math.max(subDt, 1e-12);
      const maxToReceiver = ((phiCap - phiState[receiver]) * dy) / Math.max(subDt, 1e-12);
      const limitedMagnitude = Math.min(-faceFlux, Math.max(0, maxFromDonor), Math.max(0, maxToReceiver));
      faceFlux = -limitedMagnitude;
    }

    flux[j] = faceFlux;
  }

  return flux;
}

function migratePhiByStressGradient(stressProfile) {
  if (sim.migrationRate <= 0) {
    return 0;
  }

  const dy = 1 / (sim.ny - 1);
  const targetMass = sumArray(sim.phi);
  let phiCurrent = new Float64Array(sim.phi);

  const maxStableDt = 0.30 * dy * dy / Math.max(sim.migrationRate, 1e-8);
  const nSubSteps = Math.max(1, Math.ceil(sim.dt / maxStableDt));
  const subDt = sim.dt / nSubSteps;

  for (let step = 0; step < nSubSteps; step += 1) {
    const flux = computeMigrationFluxFaces(phiCurrent, stressProfile, subDt);

    const updated = new Float64Array(sim.ny);
    for (let j = 0; j < sim.ny; j += 1) {
      const divFlux = (flux[j + 1] - flux[j]) / dy;
      updated[j] = phiCurrent[j] - subDt * divFlux;
    }

    phiCurrent = enforcePhiBoundsConservative(updated, targetMass);
  }

  let maxDeltaPhi = 0;
  for (let j = 0; j < sim.ny; j += 1) {
    maxDeltaPhi = Math.max(maxDeltaPhi, Math.abs(phiCurrent[j] - sim.phi[j]));
  }

  sim.phi = phiCurrent;
  updateMaterialFromPhi();
  return maxDeltaPhi;
}

function velocityColor(u, uMax) {
  const n = Math.min(Math.max(u / Math.max(uMax, 1e-8), 0), 1);
  const shade = Math.floor(55 + 185 * n);
  return `rgb(${shade}, ${shade}, ${shade})`;
}

function drawFlowDomain() {
  const dx = sim.width / sim.nx;
  const dyPx = sim.height / sim.ny;

  flowCtx.clearRect(0, 0, sim.width, sim.height);

  for (let i = 0; i < sim.nx; i += 1) {
    for (let j = 0; j < sim.ny; j += 1) {
      const y = sim.height - (j + 1) * dyPx;
      flowCtx.fillStyle = velocityColor(sim.u[i][j], sim.plateSpeed);
      flowCtx.fillRect(i * dx, y, Math.ceil(dx), Math.ceil(dyPx) + 1);
    }
  }

  const topPlateH = Math.max(4, sim.height * 0.025);
  flowCtx.fillStyle = "#dddddd";
  flowCtx.fillRect(0, 0, sim.width, topPlateH);

  if (sim.isStarted) {
    plateVisual.stripeOffset = (plateVisual.stripeOffset + sim.plateSpeed * 0.45) % 28;
  }
  flowCtx.fillStyle = "#8b8b8b";
  for (let x = -28 + plateVisual.stripeOffset; x < sim.width + 28; x += 28) {
    flowCtx.fillRect(x, 0, 14, topPlateH);
  }

  const bottomPlateH = topPlateH;
  flowCtx.fillStyle = "#7e7e7e";
  flowCtx.fillRect(0, sim.height - bottomPlateH, sim.width, bottomPlateH);

  flowCtx.strokeStyle = "#ffffff44";
  flowCtx.lineWidth = 2;
  flowCtx.beginPath();
  flowCtx.moveTo(2, 2);
  flowCtx.lineTo(2, sim.height - 2);
  flowCtx.moveTo(sim.width - 2, 2);
  flowCtx.lineTo(sim.width - 2, sim.height - 2);
  flowCtx.stroke();

}

function getGapGeometry() {
  const topPlateH = Math.max(4, sim.height * 0.025);
  const bottomPlateH = topPlateH;
  const gapTop = topPlateH;
  const gapBottom = sim.height - bottomPlateH;
  const gapH = gapBottom - gapTop;
  return { gapTop, gapBottom, gapH };
}

function getPhiOverlayGeometry() {
  const { gapTop, gapBottom, gapH } = getGapGeometry();
  const xMin = Math.floor(sim.width * 0.5) - 58;
  const xMax = Math.floor(sim.width * 0.5) + 58;
  const plotW = xMax - xMin;
  return { xMin, xMax, plotW, gapTop, gapBottom, gapH };
}

function drawOverlayXAxisTicks(xMin, xMax, yBase, minValue, maxValue, formatter) {
  const ticks = [0, 0.5, 1];
  flowCtx.fillStyle = "#ffffff";
  flowCtx.font = "10px system-ui, -apple-system, sans-serif";
  flowCtx.textAlign = "center";

  for (let i = 0; i < ticks.length; i += 1) {
    const t = ticks[i];
    const x = xMin + t * (xMax - xMin);
    const value = minValue + t * (maxValue - minValue);
    const label = formatter(value);
    flowCtx.fillText(label, x, yBase);
  }

  flowCtx.textAlign = "start";
}

function drawVelocityOverlay(profile) {
  const { gapTop, gapBottom, gapH } = getGapGeometry();

  const yAxisX = 16;
  const xMax = 132;
  const plotW = xMax - yAxisX;

  flowCtx.fillStyle = "rgba(255, 255, 255, 0.16)";
  flowCtx.fillRect(yAxisX - 8, gapTop, plotW + 12, gapH);

  flowCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  flowCtx.lineWidth = 1.2;
  flowCtx.beginPath();
  flowCtx.moveTo(yAxisX, gapTop);
  flowCtx.lineTo(yAxisX, gapBottom);
  flowCtx.moveTo(yAxisX, gapBottom);
  flowCtx.lineTo(xMax, gapBottom);
  flowCtx.stroke();

  const uMax = Math.max(sim.plateSpeed, 1e-8);
  flowCtx.strokeStyle = "#00d4ff";
  flowCtx.lineWidth = 2.8;
  flowCtx.beginPath();
  for (let j = 0; j < sim.ny; j += 1) {
    const yNorm = j / (sim.ny - 1);
    const xNorm = profile[j] / uMax;
    const px = yAxisX + xNorm * plotW;
    const py = gapTop + (1 - yNorm) * gapH;
    if (j === 0) {
      flowCtx.moveTo(px, py);
    } else {
      flowCtx.lineTo(px, py);
    }
  }
  flowCtx.stroke();

  if (sim.isStarted) {
    flowCtx.setLineDash([4, 4]);
    flowCtx.strokeStyle = "#ffd24a";
    flowCtx.beginPath();
    flowCtx.moveTo(yAxisX, gapBottom);
    flowCtx.lineTo(xMax, gapTop);
    flowCtx.stroke();
    flowCtx.setLineDash([]);
  }

  flowCtx.fillStyle = "#ffffff";
  flowCtx.font = "11px system-ui, -apple-system, sans-serif";
  drawOverlayXAxisTicks(yAxisX, xMax, gapBottom + 12, 0, 1, (value) => value.toFixed(1));
}

function drawStressOverlay(stress) {
  const { gapTop, gapBottom, gapH } = getGapGeometry();

  const xMin = sim.width - 132;
  const xMax = sim.width - 16;
  const plotW = xMax - xMin;

  flowCtx.fillStyle = "rgba(255, 255, 255, 0.16)";
  flowCtx.fillRect(xMin - 8, gapTop, plotW + 12, gapH);

  let tauMin = Number.POSITIVE_INFINITY;
  let tauMax = Number.NEGATIVE_INFINITY;
  let tauSum = 0;
  for (let j = 0; j < sim.ny; j += 1) {
    const tau = stress[j];
    tauMin = Math.min(tauMin, tau);
    tauMax = Math.max(tauMax, tau);
    tauSum += tau;
  }
  const tauAvg = tauSum / sim.ny;

  let tauLower = tauMin - 0.01 * Math.abs(tauMin);
  let tauUpper = tauMax + 0.01 * Math.abs(tauMax);
  if (Math.abs(tauUpper - tauLower) < 1e-8) {
    tauLower -= 1e-4;
    tauUpper += 1e-4;
  }
  const tauRange = tauUpper - tauLower;

  flowCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  flowCtx.lineWidth = 1.2;
  flowCtx.beginPath();
  flowCtx.moveTo(xMin, gapTop);
  flowCtx.lineTo(xMin, gapBottom);
  flowCtx.moveTo(xMin, gapBottom);
  flowCtx.lineTo(xMax, gapBottom);
  flowCtx.stroke();

  const avgXNorm = Math.min(Math.max((tauAvg - tauLower) / tauRange, 0), 1);
  const avgX = xMin + avgXNorm * plotW;
  flowCtx.setLineDash([3, 3]);
  flowCtx.strokeStyle = "#ffffff";
  flowCtx.lineWidth = 1.2;
  flowCtx.beginPath();
  flowCtx.moveTo(avgX, gapTop);
  flowCtx.lineTo(avgX, gapBottom);
  flowCtx.stroke();
  flowCtx.setLineDash([]);

  flowCtx.strokeStyle = "#ff6a3d";
  flowCtx.lineWidth = 2.8;
  flowCtx.beginPath();
  for (let j = 0; j < sim.ny; j += 1) {
    const yNorm = j / (sim.ny - 1);
    const xNorm = Math.min(Math.max((stress[j] - tauLower) / tauRange, 0), 1);
    const px = xMin + xNorm * plotW;
    const py = gapTop + (1 - yNorm) * gapH;
    if (j === 0) {
      flowCtx.moveTo(px, py);
    } else {
      flowCtx.lineTo(px, py);
    }
  }
  flowCtx.stroke();

  flowCtx.fillStyle = "#ffffff";
  flowCtx.font = "11px system-ui, -apple-system, sans-serif";
  drawOverlayXAxisTicks(xMin, xMax, gapBottom + 12, tauLower, tauUpper, (value) => value.toFixed(2));
}

function drawPhiOverlay(phiProfile) {
  const { xMin, xMax, plotW, gapTop, gapBottom, gapH } = getPhiOverlayGeometry();

  flowCtx.fillStyle = "rgba(255, 255, 255, 0.16)";
  flowCtx.fillRect(xMin - 8, gapTop, plotW + 12, gapH);

  flowCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  flowCtx.lineWidth = 1.2;
  flowCtx.beginPath();
  flowCtx.moveTo(xMin, gapTop);
  flowCtx.lineTo(xMin, gapBottom);
  flowCtx.moveTo(xMin, gapBottom);
  flowCtx.lineTo(xMax, gapBottom);
  flowCtx.stroke();

  flowCtx.setLineDash([5, 4]);
  flowCtx.strokeStyle = "#f2f2f2";
  flowCtx.lineWidth = 1.8;
  flowCtx.beginPath();
  for (let j = 0; j < sim.ny; j += 1) {
    const yNorm = j / (sim.ny - 1);
    const xNorm0 = Math.min(Math.max(sim.initialPhi[j] / sim.phiMaxPacking, 0), 1);
    const px0 = xMin + xNorm0 * plotW;
    const py0 = gapTop + (1 - yNorm) * gapH;
    if (j === 0) {
      flowCtx.moveTo(px0, py0);
    } else {
      flowCtx.lineTo(px0, py0);
    }
  }
  flowCtx.stroke();
  flowCtx.setLineDash([]);

  flowCtx.strokeStyle = "#b869ff";
  flowCtx.lineWidth = 2.8;
  flowCtx.beginPath();
  for (let j = 0; j < sim.ny; j += 1) {
    const yNorm = j / (sim.ny - 1);
    const xNorm = Math.min(Math.max(phiProfile[j] / sim.phiMaxPacking, 0), 1);
    const px = xMin + xNorm * plotW;
    const py = gapTop + (1 - yNorm) * gapH;
    if (j === 0) {
      flowCtx.moveTo(px, py);
    } else {
      flowCtx.lineTo(px, py);
    }
  }
  flowCtx.stroke();

  flowCtx.setLineDash([4, 4]);
  flowCtx.strokeStyle = "#ffe066";
  flowCtx.beginPath();
  flowCtx.moveTo(xMax, gapTop);
  flowCtx.lineTo(xMax, gapBottom);
  flowCtx.stroke();
  flowCtx.setLineDash([]);

  flowCtx.fillStyle = "#ffffff";
  flowCtx.font = "11px system-ui, -apple-system, sans-serif";
  drawOverlayXAxisTicks(xMin, xMax, gapBottom + 12, 0, sim.phiMaxPacking, (value) => value.toFixed(2));

}

function draw() {
  drawFlowDomain();
  const velocityProfile = averageProfile();
  const stressProfile = stressProfileFromVelocity(velocityProfile);
  drawVelocityOverlay(velocityProfile);
  drawPhiOverlay(sim.phi);
  drawStressOverlay(stressProfile);
}

function loop() {
  if (sim.isStarted) {
    let stepsDone = 0;
    let latestStep = { maxDeltaU: Number.POSITIVE_INFINITY, maxDeltaPhi: Number.POSITIVE_INFINITY };
    const minSteps = sim.stepsPerFrame;
    const maxSteps = sim.stepsPerFrame + sim.maxExtraConvergenceSteps;

    while (stepsDone < maxSteps) {
      latestStep = stepSimulation();
      stepsDone += 1;

      if (stepsDone < minSteps) {
        continue;
      }

      const convergedVelocity = latestStep.maxDeltaU < sim.convergenceTolU;
      const convergedPhi = latestStep.maxDeltaPhi < sim.convergenceTolPhi;
      if (convergedVelocity && convergedPhi) {
        break;
      }
    }
  }

  draw();
  requestAnimationFrame(loop);
}

function drawPhiAtPointerEvent(event) {
  if (sim.isStarted) {
    return;
  }

  const rect = flowCanvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) * (flowCanvas.width / rect.width);
  const py = (event.clientY - rect.top) * (flowCanvas.height / rect.height);

  const { xMin, xMax, gapTop, gapBottom, gapH } = getPhiOverlayGeometry();
  if (px < xMin || px > xMax || py < gapTop || py > gapBottom) {
    return;
  }

  const phiValue = clampPhi(((px - xMin) / (xMax - xMin)) * sim.phiMaxPacking);
  const yNorm = 1 - (py - gapTop) / gapH;
  const j = Math.min(sim.ny - 1, Math.max(0, Math.round(yNorm * (sim.ny - 1))));

  if (sim.lastDrawJ !== null && sim.lastDrawPhi !== null) {
    const j0 = sim.lastDrawJ;
    const phi0 = sim.lastDrawPhi;
    const jMin = Math.min(j0, j);
    const jMax = Math.max(j0, j);
    const span = Math.max(1, jMax - jMin);

    for (let jj = jMin; jj <= jMax; jj += 1) {
      const t = (jj - jMin) / span;
      const interpPhi = j >= j0 ? phi0 + t * (phiValue - phi0) : phiValue + t * (phi0 - phiValue);
      sim.phi[jj] = clampPhi(interpPhi);
    }
  } else {
    sim.phi[j] = phiValue;
  }

  sim.lastDrawJ = j;
  sim.lastDrawPhi = phiValue;
  updateMaterialFromPhi();
  draw();
}

flowCanvas.addEventListener("pointerdown", (event) => {
  if (sim.isStarted) {
    return;
  }
  sim.isDrawingPhi = true;
  sim.lastDrawJ = null;
  sim.lastDrawPhi = null;
  flowCanvas.setPointerCapture(event.pointerId);
  drawPhiAtPointerEvent(event);
});

flowCanvas.addEventListener("pointermove", (event) => {
  if (!sim.isDrawingPhi || sim.isStarted) {
    return;
  }
  drawPhiAtPointerEvent(event);
});

flowCanvas.addEventListener("pointerup", (event) => {
  sim.isDrawingPhi = false;
  sim.lastDrawJ = null;
  sim.lastDrawPhi = null;
  if (flowCanvas.hasPointerCapture(event.pointerId)) {
    flowCanvas.releasePointerCapture(event.pointerId);
  }
});

flowCanvas.addEventListener("pointercancel", (event) => {
  sim.isDrawingPhi = false;
  sim.lastDrawJ = null;
  sim.lastDrawPhi = null;
  if (flowCanvas.hasPointerCapture(event.pointerId)) {
    flowCanvas.releasePointerCapture(event.pointerId);
  }
});

startResetBtn.addEventListener("click", () => {
  if (!sim.isStarted) {
    sim.initialPhi = new Float64Array(sim.phi);
    updateMaterialFromPhi();
    sim.isStarted = true;
    updateStartButton();
    draw();
  } else {
    sim.isStarted = false;
    initializeField();
    updateStartButton();
    draw();
  }
});

window.addEventListener("resize", () => {
  resizeFlowCanvasToDisplay();
  updateBelowAxisLabelPositions();
  draw();
});

setup();

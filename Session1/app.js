'use strict';

// ===========================================================================
// MATRIX MATH (plain JS arrays)
// ===========================================================================

const zeros = (r, c) => Array.from({length: r}, () => Array(c).fill(0));
const randn = (r, c, s = 0.1) =>
  Array.from({length: r}, () => Array.from({length: c}, () => (Math.random() * 2 - 1) * s));
const randVec = n => Array.from({length: n}, () => Math.random() * 2 - 1);

function mm(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++)
    for (let p = 0; p < k; p++) {
      if (A[i][p] === 0) continue;
      for (let j = 0; j < n; j++) C[i][j] += A[i][p] * B[p][j];
    }
  return C;
}

const tr = A => Array.from({length: A[0].length}, (_, j) =>
  Array.from({length: A.length}, (_, i) => A[i][j]));

const addB = (Z, b) => Z.map(r => r.map((v, j) => v + b[j]));
const applyRelu = Z => Z.map(r => r.map(v => Math.max(0, v)));
const reluG    = Z => Z.map(r => r.map(v => v > 0 ? 1 : 0));
const applyS   = Z => Z.map(r => r.map(v => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, v))))));
const had      = (A, B) => A.map((r, i) => r.map((v, j) => v * B[i][j]));
const smul     = (A, s) => A.map(r => r.map(v => v * s));
const msub     = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const vsub     = (a, b) => a.map((v, i) => v - b[i]);
const vscale   = (a, s) => a.map(v => v * s);
const sumRows  = A => { const s = Array(A[0].length).fill(0); A.forEach(r => r.forEach((v,j) => s[j] += v)); return s; };
const vnorm    = v => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const vnorm2   = v => { const n = vnorm(v); return n > 1e-10 ? v.map(x => x / n) : v; };

function bceLoss(pred, y) {
  const eps = 1e-7;
  let l = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = Math.min(Math.max(pred[i][0], eps), 1 - eps);
    l -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
  }
  return l / pred.length;
}

function accuracy(pred, y) {
  let ok = 0;
  for (let i = 0; i < pred.length; i++)
    if ((pred[i][0] > 0.5 ? 1 : 0) === y[i]) ok++;
  return ok / pred.length;
}

// ===========================================================================
// NEURAL NET (binary classification, BCE + sigmoid output)
// ===========================================================================

class Net {
  constructor(layers) {
    // layers: [{in, out, act}] act = 'relu' | 'sigmoid' | 'linear'
    this.layers = layers;
    this.W = layers.map(l => {
      const s = l.act === 'relu' ? Math.sqrt(2 / l.in) : 0.5 / Math.sqrt(l.in);
      return randn(l.in, l.out, s);
    });
    this.b = layers.map(l => Array(l.out).fill(0));
    this._cache = null;
  }

  forward(X) {
    this._cache = [X];
    let A = X;
    for (let i = 0; i < this.layers.length; i++) {
      const Z = addB(mm(A, this.W[i]), this.b[i]);
      if (this.layers[i].act === 'relu')    A = applyRelu(Z);
      else if (this.layers[i].act === 'sigmoid') A = applyS(Z);
      else                                   A = Z;
      this._cache.push({Z, A});
    }
    return A;
  }

  backward(y, lr) {
    const L = this.layers.length;
    const out = this._cache[L].A;

    // Combined sigmoid+BCE gradient for output layer
    let dZ = out.map((r, i) => [(r[0] - y[i]) / y.length]);

    for (let i = L - 1; i >= 0; i--) {
      const prevA = i === 0 ? this._cache[0] : this._cache[i].A;
      const dW = mm(tr(prevA), dZ);
      const db = sumRows(dZ);

      if (i > 0) {
        const dAprev = mm(dZ, tr(this.W[i]));
        const Zprev = this._cache[i].Z;
        const actPrev = this.layers[i - 1].act;
        if (actPrev === 'relu')   dZ = had(dAprev, reluG(Zprev));
        else if (actPrev === 'linear') dZ = dAprev;
        else { // sigmoid
          const sig = this._cache[i].A;
          dZ = had(dAprev, sig.map(r => r.map(v => v * (1 - v))));
        }
      }
      this.W[i] = msub(this.W[i], smul(dW, lr));
      this.b[i] = vsub(this.b[i], vscale(db, lr));
    }
  }

  predict(X) {
    let A = X;
    for (let i = 0; i < this.layers.length; i++) {
      const Z = addB(mm(A, this.W[i]), this.b[i]);
      if (this.layers[i].act === 'relu')    A = applyRelu(Z);
      else if (this.layers[i].act === 'sigmoid') A = applyS(Z);
      else                                   A = Z;
    }
    return A;
  }

  trainStep(X, y, lr) {
    const pred = this.forward(X);
    this.backward(y, lr);
    return bceLoss(pred, y);
  }
}

// ===========================================================================
// EMBEDDING MODEL (next-token prediction)
// ===========================================================================

class EmbeddingNet {
  constructor(vocabSize, embedDim) {
    this.V = vocabSize; this.D = embedDim;
    this.E  = randn(vocabSize, embedDim, 0.5); // embedding table
    this.Wo = randn(embedDim, vocabSize, 0.1); // output projection
    this.bo = Array(vocabSize).fill(0);
  }

  forward(tokenIdxs) {
    // tokenIdxs: array of integers (N)
    this._lastIdx = tokenIdxs;
    const embs = tokenIdxs.map(i => this.E[i]); // N × D
    const logits = addB(mm(embs, this.Wo), this.bo); // N × V
    const probs = logits.map(row => {
      const m = Math.max(...row);
      const e = row.map(v => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / s);
    });
    this._lastEmbs = embs;
    this._lastProbs = probs;
    return probs;
  }

  loss(probs, targets) {
    const eps = 1e-7;
    let l = 0;
    for (let i = 0; i < targets.length; i++) l -= Math.log(Math.max(probs[i][targets[i]], eps));
    return l / targets.length;
  }

  backward(targets, lr) {
    const N = targets.length;
    // dLogits = probs - one_hot(targets), scaled by 1/N
    const dLogits = this._lastProbs.map((p, i) => p.map((v, j) => (v - (j === targets[i] ? 1 : 0)) / N));

    // Grad for Wo and bo
    const dWo = mm(tr(this._lastEmbs), dLogits); // D × V
    const dbo = sumRows(dLogits); // V

    // Grad for embeddings
    const dEmb = mm(dLogits, tr(this.Wo)); // N × D
    for (let i = 0; i < N; i++) {
      const idx = this._lastIdx[i];
      this.E[idx] = vsub(this.E[idx], vscale(dEmb[i], lr));
    }

    this.Wo = msub(this.Wo, smul(dWo, lr));
    this.bo = vsub(this.bo, vscale(dbo, lr));

    return this.loss(this._lastProbs, targets);
  }

  trainStep(tokenIdxs, targets, lr) {
    const probs = this.forward(tokenIdxs);
    return this.backward(targets, lr);
  }
}

// ===========================================================================
// PCA (2D projection of embedding table)
// ===========================================================================

function pca2D(E) {
  const n = E.length, d = E[0].length;
  const mean = Array(d).fill(0);
  E.forEach(r => r.forEach((v, j) => mean[j] += v / n));
  const Ec = E.map(r => r.map((v, j) => v - mean[j]));
  const Cov = mm(tr(Ec), Ec);

  function powerIter(C) {
    let v = vnorm2(randVec(C.length));
    for (let iter = 0; iter < 200; iter++) {
      v = vnorm2(C.map(row => row.reduce((a, c, j) => a + c * v[j], 0)));
    }
    const lam = v.reduce((a, vi, i) => a + vi * C[i].reduce((b, c, j) => b + c * v[j], 0), 0);
    return {v, lam};
  }

  const {v: v1, lam: l1} = powerIter(Cov);
  const Cov2 = Cov.map((row, i) => row.map((c, j) => c - l1 * v1[i] * v1[j]));
  const {v: v2} = powerIter(Cov2);

  return Ec.map(r => [
    r.reduce((a, v, j) => a + v * v1[j], 0),
    r.reduce((a, v, j) => a + v * v2[j], 0)
  ]);
}

// ===========================================================================
// DATA GENERATION
// ===========================================================================

function makeRings(n = 300, noise = 0.12) {
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const cls = i < n / 2 ? 0 : 1;
    const r = cls === 0 ? (0.2 + Math.random() * 0.25) : (0.55 + Math.random() * 0.25);
    const theta = Math.random() * Math.PI * 2;
    X.push([Math.cos(theta) * r + (Math.random() - 0.5) * noise,
            Math.sin(theta) * r + (Math.random() - 0.5) * noise]);
    y.push(cls);
  }
  return {X, y};
}

function makeRingsN(n, noise = 0.12, seed = 42) {
  // Simple seeded-ish version for reproducibility
  const rng = (() => { let s = seed; return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; }; })();
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const r = cls === 0 ? (0.2 + rng() * 0.25) : (0.55 + rng() * 0.25);
    const theta = rng() * Math.PI * 2;
    X.push([Math.cos(theta) * r + (rng() - 0.5) * noise,
            Math.sin(theta) * r + (rng() - 0.5) * noise]);
    y.push(cls);
  }
  return {X, y};
}

// Toy language for embeddings
const VOCAB  = ['cat','dog','cow','apple','mango','eat','chase','see'];
const CATS   = {cat:0, dog:0, cow:0, apple:1, mango:1, eat:2, chase:2, see:2}; // 0=animal,1=fruit,2=verb
const ANIMALS = [0,1,2], FRUITS = [3,4], VERBS = [5,6,7];

function makeSentences(n = 600) {
  const pairs = [];
  for (let i = 0; i < n; i++) {
    const a1 = ANIMALS[Math.floor(Math.random() * ANIMALS.length)];
    const v  = VERBS[Math.floor(Math.random() * VERBS.length)];
    const obj = Math.random() < 0.5
      ? ANIMALS[Math.floor(Math.random() * ANIMALS.length)]
      : FRUITS[Math.floor(Math.random() * FRUITS.length)];
    // pairs: (a1→v), (v→obj)
    pairs.push([a1, v]);
    pairs.push([v, obj]);
  }
  return pairs; // array of [input_idx, target_idx]
}

// ===========================================================================
// VISUALIZATION UTILITIES
// ===========================================================================

function drawDecisionBoundary(canvas, predictFn, data, resolution = 70) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  if (W === 0 || H === 0) return;

  // Build grid
  const R = resolution;
  const batchX = [];
  for (let py = 0; py < R; py++)
    for (let px = 0; px < R; px++) {
      const x = -1 + (px / (R - 1)) * 2;
      const y = -1 + (py / (R - 1)) * 2;
      batchX.push([x, y]);
    }

  const preds = predictFn(batchX);
  const imgData = ctx.createImageData(W, H);

  for (let py = 0; py < R; py++) {
    for (let px = 0; px < R; px++) {
      const p = preds[py * R + px][0];
      // Map pixel to actual canvas coords
      const cx0 = Math.floor((px / R) * W);
      const cy0 = Math.floor((py / R) * H);
      const cx1 = Math.floor(((px + 1) / R) * W);
      const cy1 = Math.floor(((py + 1) / R) * H);

      const r0 = p < 0.5 ? Math.round(239 * (1 - 2 * p)) : 0;
      const g0 = 0;
      const b0 = p > 0.5 ? Math.round(200 * (2 * p - 1)) : 0;
      const base_r = 14, base_g = 20, base_b = 40;
      const alpha = Math.abs(p - 0.5) * 1.8;
      const rr = Math.round(base_r + r0 * alpha);
      const gg = Math.round(base_g);
      const bb = Math.round(base_b + b0 * alpha);

      for (let cy = cy0; cy < cy1; cy++) {
        for (let cx = cx0; cx < cx1; cx++) {
          const idx = (cy * W + cx) * 4;
          imgData.data[idx]     = rr;
          imgData.data[idx + 1] = gg;
          imgData.data[idx + 2] = bb;
          imgData.data[idx + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // Draw boundary line (0.5 contour approximation)
  // ...omit for speed, the color gradient shows it

  // Draw data points
  if (data) {
    const {X, y} = data;
    for (let i = 0; i < X.length; i++) {
      const px = ((X[i][0] + 1) / 2) * W;
      const py = ((X[i][1] + 1) / 2) * H;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = y[i] === 0 ? 'rgba(248,113,113,0.9)' : 'rgba(139,92,246,0.9)';
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 0.5;
      ctx.fill();
      ctx.stroke();
    }
  }
}

function drawLossChart(canvas, datasets) {
  // datasets: [{label, color, values, dashed}]
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {l: 42, r: 20, t: 16, b: 32};

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(0,0,0,0)';

  const allVals = datasets.flatMap(d => d.values);
  if (allVals.length === 0) return;
  const maxV = Math.min(Math.max(...allVals) * 1.1, 3);
  const minV = 0;
  const maxLen = Math.max(...datasets.map(d => d.values.length));

  const toX = i => pad.l + (i / (maxLen - 1)) * (W - pad.l - pad.r);
  const toY = v => pad.t + (1 - (v - minV) / (maxV - minV)) * (H - pad.t - pad.b);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + (g / 4) * (H - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    const val = (maxV - minV) * (1 - g / 4) + minV;
    ctx.fillStyle = 'rgba(100,116,139,0.8)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(2), pad.l - 4, y + 3);
  }

  // Axis labels
  ctx.fillStyle = 'rgba(100,116,139,0.6)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Epoch', W / 2, H - 4);

  // Plot each dataset
  for (const ds of datasets) {
    if (ds.values.length < 2) continue;
    ctx.beginPath();
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = 2;
    if (ds.dashed === 1)      ctx.setLineDash([10, 4]);
    else if (ds.dashed === 2) ctx.setLineDash([5, 3]);
    else if (ds.dashed === 3) ctx.setLineDash([2, 3]);
    else ctx.setLineDash([]);
    ctx.shadowColor = ds.color;
    ctx.shadowBlur = 4;

    for (let i = 0; i < ds.values.length; i++) {
      const x = toX(i), y = toY(ds.values[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
  }

  // Legend
  let lx = pad.l;
  for (const ds of datasets) {
    ctx.fillStyle = ds.color;
    ctx.fillRect(lx, H - pad.b + 8, 18, 3);
    ctx.fillStyle = 'rgba(148,163,184,0.9)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(ds.label, lx + 22, H - pad.b + 13);
    lx += ctx.measureText(ds.label).width + 40;
  }
}

function drawEmbedding(canvas, coords2D, labels, colors, tokenNames) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!coords2D || coords2D.length === 0) return;

  const xs = coords2D.map(c => c[0]), ys = coords2D.map(c => c[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = 50;

  const toSX = x => pad + ((x - xmin) / (xmax - xmin + 1e-8)) * (W - 2 * pad);
  const toSY = y => pad + ((y - ymin) / (ymax - ymin + 1e-8)) * (H - 2 * pad);

  // Draw axes
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  const cx = toSX((xmin + xmax) / 2), cy = toSY((ymin + ymax) / 2);
  ctx.beginPath(); ctx.moveTo(pad, cy); ctx.lineTo(W - pad, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, H - pad); ctx.stroke();

  // Draw cluster halos
  const catGroups = {};
  coords2D.forEach((c, i) => {
    const cat = labels[i];
    if (!catGroups[cat]) catGroups[cat] = [];
    catGroups[cat].push({x: toSX(c[0]), y: toSY(c[1])});
  });

  for (const cat in catGroups) {
    const pts = catGroups[cat];
    const cx2 = pts.reduce((a, p) => a + p.x, 0) / pts.length;
    const cy2 = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    const r = Math.max(30, Math.max(...pts.map(p => Math.hypot(p.x - cx2, p.y - cy2))) + 20);
    const grad = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, r);
    grad.addColorStop(0, colors[cat].replace(')', ', 0.12)').replace('rgb', 'rgba'));
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2);
    ctx.fillStyle = grad; ctx.fill();
  }

  // Draw nodes
  coords2D.forEach((c, i) => {
    const x = toSX(c[0]), y = toSY(c[1]);
    const col = colors[labels[i]];

    ctx.beginPath(); ctx.arc(x, y, 18, 0, Math.PI * 2);
    ctx.fillStyle = col.replace(')', ', 0.15)').replace('rgb', 'rgba');
    ctx.fill();

    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.shadowColor = col; ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '500 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(tokenNames[i], x, y + 24);
  });

  // Category labels
  const catLabels = ['Animals', 'Fruits', 'Verbs'];
  for (const cat in catGroups) {
    const pts = catGroups[cat];
    const cx2 = pts.reduce((a, p) => a + p.x, 0) / pts.length;
    const cy2 = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillStyle = colors[cat];
    ctx.textAlign = 'center';
    ctx.fillText(catLabels[Number(cat)], cx2, cy2 - 28);
  }
}

// ===========================================================================
// ASYNC HELPERS
// ===========================================================================

const nextFrame = () => new Promise(r => requestAnimationFrame(r));
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ===========================================================================
// EXPERIMENT S1-1: ACTIVATIONS
// ===========================================================================

let s1Running = false;

async function runS1() {
  if (s1Running) return;
  s1Running = true;
  const btn = document.getElementById('s1-run');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Training…';

  const statsEl = document.getElementById('s1-stats');
  statsEl.style.display = 'flex';

  const data = makeRings(300, 0.12);
  const {X, y} = data;

  const netLin = new Net([{in: 2, out: 1, act: 'sigmoid'}]);
  const netRelu = new Net([{in: 2, out: 32, act: 'relu'}, {in: 32, out: 1, act: 'sigmoid'}]);

  const lossLin = [], lossRelu = [];
  const EPOCHS = 1000;
  const LR = 0.08;
  const UPDATE_EVERY = 15;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    lossLin.push(netLin.trainStep(X, y, LR));
    lossRelu.push(netRelu.trainStep(X, y, LR));

    if (epoch % UPDATE_EVERY === 0 || epoch === EPOCHS - 1) {
      const accLin  = (accuracy(netLin.predict(X), y) * 100).toFixed(1) + '%';
      const accRelu = (accuracy(netRelu.predict(X), y) * 100).toFixed(1) + '%';
      document.getElementById('s1-acc-linear').textContent = accLin;
      document.getElementById('s1-acc-relu').textContent   = accRelu;
      document.getElementById('s1-epoch').textContent      = epoch + 1;
      document.getElementById('s1-prog-linear').style.width = ((epoch + 1) / EPOCHS * 100) + '%';
      document.getElementById('s1-prog-relu').style.width   = ((epoch + 1) / EPOCHS * 100) + '%';

      drawDecisionBoundary(document.getElementById('s1-canvas-linear'),
        bX => netLin.predict(bX), data);
      drawDecisionBoundary(document.getElementById('s1-canvas-relu'),
        bX => netRelu.predict(bX), data);
      drawLossChart(document.getElementById('s1-loss-chart'), [
        {label: 'Linear Loss', color: '#ef4444', values: lossLin},
        {label: 'ReLU Loss',   color: '#8b5cf6', values: lossRelu}
      ]);

      await nextFrame();
    }
  }

  btn.innerHTML = '<span class="btn-icon">✓</span> Done — try again';
  btn.disabled = false;
  s1Running = false;
}

// ===========================================================================
// EXPERIMENT S1-2: DEPTH
// ===========================================================================

let s2Running = false;

async function runS2() {
  if (s2Running) return;
  s2Running = true;
  const btn = document.getElementById('s2-run');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Training…';
  document.getElementById('s2-stats').style.display = 'flex';

  const data = makeRings(300, 0.12);
  const {X, y} = data;

  const net1 = new Net([{in:2, out:1, act:'sigmoid'}]);
  const net5L = new Net([
    {in:2, out:4, act:'linear'}, {in:4, out:4, act:'linear'},
    {in:4, out:4, act:'linear'}, {in:4, out:4, act:'linear'},
    {in:4, out:1, act:'sigmoid'}
  ]);
  const net5R = new Net([
    {in:2, out:16, act:'relu'}, {in:16, out:16, act:'relu'},
    {in:16, out:8, act:'relu'}, {in:8, out:4, act:'relu'},
    {in:4, out:1, act:'sigmoid'}
  ]);

  const EPOCHS = 1200, LR = 0.08;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    net1.trainStep(X, y, LR);
    net5L.trainStep(X, y, LR);
    net5R.trainStep(X, y, LR);

    if (epoch % 20 === 0 || epoch === EPOCHS - 1) {
      const a1  = (accuracy(net1.predict(X), y)   * 100).toFixed(1) + '%';
      const a5L = (accuracy(net5L.predict(X), y) * 100).toFixed(1) + '%';
      const a5R = (accuracy(net5R.predict(X), y) * 100).toFixed(1) + '%';
      document.getElementById('s2-acc-1').textContent  = a1;
      document.getElementById('s2-acc-5').textContent  = a5L;
      document.getElementById('s2-acc-5r').textContent = a5R;

      drawDecisionBoundary(document.getElementById('s2-canvas-1'),
        bX => net1.predict(bX), data, 55);
      drawDecisionBoundary(document.getElementById('s2-canvas-5'),
        bX => net5L.predict(bX), data, 55);
      drawDecisionBoundary(document.getElementById('s2-canvas-5r'),
        bX => net5R.predict(bX), data, 55);

      await nextFrame();
    }
  }

  // Matrix collapse demonstration
  // Multiply all 5 weight matrices of net5L
  let Wcollapsed = net5L.W[0];
  for (let i = 1; i < net5L.W.length; i++) Wcollapsed = mm(Wcollapsed, net5L.W[i]);
  // Wcollapsed is now 2 × 1

  const matBox = document.getElementById('s2-matrix-box');
  matBox.style.display = 'block';

  const matSizes = net5L.W.map((w, i) => `${w.length}×${w[0].length}`);
  const eq = document.getElementById('s2-matrix-eq');
  let html = '';
  matSizes.forEach((s, i) => {
    html += `<div class="matrix-block" style="border-color:rgba(245,158,11,0.3)">W${i+1}<br><span style="color:#f59e0b;font-size:0.8em">${s}</span></div>`;
    if (i < matSizes.length - 1) html += `<div class="matrix-op">×</div>`;
  });
  html += `<div class="matrix-op" style="font-size:1.2em">=</div>`;
  html += `<div class="matrix-block" style="border-color:rgba(6,182,212,0.5);color:#06b6d4">
    W<sub>eff</sub><br>
    <span style="font-size:0.75em;color:#94a3b8">2×1 → same rank</span><br>
    <span style="color:#06b6d4">[${Wcollapsed[0][0].toFixed(3)}]</span><br>
    <span style="color:#06b6d4">[${Wcollapsed[1][0].toFixed(3)}]</span>
  </div>`;
  eq.innerHTML = html;

  btn.innerHTML = '<span class="btn-icon">✓</span> Done';
  btn.disabled = false;
  s2Running = false;
}

// ===========================================================================
// EXPERIMENT S1-3: EMBEDDINGS
// ===========================================================================

let s3Running = false;

function updateNeighbors(embedNet) {
  const E = embedNet.E;
  const container = document.getElementById('s3-neighbors');
  const catColors = ['rgb(96,165,250)', 'rgb(52,211,153)', 'rgb(248,113,113)'];
  const catNames  = ['animal', 'fruit', 'verb'];

  let html = '';
  VOCAB.forEach((token, i) => {
    // Find nearest neighbor
    let bestSim = -Infinity, bestJ = -1;
    VOCAB.forEach((_, j) => {
      if (j === i) return;
      const dot = E[i].reduce((a, v, k) => a + v * E[j][k], 0);
      const ni = vnorm(E[i]), nj = vnorm(E[j]);
      const cos = dot / (ni * nj + 1e-8);
      if (cos > bestSim) { bestSim = cos; bestJ = j; }
    });
    const cat = CATS[token];
    const neighborCat = CATS[VOCAB[bestJ]];
    const correct = cat === neighborCat;
    html += `<div class="neighbor-item">
      <span class="neighbor-token" style="color:${catColors[cat]}">${token}</span>
      <span style="font-size:0.65rem;color:#475569">→</span>
      <span class="neighbor-token" style="color:${catColors[neighborCat]}">${VOCAB[bestJ]}</span>
      <span class="neighbor-cat" style="background:${correct?'rgba(16,185,129,0.15)':'rgba(239,68,68,0.1)'};color:${correct?'#10b981':'#ef4444'}">${correct?'✓':'✗'}</span>
    </div>`;
  });
  container.innerHTML = html;
}

async function runS3() {
  if (s3Running) return;
  s3Running = true;
  const btn = document.getElementById('s3-run');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Training…';
  document.getElementById('s3-stats').style.display = 'flex';

  const pairs = makeSentences(800);
  const inputs  = pairs.map(p => p[0]);
  const targets = pairs.map(p => p[1]);

  const embedNet = new EmbeddingNet(VOCAB.length, 16);
  const lossHistory = [];
  const EPOCHS = 300, LR = 0.04;
  const catColors = ['rgb(96,165,250)', 'rgb(52,211,153)', 'rgb(248,113,113)'];
  const tokenLabels = VOCAB.map(t => CATS[t]);

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Shuffle
    for (let i = pairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [inputs[i], inputs[j]] = [inputs[j], inputs[i]];
      [targets[i], targets[j]] = [targets[j], targets[i]];
    }
    const l = embedNet.trainStep(inputs, targets, LR);
    lossHistory.push(l);

    if (epoch % 5 === 0 || epoch === EPOCHS - 1) {
      document.getElementById('s3-loss').textContent  = l.toFixed(4);
      document.getElementById('s3-epoch').textContent = epoch + 1;

      const coords = pca2D(embedNet.E);
      drawEmbedding(
        document.getElementById('s3-canvas'),
        coords, tokenLabels, catColors, VOCAB
      );
      drawLossChart(document.getElementById('s3-loss-chart'), [
        {label: 'CE Loss', color: '#10b981', values: lossHistory}
      ]);
      if (epoch % 20 === 0) updateNeighbors(embedNet);
      await nextFrame();
    }
  }

  updateNeighbors(embedNet);
  btn.innerHTML = '<span class="btn-icon">✓</span> Done';
  btn.disabled = false;
  s3Running = false;
}

// ===========================================================================
// EXPERIMENT S1-4: GENERALIZATION
// ===========================================================================

let s4Running = false;

async function runS4() {
  if (s4Running) return;
  s4Running = true;
  const btn = document.getElementById('s4-run');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Training…';
  document.getElementById('s4-stats').style.display = 'flex';

  const TRAIN_SIZES = [20, 200, 2000];
  const TEST_N = 500;
  const EPOCHS = 600;
  const LR = 0.06;

  // Fixed test set
  const testData  = makeRingsN(TEST_N, 0.12, 99);

  const runs = TRAIN_SIZES.map((n, idx) => ({
    n,
    data: makeRingsN(n, 0.12, idx * 7 + 3),
    net: new Net([{in:2,out:64,act:'relu'},{in:64,out:64,act:'relu'},{in:64,out:1,act:'sigmoid'}]),
    trainLoss: [],
    testLoss: []
  }));

  const colors = ['#ef4444', '#f59e0b', '#10b981'];
  const canvases = ['s4-boundary-20', 's4-boundary-200', 's4-boundary-2000'];

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (let ri = 0; ri < runs.length; ri++) {
      const {data, net} = runs[ri];
      const tl = net.trainStep(data.X, data.y, LR);
      const pTest = net.predict(testData.X);
      const testL = bceLoss(pTest, testData.y);
      runs[ri].trainLoss.push(tl);
      runs[ri].testLoss.push(testL);
    }

    if (epoch % 25 === 0 || epoch === EPOCHS - 1) {
      for (let ri = 0; ri < runs.length; ri++) {
        const {data, net} = runs[ri];
        drawDecisionBoundary(document.getElementById(canvases[ri]),
          bX => net.predict(bX), data, 50);
      }

      // Gap stats
      for (let ri = 0; ri < runs.length; ri++) {
        const tl = runs[ri].trainLoss;
        const tstl = runs[ri].testLoss;
        const gap = (tstl[tstl.length-1] - tl[tl.length-1]).toFixed(3);
        const ids = ['s4-gap-20', 's4-gap-200', 's4-gap-2000'];
        document.getElementById(ids[ri]).textContent = gap;
      }

      // Combined chart
      const datasets = [];
      runs.forEach((r, ri) => {
        datasets.push({label: `Train n=${r.n}`, color: colors[ri], values: r.trainLoss, dashed: 0});
        datasets.push({label: `Test n=${r.n}`,  color: colors[ri], values: r.testLoss,  dashed: ri + 1});
      });
      drawLossChart(document.getElementById('s4-gap-chart'), datasets);

      await nextFrame();
    }
  }

  btn.innerHTML = '<span class="btn-icon">✓</span> Done';
  btn.disabled = false;
  s4Running = false;
}

// ===========================================================================
// HERO CANVAS ANIMATION
// ===========================================================================

function initHero() {
  const canvas = document.getElementById('hero-canvas');
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const LAYERS = [2, 5, 5, 3, 1];
  const particles = [];
  let time = 0;

  function nodePos(layer, node) {
    const lx = 0.1 + (layer / (LAYERS.length - 1)) * 0.8;
    const total = LAYERS[layer];
    const ly = 0.5 + ((node - (total - 1) / 2) / Math.max(total - 1, 1)) * 0.55;
    return {x: lx * canvas.width, y: ly * canvas.height};
  }

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.layer = 0;
      this.fromNode = Math.floor(Math.random() * LAYERS[0]);
      this.toNode = Math.floor(Math.random() * LAYERS[1]);
      this.t = Math.random();
      this.speed = 0.005 + Math.random() * 0.008;
      this.opacity = 0.3 + Math.random() * 0.5;
      const r = Math.random();
      this.color = r < 0.33 ? '#8b5cf6' : r < 0.66 ? '#06b6d4' : '#10b981';
    }
    update() {
      this.t += this.speed;
      if (this.t >= 1) {
        this.layer++;
        if (this.layer >= LAYERS.length - 1) { this.reset(); return; }
        this.fromNode = this.toNode;
        this.toNode = Math.floor(Math.random() * LAYERS[this.layer + 1]);
        this.t = 0;
      }
    }
    draw() {
      const p1 = nodePos(this.layer, this.fromNode);
      const p2 = nodePos(this.layer + 1, this.toNode);
      const x = p1.x + (p2.x - p1.x) * this.t;
      const y = p1.y + (p2.y - p1.y) * this.t;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = this.color.replace(')', `, ${this.opacity})`).replace('#', 'rgba(') ;
      // Actually just use globalAlpha
      ctx.globalAlpha = this.opacity;
      ctx.fillStyle = this.color;
      ctx.shadowColor = this.color; ctx.shadowBlur = 8;
      ctx.fill();
      ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    }
  }

  for (let i = 0; i < 80; i++) {
    const p = new Particle();
    p.t = Math.random(); p.layer = Math.floor(Math.random() * (LAYERS.length - 1));
    p.fromNode = Math.floor(Math.random() * LAYERS[p.layer]);
    p.toNode = Math.floor(Math.random() * LAYERS[p.layer + 1]);
    particles.push(p);
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    time += 0.01;

    // Draw edges
    for (let l = 0; l < LAYERS.length - 1; l++) {
      for (let n = 0; n < LAYERS[l]; n++) {
        for (let m = 0; m < LAYERS[l + 1]; m++) {
          const p1 = nodePos(l, n), p2 = nodePos(l + 1, m);
          ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = 'rgba(148,163,184,0.06)';
          ctx.lineWidth = 0.8; ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (let l = 0; l < LAYERS.length; l++) {
      for (let n = 0; n < LAYERS[l]; n++) {
        const {x, y} = nodePos(l, n);
        const pulse = 0.7 + 0.3 * Math.sin(time + l * 1.2 + n * 0.8);
        const r = 5 * pulse;
        ctx.beginPath(); ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(139,92,246,0.08)'; ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(139,92,246,0.6)';
        ctx.shadowColor = '#8b5cf6'; ctx.shadowBlur = 10 * pulse;
        ctx.fill(); ctx.shadowBlur = 0;
      }
    }

    // Update + draw particles
    particles.forEach(p => { p.update(); p.draw(); });

    requestAnimationFrame(draw);
  }
  draw();
}

// ===========================================================================
// SCROLL ANIMATIONS
// ===========================================================================

function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity = '1';
        e.target.style.transform = 'translateY(0)';
      }
    });
  }, {threshold: 0.1});

  document.querySelectorAll('.section-header, .canvas-card, .insight-box, .vocab-display, .matrix-collapse-box').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });
}

// ===========================================================================
// SECTION COLOR INJECTION
// ===========================================================================

function initSectionColors() {
  document.querySelectorAll('.experiment-section').forEach(sec => {
    const c = sec.dataset.color;
    if (c) sec.style.setProperty('--c-section', c);
  });
}

// ===========================================================================
// NAV ACTIVE STATE
// ===========================================================================

function initNav() {
  const sections = ['hero', 's1', 's2', 's3', 's4'];
  const links = document.querySelectorAll('.nav-link');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const id = e.target.id;
        links.forEach(l => {
          l.style.color = l.getAttribute('href') === '#' + id
            ? '#e2e8f0' : '';
          l.style.background = l.getAttribute('href') === '#' + id
            ? 'rgba(255,255,255,0.07)' : '';
        });
      }
    });
  }, {threshold: 0.4});

  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

// ===========================================================================
// INIT
// ===========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initHero();
  initScrollAnimations();
  initSectionColors();
  initNav();

  document.getElementById('s1-run').addEventListener('click', runS1);
  document.getElementById('s2-run').addEventListener('click', runS2);
  document.getElementById('s3-run').addEventListener('click', runS3);
  document.getElementById('s4-run').addEventListener('click', runS4);

  // Draw empty placeholders
  ['s1-canvas-linear','s1-canvas-relu'].forEach(id => {
    const c = document.getElementById(id);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0d1120';
    ctx.fillRect(0,0,c.width,c.height);
    ctx.fillStyle = 'rgba(100,116,139,0.3)';
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Click "Run Experiment" to train', c.width/2, c.height/2);
  });
  ['s2-canvas-1','s2-canvas-5','s2-canvas-5r',
   's4-boundary-20','s4-boundary-200','s4-boundary-2000'].forEach(id => {
    const c = document.getElementById(id);
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0d1120';
    ctx.fillRect(0,0,c.width,c.height);
  });
  ['s3-canvas'].forEach(id => {
    const c = document.getElementById(id);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0d1120';
    ctx.fillRect(0,0,c.width,c.height);
    ctx.fillStyle = 'rgba(100,116,139,0.3)';
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Train to see clusters emerge', c.width/2, c.height/2 + 20);
  });
});

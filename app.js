/* Concrete defect recognition - live demo (runs fully in the browser with ONNX Runtime Web).

   What happens when a photo is chosen:
   1. prepare(): the photo is made into the same 224 x 224 input as in training
      (whole photo squeezed to 256 x 256 with bicubic resizing like Pillow, then the centre 224 x 224,
       RGB values / 255; the ImageNet normalisation is inside the .onnx file).
   2. Both networks of the chosen task run on it (own CNN and ResNet-18).
   3. The probabilities are shown as bars (the tick on a bar = decision threshold tuned on validation photos).
*/
"use strict";

ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;   // the WebAssembly engine is in ort/ (works offline)
ort.env.wasm.numThreads = 1;       // one thread: works on every simple web server

// ---------------------------------------------------------------- the models (numbers from the model cards)
const MODES = {
  real: {
    classes: ["crack", "rust", "spalling"],
    kind: "sigmoid",               // one independent probability per label
    models: [
      // real test mAP 0.746; thresholds tuned on real_val (crack, rust, spalling)
      { name: "ResNet-18", file: "models/resnet18_real.onnx", thresholds: [0.35, 0.45, 0.30] },
      // real test mAP 0.704
      { name: "Own CNN", file: "models/own_cnn_real.onnx", thresholds: [0.25, 0.35, 0.30] },
    ],
  },
  synthetic: {
    classes: ["pristine", "crack", "rust", "spalling"],
    kind: "softmax",               // exactly one class, probabilities add up to 1
    models: [
      { name: "ResNet-18", file: "models/resnet18_finetuned_synthetic.onnx" },   // synthetic test accuracy 0.866
      { name: "Own CNN", file: "models/own_cnn_synthetic.onnx" },                // synthetic test accuracy 0.853
    ],
  },
};

const $ = (id) => document.getElementById(id);
const sessions = {};               // loaded networks, so each file is loaded only once
let mode = "real";
let current = null;                // { tensor, truth }

// ---------------------------------------------------------------- 1. image preparation (as in training)
function cubic(x) {                // Pillow's bicubic filter (a = -0.5)
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function resizeWeights(inSize, outSize) {   // same coefficients as Pillow (antialiased when shrinking)
  const scale = inSize / outSize, filterScale = Math.max(scale, 1), support = 2 * filterScale;
  const starts = [], weights = [];
  for (let o = 0; o < outSize; o++) {
    const centre = (o + 0.5) * scale;
    const lo = Math.max(Math.trunc(centre - support + 0.5), 0);
    const hi = Math.min(Math.trunc(centre + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let i = lo; i < hi; i++) { const v = cubic((i - centre + 0.5) / filterScale); w.push(v); sum += v; }
    starts.push(lo);
    weights.push(w.map((v) => v / sum));
  }
  return { starts, weights };
}

function resizeBicubic(rgba, w, h, outW, outH) {   // RGBA bytes -> RGB bytes, horizontal then vertical pass
  const hw = resizeWeights(w, outW), vw = resizeWeights(h, outH);
  const tmp = new Uint8ClampedArray(outW * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < outW; x++) {
      const s = hw.starts[x], k = hw.weights[x];
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < k.length; i++) {
        const p = (y * w + s + i) * 4;
        r += rgba[p] * k[i]; g += rgba[p + 1] * k[i]; b += rgba[p + 2] * k[i];
      }
      const q = (y * outW + x) * 3;
      tmp[q] = Math.round(r); tmp[q + 1] = Math.round(g); tmp[q + 2] = Math.round(b);   // clamped to 0..255
    }
  }
  const out = new Uint8ClampedArray(outW * outH * 3);
  for (let y = 0; y < outH; y++) {
    const s = vw.starts[y], k = vw.weights[y];
    for (let x = 0; x < outW; x++) {
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < k.length; i++) {
        const p = ((s + i) * outW + x) * 3;
        r += tmp[p] * k[i]; g += tmp[p + 1] * k[i]; b += tmp[p + 2] * k[i];
      }
      const q = (y * outW + x) * 3;
      out[q] = Math.round(r); out[q + 1] = Math.round(g); out[q + 2] = Math.round(b);
    }
  }
  return out;
}

async function prepare(blob) {
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  let w = bitmap.width, h = bitmap.height;
  // Training read JPEGs with Pillow's draft mode (decoded 2x, 4x or 8x smaller, but still >= 256 px).
  // Do the same; also shrink very large PNGs first so the page stays fast.
  let f = 1;
  if (blob.type === "image/jpeg") {
    const fit = Math.min(Math.floor(w / 256), Math.floor(h / 256));
    for (const c of [8, 4, 2]) { if (fit >= c) { f = c; break; } }
  } else if (Math.min(w, h) > 2048) {
    f = Math.min(w, h) / 1024;          // only to keep very large PNG files fast
  }
  const dw = Math.ceil(w / f), dh = Math.ceil(h / f);
  const canvas = document.createElement("canvas");
  canvas.width = dw; canvas.height = dh;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, dw, dh);
  bitmap.close();
  const rgb256 = resizeBicubic(ctx.getImageData(0, 0, dw, dh).data, dw, dh, 256, 256);

  // centre 224 x 224 crop -> float32, channels first, values 0..1
  const S = 224, off = 16, data = new Float32Array(3 * S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const p = ((y + off) * 256 + (x + off)) * 3, i = y * S + x;
      data[i] = rgb256[p] / 255; data[S * S + i] = rgb256[p + 1] / 255; data[2 * S * S + i] = rgb256[p + 2] / 255;
    }
  }
  return new ort.Tensor("float32", data, [1, 3, S, S]);
}

// ---------------------------------------------------------------- 2. running the networks
async function getSession(file) {
  if (!sessions[file]) sessions[file] = ort.InferenceSession.create(file, { executionProviders: ["wasm"] });
  return sessions[file];
}

async function predict(model, tensor) {
  const session = await getSession(model.file);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  return { probs: Array.from(out[session.outputNames[0]].data) };   // the .onnx already outputs probabilities
}

// ---------------------------------------------------------------- 3. showing the result
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);

function card(model, cfg, res) {
  const el = document.createElement("article");
  el.className = "card";
  const top = res.probs.indexOf(Math.max(...res.probs));
  let verdict;
  if (cfg.kind === "sigmoid") {
    const hits = cfg.classes.filter((c, k) => res.probs[k] >= model.thresholds[k]);
    verdict = hits.length ? `<p class="verdict">${cap(hits.join(" + "))}</p>`
                          : `<p class="verdict none">No defect detected</p>`;
  } else {
    verdict = `<p class="verdict">${cap(cfg.classes[top])}</p>`;
  }
  const rows = cfg.classes.map((c, k) => {
    const p = res.probs[k];
    const hit = cfg.kind === "sigmoid" ? p >= model.thresholds[k] : k === top;
    const tick = cfg.kind === "sigmoid" ? `<span class="tick" style="left:${model.thresholds[k] * 100}%"></span>` : "";
    return `<div class="row${hit ? " hit" : ""}"><span class="name">${c}</span>` +
      `<span class="bar"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span>${tick}</span>` +
      `<span class="pct">${(p * 100).toFixed(0)} %</span></div>`;
  }).join("");
  el.innerHTML = `<h2>${model.name}</h2>${verdict}${rows}`;
  return el;
}

async function run() {
  if (!current) return;
  const cfg = MODES[mode];
  setStatus("Analysing …");
  const cards = $("cards");
  cards.innerHTML = "";
  try {
    for (const model of cfg.models) {
      const res = await predict(model, current.tensor);
      cards.appendChild(card(model, cfg, res));
    }
    setStatus("");
  } catch (err) {
    setStatus("Something went wrong: " + err.message, true);
  }
}

async function useImage(blob, truth) {
  $("result").hidden = false;
  $("photo").src = URL.createObjectURL(blob);
  const t = $("truth");
  if (truth) {
    t.textContent = "Expert label: " + (truth.labels.length ? truth.labels.join(" + ") : "no defect");   // dacl10k annotation = correct answer
    t.hidden = false;
  } else {
    t.hidden = true;
  }
  setStatus("Analysing …");
  current = { tensor: await prepare(blob), truth };
  await run();
}

function setStatus(text, error = false) {
  const s = $("status");
  s.textContent = text;
  s.classList.toggle("error", error);
}

function setMode(m) {
  mode = m;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  MODES[m].models.forEach((model) => getSession(model.file));   // start loading in the background
  run();
}

// ---------------------------------------------------------------- page set-up
async function init() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  const fileInput = $("file"), drop = $("drop");
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) { clearActive(); useImage(fileInput.files[0], null); } });
  drop.tabIndex = 0;
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
    if (f) { clearActive(); useImage(f, null); }
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData || {}).items || []].find((x) => x.type.startsWith("image/"));
    if (item) { clearActive(); useImage(item.getAsFile(), null); }
  });

  const examples = await (await fetch("examples/examples.json")).json();
  const box = $("examples");
  examples.forEach((ex) => {
    const b = document.createElement("button");
    b.type = "button";
    const label = ex.labels.length ? ex.labels.join(" + ") : "none";
    b.innerHTML = `<img src="${ex.file}" alt="Example: ${label}">`;
    b.title = "Example: " + label;
    b.addEventListener("click", async () => {
      clearActive(); b.classList.add("active");
      useImage(await (await fetch(ex.file)).blob(), ex);
    });
    box.appendChild(b);
  });

  setMode("real");
  try {
    await Promise.all(MODES.real.models.map((m) => getSession(m.file)));
    setStatus("");
  } catch (err) {
    setStatus("The networks could not be loaded (" + err.message + "). Offline: start the page with start_demo.bat.", true);
  }
  const params = new URLSearchParams(location.search);
  if (params.has("mode")) setMode(params.get("mode") === "synthetic" ? "synthetic" : "real");
  if (params.has("example")) box.children[Number(params.get("example")) - 1]?.click();   // e.g. ?example=6
  if (params.has("selftest")) selfTest(examples);
}

function clearActive() { document.querySelectorAll(".examples button").forEach((b) => b.classList.remove("active")); }

// ---------------------------------------------------------------- self-test (index.html?selftest): used to check browser = Python
async function selfTest(examples) {
  const results = [];
  for (const ex of examples) {
    const tensor = await prepare(await (await fetch(ex.file)).blob());
    const row = { file: ex.file };
    for (const m of [...MODES.real.models, ...MODES.synthetic.models]) row[m.file] = (await predict(m, tensor)).probs;
    results.push(row);
  }
  const pre = document.createElement("pre");
  pre.id = "selftest";
  pre.textContent = JSON.stringify(results);
  document.body.appendChild(pre);
  document.title = "SELFTEST DONE";
  fetch("selftest-result", { method: "POST", body: pre.textContent }).catch(() => {});   // only the test server listens
}

init();

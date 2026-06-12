"""
Face Animator Web UI — Flask  (Professional Edition)
Run: python app.py
Open: http://localhost:5000
"""
import os, tempfile, base64, io, time

# ── Load Env ─────────────────────────────────────────────────────
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

load_env()

from flask import Flask, render_template_string, request, send_file, jsonify
from animate import init_gemini, get_landmarks, animate_blink, animate_wink, animate_talk, animate_smile, animate_eyebrow_raise, animate_nod
import imageio, numpy as np, cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDJwfTmehVJWYpefIa_mAsR5tTUjMTcQ-A")

BLINK_CYCLE  = [1.0, 0.85, 0.6, 0.3, 0.08, 0.3, 0.6, 0.85, 1.0]
TALK_CYCLE   = [1.0, 0.7, 0.5, 0.8, 0.4, 0.6, 0.3, 0.7, 0.5, 1.0, 0.6, 0.3, 0.6, 0.8, 1.0]

# ─── Quickfire demo phrases ───────────────────────────────────────
DEMO_PHRASES = [
    "Good morning!",
    "How are you?",
    "Hello, nice to meet you!",
    "Have a great day!",
    "Thank you so much.",
    "I am very happy today.",
    "Welcome to the future.",
]

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Face Animator — Lifelike Expression Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js" crossorigin="anonymous"></script>
<style>
/* ── Reset & base ─────────────────────────────────────────────── */
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

:root {
  --bg:        #08090d;
  --surface:   #0e1018;
  --surface2:  #151923;
  --border:    #1e2535;
  --accent:    #e8c47a;
  --accent2:   #d4916a;
  --text:      #dce4f0;
  --muted:     #5a6580;
  --success:   #4ade80;
  --error:     #f87171;
  --radius:    14px;
  --shadow:    0 24px 64px rgba(0,0,0,.6);
}

body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  padding: 32px 16px 64px;
  background-image:
    radial-gradient(ellipse 60% 40% at 20% 10%, rgba(232,196,122,.06) 0%, transparent 60%),
    radial-gradient(ellipse 50% 30% at 80% 80%, rgba(212,145,106,.05) 0%, transparent 60%);
}

/* ── Layout ───────────────────────────────────────────────────── */
.wrap { max-width: 960px; width: 100%; }

/* ── Header ───────────────────────────────────────────────────── */
.header { text-align: center; margin-bottom: 40px; }
.header-eyebrow {
  font-size: .72rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--accent); font-weight: 500; margin-bottom: 10px;
}
.header h1 {
  font-family: 'DM Serif Display', serif;
  font-size: clamp(2rem, 5vw, 3.2rem);
  line-height: 1.1;
  background: linear-gradient(135deg, #f5e0a8 0%, #e8c47a 40%, #d4916a 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.header .sub {
  margin-top: 10px; font-size: .95rem; color: var(--muted); font-weight: 300;
}

/* ── Cards ────────────────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  margin-bottom: 20px;
}
.card-title {
  font-size: .72rem; letter-spacing: .15em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.card-title .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent);
}

/* ── Two-column grid ──────────────────────────────────────────── */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 680px) { .grid-2 { grid-template-columns: 1fr; } }

/* ── Upload zone ──────────────────────────────────────────────── */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 12px;
  padding: 36px 20px;
  text-align: center;
  cursor: pointer;
  transition: border-color .2s, background .2s;
  position: relative;
}
.upload-zone:hover, .upload-zone.dragover {
  border-color: var(--accent);
  background: rgba(232,196,122,.04);
}
.upload-zone .icon { font-size: 2.2rem; margin-bottom: 8px; }
.upload-zone p { font-size: .85rem; color: var(--muted); }
.upload-zone input[type=file] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer;
}
.upload-thumb {
  width: 100%; border-radius: 10px;
  max-height: 380px; object-fit: contain;
  display: none;
  border: 1px solid var(--border);
  background: var(--surface2);
}

/* ── Controls ─────────────────────────────────────────────────── */
label.lbl {
  display: block; font-size: .76rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .12em;
  color: var(--muted); margin-bottom: 8px; margin-top: 16px;
}
label.lbl:first-child { margin-top: 0; }

textarea, select, input[type=text] {
  width: 100%; padding: 12px 14px;
  background: var(--surface2);
  border: 1.5px solid var(--border);
  border-radius: 9px;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-size: .95rem;
  resize: none;
  transition: border-color .2s;
}
textarea:focus, select:focus, input[type=text]:focus {
  outline: none; border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(232,196,122,.12);
}
textarea { min-height: 96px; line-height: 1.5; }

select {
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%235a6580' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 14px center;
  cursor: pointer;
}

/* ── Demo pill chips ──────────────────────────────────────────── */
.chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
.chip {
  padding: 5px 13px; border-radius: 20px; font-size: .78rem;
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--muted);
  cursor: pointer;
  transition: all .18s;
  user-select: none;
}
.chip:hover {
  border-color: var(--accent); color: var(--accent);
  background: rgba(232,196,122,.08);
}

/* ── Animate button ───────────────────────────────────────────── */
.btn-primary {
  width: 100%; padding: 15px;
  background: linear-gradient(135deg, #e8c47a, #d4916a);
  color: #08090d;
  font-family: 'DM Sans', sans-serif;
  font-size: 1rem; font-weight: 700;
  border: none; border-radius: 10px;
  cursor: pointer; margin-top: 20px;
  letter-spacing: .04em;
  transition: opacity .2s, transform .15s, box-shadow .2s;
  box-shadow: 0 4px 20px rgba(232,196,122,.3);
}
.btn-primary:hover:not(:disabled) {
  opacity: .92; transform: translateY(-2px);
  box-shadow: 0 10px 32px rgba(232,196,122,.4);
}
.btn-primary:disabled { opacity: .4; cursor: not-allowed; transform: none; }

.btn-secondary {
  width: 100%; padding: 12px;
  background: var(--surface2); color: var(--muted);
  font-family: 'DM Sans', sans-serif; font-size: .9rem; font-weight: 500;
  border: 1.5px solid var(--border);
  border-radius: 10px; cursor: pointer; margin-top: 10px;
  transition: all .2s;
}
.btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

/* ── Status bar ───────────────────────────────────────────────── */
.status-bar {
  padding: 11px 16px; border-radius: 8px;
  font-size: .85rem; margin-top: 14px;
  display: none; align-items: center; gap: 8px;
}
.status-bar.info  { display:flex; background:rgba(90,101,128,.12); border:1px solid var(--border); }
.status-bar.ok    { display:flex; background:rgba(74,222,128,.08); border:1px solid rgba(74,222,128,.25); color:var(--success); }
.status-bar.err   { display:flex; background:rgba(248,113,113,.08); border:1px solid rgba(248,113,113,.25); color:var(--error); }

/* ── Loader ───────────────────────────────────────────────────── */
.loader-wrap {
  display: none; flex-direction: column; align-items: center;
  gap: 14px; padding: 28px; text-align: center;
}
.spinner {
  width: 44px; height: 44px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .75s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loader-wrap p { font-size: .88rem; color: var(--muted); }
.loader-wrap .sub-hint { font-size: .76rem; color: var(--border); margin-top: 4px; }

/* ── Result panel ─────────────────────────────────────────────── */
.result-panel {
  display: none;
  flex-direction: column; align-items: center; gap: 16px;
  margin-top: 20px;
}
.result-gif-wrap {
  position: relative; display: inline-block;
}
.result-gif-wrap img {
  max-width: 420px; width: 100%;
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
.result-label {
  position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
  background: rgba(8,9,13,.75); backdrop-filter: blur(8px);
  padding: 4px 12px; border-radius: 20px;
  font-size: .72rem; color: var(--muted); white-space: nowrap;
}
.result-meta { font-size: .8rem; color: var(--muted); }

/* ── Gemini badge ─────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 20px;
  font-size: .7rem; font-weight: 600; letter-spacing: .05em;
}
.badge-on  { background:rgba(74,222,128,.12); color:var(--success); border:1px solid rgba(74,222,128,.2); }
.badge-off { background:rgba(245,158,11,.08); color:#f59e0b; border:1px solid rgba(245,158,11,.2); }
.badge .dot { width:5px; height:5px; border-radius:50%; background:currentColor; }

/* ── Tips sidebar ─────────────────────────────────────────────── */
.tips { margin-top: 20px; }
.tips-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px,1fr)); gap: 12px; }
.tip-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}
.tip-card .tip-icon { font-size: 1.4rem; margin-bottom: 6px; }
.tip-card .tip-title { font-size: .78rem; font-weight: 600; color: var(--text); margin-bottom: 3px; }
.tip-card .tip-desc { font-size: .72rem; color: var(--muted); line-height: 1.45; }

/* ── Footer ───────────────────────────────────────────────────── */
.footer {
  text-align: center; margin-top: 48px;
  font-size: .72rem; color: var(--border);
}
</style>
</head>
<body>
<div class="wrap">

  <!-- Header -->
  <div class="header">
    <div class="header-eyebrow">✦ Powered by MediaPipe · Gemini</div>
    <h1>AI Face Animator</h1>
    <p class="sub">Upload any portrait · Type what you want the face to say or do · Get a fluid animated GIF</p>
  </div>

  <!-- Main grid -->
  <div class="grid-2">

    <!-- LEFT: Upload -->
    <div class="card">
      <div class="card-title"><span class="dot"></span> Face Photo</div>

      <div class="upload-zone" id="uploadZone">
        <div class="icon">🖼️</div>
        <p>Drag &amp; drop or click to select<br><small>JPG · PNG · WEBP · max 16 MB</small></p>
        <input type="file" id="imageInput" accept="image/*">
      </div>
      <img id="uploadThumb" class="upload-thumb" alt="Preview">

      <!-- API Key input -->
      <div style="margin-top:14px;">
        <label class="lbl" style="margin-top:0;margin-bottom:6px;">🔑 Gemini API Key</label>
        <input type="password" id="apiKeyInput" value="{{ default_key }}" placeholder="Enter API Key to enable AI..." style="padding:10px 12px;font-size:0.85rem;">
      </div>

      <!-- Gemini status row -->
      <div style="margin-top:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <span style="font-size:.75rem;color:var(--muted);">AI Status</span>
        <span class="badge badge-off" id="geminiBadge"><span class="dot"></span> <span id="geminiLabel">Checking…</span></span>
      </div>
    </div>

    <!-- RIGHT: Controls -->
    <div class="card">
      <div class="card-title"><span class="dot"></span> What Should the Face Do?</div>

      <!-- Speech textarea -->
      <label class="lbl">💬 Speech / Sentence (for lip-sync)</label>
      <textarea id="speechText" placeholder="Type a sentence the person should say…&#10;e.g.  Good morning! How are you?&#10;e.g.  Hello, nice to meet you today."></textarea>

      <!-- Quick phrase chips -->
      <div class="chips" id="phraseChips">
        {% for phrase in phrases %}
        <div class="chip" onclick="usePhrase('{{ phrase }}')">{{ phrase }}</div>
        {% endfor %}
      </div>

      <!-- Divider -->
      <div style="margin:18px 0;display:flex;align-items:center;gap:10px;">
        <div style="flex:1;height:1px;background:var(--border)"></div>
        <span style="font-size:.72rem;color:var(--muted)">OR pick an animation preset</span>
        <div style="flex:1;height:1px;background:var(--border)"></div>
      </div>

      <!-- Animation dropdown -->
      <label class="lbl">🎬 Animation Preset</label>
      <select id="animSelect">
        <option value="talk">🗣 Talk (lip-sync to sentence above)</option>
        <option value="blink">👁 Blink (both eyes)</option>
        <option value="wink_left">😉 Wink — left eye</option>
        <option value="wink_right">😉 Wink — right eye</option>
        <option value="smile">😊 Smile</option>
        <option value="eyebrow_raise">🤨 Eyebrow Raise (surprised)</option>
        <option value="nod">🙂 Nod</option>
        <option value="blink_smile">✨ Blink + Smile</option>
      </select>

      <!-- OR AI text box -->
      <label class="lbl" style="margin-top:14px;">🤖 AI Instruction (Gemini interprets freely)</label>
      <input type="text" id="aiText" placeholder='e.g. "make him wink 3 times then smile"'>
      <div style="font-size:.72rem;color:var(--muted);margin-top:5px;">Leave blank to use the preset above directly.</div>

    </div>
  </div>

  <!-- Animate button + loader + result -->
  <div class="card">
    <button class="btn-primary" id="animateBtn" onclick="runAnimate()">✨ Animate Face</button>

    <div class="status-bar info" id="statusBar">
      <span>📌</span> <span id="statusMsg">Upload a photo and click Animate.</span>
    </div>

    <div class="loader-wrap" id="loader">
      <div class="spinner"></div>
      <p>Generating animation…</p>
      <p class="sub-hint" id="loaderHint">Detecting face landmarks…</p>
    </div>

    <div class="result-panel" id="resultPanel">
      <div class="result-gif-wrap">
        <img id="resultGif" src="" alt="Animated result">
        <div class="result-label" id="resultLabel">Animated GIF</div>
      </div>
      <div class="result-meta" id="resultMeta"></div>
      <button class="btn-secondary" onclick="downloadGif()">⬇️ Download GIF</button>
    </div>
  </div>

  <!-- Tips -->
  <div class="tips">
    <div class="card-title" style="margin-bottom:14px;"><span class="dot"></span> Tips &amp; Examples</div>
    <div class="tips-grid">
      <div class="tip-card">
        <div class="tip-icon">💬</div>
        <div class="tip-title">Lip-sync anything</div>
        <div class="tip-desc">Type "Good morning, how are you?" in the sentence box — visemes are auto-generated for each phoneme.</div>
      </div>
      <div class="tip-card">
        <div class="tip-icon">👁</div>
        <div class="tip-title">Human blink timing</div>
        <div class="tip-desc">Blinks happen every 6–9 seconds naturally, matching real human behaviour during speech.</div>
      </div>
      <div class="tip-card">
        <div class="tip-icon">🤖</div>
        <div class="tip-title">AI free instructions</div>
        <div class="tip-desc">Use the AI box for creative combos: "wink three times then raise eyebrows and smile".</div>
      </div>
      <div class="tip-card">
        <div class="tip-icon">🖼</div>
        <div class="tip-title">Best photo tips</div>
        <div class="tip-desc">Front-facing, well-lit, single face, neutral background. Higher resolution = better results.</div>
      </div>
    </div>
  </div>

  <div class="footer">AI Face Animator · MediaPipe FaceLandmarker 478-pt · Gemini Vision · RBF Warp Engine</div>
</div>

<script>
let resultBlob = null;
let faceLandmarker = null;
let detectedLandmarks = null;
let landmarkError = null;

async function initFaceLandmarker() {
  try {
    const FilesetResolver = window.FilesetResolver || (window.vision && window.vision.FilesetResolver);
    const FaceLandmarker = window.FaceLandmarker || (window.vision && window.vision.FaceLandmarker);
    if (!FilesetResolver || !FaceLandmarker) {
      console.warn("MediaPipe library not loaded yet.");
      return;
    }
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
      runningMode: "IMAGE",
      numFaces: 1
    });
    console.log("Client-side MediaPipe FaceLandmarker initialized.");
  } catch (err) {
    console.error("Failed to load client-side FaceLandmarker:", err);
  }
}
let loaderHints = [
  "Detecting face landmarks…",
  "Running RBF warp engine…",
  "Generating lip-sync visemes…",
  "Compositing frames…",
  "Encoding GIF…"
];
let hintIdx = 0, hintTimer = null;

/* ── Image upload preview ──────────────────────────────── */
const imageInput = document.getElementById('imageInput');
const uploadZone = document.getElementById('uploadZone');
const thumb      = document.getElementById('uploadThumb');

thumb.onload = async () => {
  detectedLandmarks = null;
  landmarkError = null;
  
  if (!faceLandmarker) {
    setStatus("Initializing face detector in your browser...", "info");
    await initFaceLandmarker();
  }
  
  if (!faceLandmarker) {
    setStatus("⚠️ Could not load face detector. Server fallback will be used.", "info");
    landmarkError = "FaceLandmarker not initialized";
    return;
  }
  
  setStatus("Detecting face landmarks in browser...", "info");
  try {
    const result = faceLandmarker.detect(thumb);
    if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
      const w = thumb.naturalWidth;
      const h = thumb.naturalHeight;
      const pts = result.faceLandmarks[0];
      detectedLandmarks = pts.map(p => [Math.round(p.x * w), Math.round(p.y * h)]);
      setStatus(`✅ Face detected! Ready to animate.`, "ok");
    } else {
      setStatus("⚠️ No face detected. Try a clear front-facing photo.", "err");
      landmarkError = "No face detected in browser";
    }
  } catch (err) {
    console.error("Landmark detection error:", err);
    setStatus("⚠️ Face detection failed in browser.", "err");
    landmarkError = err.message;
  }
};

imageInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    thumb.src = ev.target.result;
    thumb.style.display = 'block';
    uploadZone.style.display = 'none';
  };
  reader.readAsDataURL(file);
  resetResult();
});

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) {
    imageInput.files = e.dataTransfer.files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

/* ── Demo phrase chips ─────────────────────────────────── */
function usePhrase(text) {
  document.getElementById('speechText').value = text;
  document.getElementById('animSelect').value = 'talk';
  document.getElementById('aiText').value = '';
  document.getElementById('speechText').focus();
}

/* ── Status helpers ────────────────────────────────────── */
function setStatus(msg, type) {
  const bar = document.getElementById('statusBar');
  document.getElementById('statusMsg').textContent = msg;
  bar.className = 'status-bar ' + type;
}

function resetResult() {
  document.getElementById('resultPanel').style.display = 'none';
  setStatus('Upload a photo and click Animate.', 'info');
}

/* ── Loader hints cycling ──────────────────────────────── */
function startHints() {
  hintIdx = 0;
  document.getElementById('loaderHint').textContent = loaderHints[0];
  hintTimer = setInterval(() => {
    hintIdx = (hintIdx + 1) % loaderHints.length;
    document.getElementById('loaderHint').textContent = loaderHints[hintIdx];
  }, 1800);
}
function stopHints() { clearInterval(hintTimer); }

/* ── Main animate function ─────────────────────────────── */
async function runAnimate() {
  const fileInput = document.getElementById('imageInput');
  if (!fileInput.files[0]) { setStatus('⚠️ Please upload a face photo first.', 'err'); return; }

  const speechText = document.getElementById('speechText').value.trim();
  const aiText     = document.getElementById('aiText').value.trim();
  const animSel    = document.getElementById('animSelect').value;

  // If speech text given, force talk mode
  const finalAnim = speechText && !aiText ? 'talk' : animSel;

  const apiKey     = document.getElementById('apiKeyInput').value.trim();

  const formData = new FormData();
  formData.append('image',     fileInput.files[0]);
  formData.append('animation', finalAnim);
  formData.append('text',      aiText || (speechText ? '' : ''));
  formData.append('speech',    speechText);
  formData.append('api_key',   apiKey);
  
  if (detectedLandmarks) {
    formData.append('landmarks', JSON.stringify(detectedLandmarks));
  } else {
    console.warn("No client-side landmarks detected. Falling back to server-side detection.");
  }

  // UI: loading state
  document.getElementById('animateBtn').disabled = true;
  document.getElementById('loader').style.display = 'flex';
  document.getElementById('resultPanel').style.display = 'none';
  document.getElementById('statusBar').style.display = 'none';
  startHints();

  const t0 = performance.now();

  try {
    const resp = await fetch('/animate', { method: 'POST', body: formData });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'Server error' }));
      setStatus('❌ ' + (err.error || 'Animation failed.'), 'err');
      return;
    }

    const blob = await resp.blob();
    resultBlob = blob;
    const url  = URL.createObjectURL(blob);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    document.getElementById('resultGif').src = url;
    document.getElementById('resultLabel').textContent = 'Animated GIF';
    document.getElementById('resultMeta').textContent =
      `✅ Generated in ${elapsed}s · ${(blob.size / 1024).toFixed(1)} KB`;
    document.getElementById('resultPanel').style.display = 'flex';

  } catch (e) {
    setStatus('❌ Network error: ' + e.message, 'err');
  } finally {
    stopHints();
    document.getElementById('loader').style.display = 'none';
    document.getElementById('animateBtn').disabled = false;
  }
}

/* ── Download ──────────────────────────────────────────── */
function downloadGif() {
  if (!resultBlob) return;
  const url = URL.createObjectURL(resultBlob);
  const a   = document.createElement('a');
  a.href = url; a.download = 'face_animation.gif'; a.click();
  URL.revokeObjectURL(url);
}

/* ── Gemini badge check ────────────────────────────────── */
async function checkGemini() {
  const key = document.getElementById('apiKeyInput').value.trim();
  const badge = document.getElementById('geminiBadge');
  const lbl   = document.getElementById('geminiLabel');
  
  badge.className = 'badge badge-off';
  lbl.textContent = 'Checking…';
  
  try {
    const resp = await fetch('/gemini_status?key=' + encodeURIComponent(key));
    const data = await resp.json();
    if (data.available) {
      badge.className = 'badge badge-on';
      lbl.textContent = 'AI ON';
    } else {
      badge.className = 'badge badge-off';
      lbl.textContent = 'AI OFF';
    }
  } catch(e) {
    lbl.textContent = 'Unavailable';
  }
}
document.getElementById('apiKeyInput').addEventListener('change', checkGemini);
checkGemini();
</script>
</body>
</html>"""

@app.route("/")
def index():
    default_key = os.environ.get("GEMINI_API_KEY", "")
    return render_template_string(HTML, phrases=DEMO_PHRASES, default_key=default_key)

@app.route("/gemini_status")
def gemini_status():
    key_param = request.args.get("key", "").strip()
    key_to_test = key_param or GEMINI_KEY
    if not key_to_test:
        return jsonify({"available": False, "reason": "No API key"})
    try:
        init_gemini(key_to_test)
        from google import genai
        client = genai.Client(api_key=key_to_test)
        model_name = os.environ.get("GEMINI_AI_MODEL", "gemini-2.0-flash")
        client.models.generate_content(model=model_name, contents="ok")
        return jsonify({"available": True})
    except Exception as e:
        return jsonify({"available": False, "reason": str(e)})

@app.route("/animate", methods=["POST"])
def animate():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        animation   = request.form.get("animation", "blink")
        user_text   = request.form.get("text", "").strip()     # AI free-text instruction
        speech_text = request.form.get("speech", "").strip()   # Direct lip-sync sentence
        user_key    = request.form.get("api_key", "").strip()  # Key passed from UI

        # Save uploaded image
        suffix = os.path.splitext(file.filename or "face.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Detect landmarks (accept client-side detection to bypass MediaPipe on server)
        landmarks_json = request.form.get("landmarks")
        img = None
        landmarks = None
        w, h = 0, 0

        if landmarks_json:
            try:
                import json
                raw_lms = json.loads(landmarks_json)
                landmarks = [(int(pt[0]), int(pt[1])) for pt in raw_lms]
                img = cv2.imread(tmp_path)
                if img is not None:
                    h, w = img.shape[:2]
                    # Scale landmarks if image needs downscaling
                    max_dim = 1024
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        landmarks = [(int(x * scale), int(y * scale)) for (x, y) in landmarks]
                        w, h = new_w, new_h
            except Exception as e:
                print(f"[Client Landmarks] Error parsing landmarks: {e}")
                landmarks = None

        if landmarks is None:
            # Fallback to server-side MediaPipe landmarker
            img, (w, h), landmarks = get_landmarks(tmp_path)

        if landmarks is None:
            os.unlink(tmp_path)
            return jsonify({"error": "No face detected. Try a clear front-facing photo."}), 400

        # ── Determine what to animate ──────────────────────────────
        loops = 1
        fps   = 15

        key_to_use = user_key or GEMINI_KEY

        if user_text:
            # Gemini interprets the free-text instruction
            try:
                if key_to_use:
                    init_gemini(key_to_use)
                from animate import interpret_text
                config = interpret_text(user_text)
                animation   = config.get("animation", animation)
                loops       = config.get("loops", 1)
                fps         = config.get("fps", 15)
                # If Gemini extracted speech, use it (override)
                if config.get("speech_text"):
                    speech_text = config["speech_text"]
                print(f"[Gemini] anim={animation} loops={loops} fps={fps} speech='{speech_text}'")
            except Exception as e:
                print(f"[Gemini] interpret failed: {e}")

        # Direct speech text always forces talk mode
        if speech_text and animation not in ("talk",):
            animation = "talk"

        # ── Generate frames ────────────────────────────────────────
        all_frames = []
        for _ in range(max(1, loops)):
            if animation == "blink":
                all_frames.extend(animate_blink(img, landmarks, w, h, BLINK_CYCLE, fps))
            elif animation == "wink_left":
                all_frames.extend(animate_wink(img, landmarks, w, h, "left", BLINK_CYCLE))
            elif animation == "wink_right":
                all_frames.extend(animate_wink(img, landmarks, w, h, "right", BLINK_CYCLE))
            elif animation == "talk":
                all_frames.extend(animate_talk(img, landmarks, w, h,
                                               cycle=TALK_CYCLE if not speech_text else None,
                                               text=speech_text or None))
            elif animation == "smile":
                all_frames.extend(animate_smile(img, landmarks, w, h))
            elif animation == "eyebrow_raise":
                all_frames.extend(animate_eyebrow_raise(img, landmarks, w, h))
            elif animation == "nod":
                all_frames.extend(animate_nod(img, landmarks, w, h))
            elif animation == "blink_smile":
                all_frames.extend(animate_blink(img, landmarks, w, h, BLINK_CYCLE, fps))
                all_frames.extend(animate_smile(img, landmarks, w, h, intensity=1.0))
            else:
                all_frames.extend(animate_blink(img, landmarks, w, h, BLINK_CYCLE, fps))

        if not all_frames:
            os.unlink(tmp_path)
            return jsonify({"error": "Animation produced no frames."}), 500

        # ── Encode GIF ─────────────────────────────────────────────
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
        imageio.mimsave(output_path, all_frames, fps=fps, loop=0)

        os.unlink(tmp_path)

        response = send_file(output_path, mimetype="image/gif", as_attachment=False)

        @response.call_on_close
        def _cleanup():
            if os.path.exists(output_path):
                os.unlink(output_path)

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not GEMINI_KEY:
        print("[WARN] GEMINI_API_KEY not set — AI interpretation disabled.")
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 42)
    print("  AI Face Animator  v2.0")
    print("  http://localhost:5000")
    print(f"  Gemini: {'ON' if GEMINI_KEY else 'OFF'}")
    print("  Press Ctrl+C to stop")
    print("=" * 42)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
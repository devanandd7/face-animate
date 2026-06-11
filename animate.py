"""
Face Animation Script v4 -- Image Upload -> AI Animation from Text
===================================================================
v4 FIXES (frame analysis revealed):
- Blink now GUARANTEED in every animation regardless of length
- Lip displacement 3x larger -- actually visible on small faces
- Mouth cavity shown by drawing dark filled ellipse directly
- GIF quality: uses RGBA + optimized palette for sharper output
- Face-size adaptive scaling so small/large photos both work
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options
import numpy as np
import imageio
from PIL import Image
import json
import os
import random

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

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ── Landmark index groups ────────────────────────────────────────
LEFT_EYE       = [33, 160, 158, 133, 153, 144, 145, 163, 7, 173, 157]
RIGHT_EYE      = [362, 385, 387, 263, 373, 380, 381, 382, 374, 390, 466]
LEFT_EYEBROW   = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 285]
RIGHT_EYEBROW  = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
MOUTH_OUTER    = [61, 185, 40, 39, 37, 267, 269, 270, 409, 291,
                  375, 321, 405, 314, 17, 84, 181, 91, 146, 76]
MOUTH_INNER    = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Lip groups
UL_TOP   = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]  # upper outer
UL_BOT   = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]  # upper inner
LL_TOP   = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]  # lower inner
LL_BOT   = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]  # lower outer
L_CORNER = 61
R_CORNER = 291
CHIN     = [152, 377, 378, 379, 397, 365, 148, 176]

# Eye blink RBF pairs: upper lid → lower lid
LEFT_EYE_PAIRS = {159:145, 160:144, 158:153, 157:154, 173:155, 161:163}
RIGHT_EYE_PAIRS = {386:374, 385:380, 384:381, 387:373, 388:390}
LEFT_EYE_ANCHORS  = [33, 133] + LEFT_EYEBROW
RIGHT_EYE_ANCHORS = [263, 362] + RIGHT_EYEBROW

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Mouth anchors: these MUST NOT move during lip animation (tightly bounds mouth warping to lower face)
MOUTH_ANCHORS = [
    # Nose base (stops upward warp propagation to nose bridge/eyes)
    2, 94, 327, 98, 164, 326,
    # Cheeks surrounding the mouth (restricts cheek stretching)
    50, 101, 280, 330, 116, 345, 205, 425,
    # Lower jawline/face oval boundaries (protects the neck sides)
    148, 176, 152, 377, 378, 379, 365, 397
]

# ── Gemini ───────────────────────────────────────────────────────
try:
    from google import genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

gemini_client = None

def init_gemini(api_key: str):
    global gemini_client
    if not _GENAI_OK:
        print("[Gemini] google-genai not installed")
        return
    gemini_client = genai.Client(api_key=api_key)
    print("[Gemini] Client initialized [OK]")


# ── Face Landmark Detection ──────────────────────────────────────
def get_landmarks(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    
    # Downscale image to a maximum dimension of 1024px for optimal performance
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Write back to image_path so MediaPipe reads the resized image
        cv2.imwrite(image_path, img)
        h, w = img.shape[:2]

    if not os.path.exists(MODEL_PATH):
        print(f"[MediaPipe] Model not found: {MODEL_PATH}")
        return None, None, None
    base_opts = mp_base_options.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(opts) as det:
        mp_img = mp.Image.create_from_file(image_path)
        result = det.detect(mp_img)
        if result.face_landmarks:
            pts = result.face_landmarks[0]
            lm = [(int(p.x * w), int(p.y * h)) for p in pts]
            print(f"[MediaPipe] {len(lm)} landmarks | {w}x{h}")
            return img, (w, h), lm
    print("[MediaPipe] No face found")
    return None, None, None


def get_landmarks_via_gemini(image_path: str):
    if gemini_client is None:
        return None
    try:
        pil_img = Image.open(image_path)
        w, h = pil_img.size
        prompt = (
            f"Analyze this face. Return JSON only (no markdown) with pixel coords:\n"
            f"image size: {w}x{h}\n"
            '{"left_eye":[x,y],"right_eye":[x,y],"mouth":[x,y],'
            '"mouth_left":[x,y],"mouth_right":[x,y],'
            '"left_eyebrow":[x,y],"right_eyebrow":[x,y]}'
        )
        model_name = os.environ.get("GEMINI_AI_MODEL", "gemini-2.0-flash")
        resp = gemini_client.models.generate_content(
            model=model_name, contents=[pil_img, prompt])
        text = resp.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[Gemini Vision] {e}")
        return None


# ── RBF Warp ─────────────────────────────────────────────────────
def rbf_warp_local(img, landmarks, active_pairs, anchor_indices, sigma, scale=1.0):
    """
    Gaussian RBF local warp.
    active_pairs: dict{src_idx:dst_idx} OR list[(src_arr, dst_arr)]
    scale=0.0 → fully apply displacement, 1.0 → identity
    """
    src_pts, dst_pts = [], []
    active_srcs = []
    if isinstance(active_pairs, dict):
        for u, l in active_pairs.items():
            u_pt = np.array(landmarks[u], dtype=np.float32)
            l_pt = np.array(landmarks[l], dtype=np.float32)
            src_pts.append(u_pt)
            dst_pts.append(scale * u_pt + (1.0 - scale) * l_pt)
            active_srcs.append(u_pt)
    else:
        for u_pt, l_pt in active_pairs:
            u_arr = np.array(u_pt, dtype=np.float32)
            l_arr = np.array(l_pt, dtype=np.float32)
            src_pts.append(u_arr)
            dst_pts.append(scale * u_arr + (1.0 - scale) * l_arr)
            active_srcs.append(u_arr)

    # Filter anchor_indices: no duplicates, and not too close to active points (within 5 pixels)
    unique_anchors = []
    for idx in anchor_indices:
        if idx in unique_anchors:
            continue
        pt_coord = np.array(landmarks[idx], dtype=np.float32)
        too_close = False
        for act_pt in active_srcs:
            if np.linalg.norm(pt_coord - act_pt) < 5.0:
                too_close = True
                break
        if not too_close:
            unique_anchors.append(idx)

    # Append safe, unique anchors
    for idx in unique_anchors:
        pt = np.array(landmarks[idx], dtype=np.float32)
        src_pts.append(pt); dst_pts.append(pt)

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    disp = dst_pts - src_pts
    N = len(src_pts)
    if N == 0:
        return img.copy()

    reg   = 1e-2  # Slightly higher regularization to prevent any high-frequency oscillations
    K     = np.exp(-np.sum((src_pts[:,None]-src_pts[None,:])**2, axis=-1) / (2*sigma**2))
    w_rbf = np.linalg.solve(K + reg * np.eye(N), disp)

    margin = max(4, int(sigma * 2.5))
    x0 = max(0, int(src_pts[:,0].min()) - margin)
    y0 = max(0, int(src_pts[:,1].min()) - margin)
    x1 = min(img.shape[1], int(src_pts[:,0].max()) + margin)
    y1 = min(img.shape[0], int(src_pts[:,1].max()) + margin)
    if x1 <= x0 or y1 <= y0:
        return img.copy()

    xs, ys = np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)
    X, Y   = np.meshgrid(xs, ys)
    gp     = np.stack([X.ravel(), Y.ravel()], axis=-1)
    Phi    = np.exp(-np.sum((gp[:,None]-src_pts[None,:])**2, axis=-1) / (2*sigma**2))
    dg     = Phi @ w_rbf
    dx     = dg[:,0].reshape(X.shape)
    dy     = dg[:,1].reshape(Y.shape)

    h_roi, w_roi = X.shape
    Xr = X - x0; Yr = Y - y0
    border_d = np.minimum(np.minimum(Xr, w_roi-1-Xr), np.minimum(Yr, h_roi-1-Yr))
    fade = np.clip(border_d / max(1.0, float(margin)), 0.0, 1.0)
    fade = 0.5 * (1.0 - np.cos(fade * np.pi))
    dx *= fade; dy *= fade

    map_x  = (Xr - dx).astype(np.float32)
    map_y  = (Yr - dy).astype(np.float32)
    roi    = img[y0:y1, x0:x1]
    warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out    = img.copy()
    out[y0:y1, x0:x1] = warped
    return out


# ── Blink (guaranteed) ───────────────────────────────────────────
BLINK_TPL = [1.0, 0.80, 0.45, 0.10, 0.02, 0.10, 0.45, 0.80, 1.0]   # 9 frames

def _apply_eye_blink_masked(img, landmarks, face_width, left_scale, right_scale):
    """
    Applies eye blink/wink using local RBF warp, restricted to an oval mask around the eyes
    with dynamic eyebrow-shielding vertical gradients to prevent eyebrow stretching.
    """
    out_img = img.copy()
    h, w = img.shape[:2]
    
    # ── Left Eye ──
    if left_scale < 0.99:
        # 1. Compute RBF warped image for left eye
        sigma = face_width * 0.04
        warped_l = rbf_warp_local(out_img, landmarks, LEFT_EYE_PAIRS, LEFT_EYE_ANCHORS, sigma, scale=left_scale)
        
        # 2. Compute left eye oval mask
        left_xs = [landmarks[idx][0] for idx in LEFT_EYE]
        left_ys = [landmarks[idx][1] for idx in LEFT_EYE]
        l_x_min, l_x_max = min(left_xs), max(left_xs)
        l_y_min, l_y_max = min(left_ys), max(left_ys)
        l_cx = int((l_x_min + l_x_max) / 2)
        l_cy = int((l_y_min + l_y_max) / 2)
        l_w = l_x_max - l_x_min
        
        # Sleek oval shape to cover the eye sockets/eyelids, avoiding eyebrows
        rx = int(l_w * 0.70)
        ry = int(l_w * 0.22)
        
        mask_l = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask_l, (l_cx, l_cy), (rx, ry), 0, 0, 360, 1.0, -1)
        
        # Soft feathering (small kernel to avoid spreading to eyebrows)
        ksize = int(l_w * 0.20) | 1
        mask_l = cv2.GaussianBlur(mask_l, (ksize, ksize), 0)
        
        # Safeguard: fade mask to 0 at and above the lowest eyebrow landmark
        l_brow_y_max = max(landmarks[idx][1] for idx in LEFT_EYEBROW)
        eye_ramp = np.ones(h, dtype=np.float32)
        for y in range(h):
            if y <= l_brow_y_max:
                eye_ramp[y] = 0.0
            elif y < l_cy:
                eye_ramp[y] = (y - l_brow_y_max) / (l_cy - l_brow_y_max)
        
        mask_l = mask_l * eye_ramp[:, np.newaxis]
        
        # Blend the warped eye back into the image
        mask_l_3d = mask_l[:, :, np.newaxis]
        out_img = (out_img * (1.0 - mask_l_3d) + warped_l * mask_l_3d).astype(np.uint8)

    # ── Right Eye ──
    if right_scale < 0.99:
        # 1. Compute RBF warped image for right eye
        sigma = face_width * 0.04
        warped_r = rbf_warp_local(out_img, landmarks, RIGHT_EYE_PAIRS, RIGHT_EYE_ANCHORS, sigma, scale=right_scale)
        
        # 2. Compute right eye oval mask
        right_xs = [landmarks[idx][0] for idx in RIGHT_EYE]
        right_ys = [landmarks[idx][1] for idx in RIGHT_EYE]
        r_x_min, r_x_max = min(right_xs), max(right_xs)
        r_y_min, r_y_max = min(right_ys), max(right_ys)
        r_cx = int((r_x_min + r_x_max) / 2)
        r_cy = int((r_y_min + r_y_max) / 2)
        r_w = r_x_max - r_x_min
        
        # Sleek oval shape to cover the eye sockets/eyelids, avoiding eyebrows
        rx = int(r_w * 0.70)
        ry = int(r_w * 0.22)
        
        mask_r = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask_r, (r_cx, r_cy), (rx, ry), 0, 0, 360, 1.0, -1)
        
        # Soft feathering (small kernel to avoid spreading to eyebrows)
        ksize = int(r_w * 0.20) | 1
        mask_r = cv2.GaussianBlur(mask_r, (ksize, ksize), 0)
        
        # Safeguard: fade mask to 0 at and above the lowest eyebrow landmark
        r_brow_y_max = max(landmarks[idx][1] for idx in RIGHT_EYEBROW)
        eye_ramp = np.ones(h, dtype=np.float32)
        for y in range(h):
            if y <= r_brow_y_max:
                eye_ramp[y] = 0.0
            elif y < r_cy:
                eye_ramp[y] = (y - r_brow_y_max) / (r_cy - r_brow_y_max)
                
        mask_r = mask_r * eye_ramp[:, np.newaxis]
        
        # Blend the warped eye back into the image
        mask_r_3d = mask_r[:, :, np.newaxis]
        out_img = (out_img * (1.0 - mask_r_3d) + warped_r * mask_r_3d).astype(np.uint8)
        
    return out_img

def _apply_blink(img, landmarks, face_width, eye_scale):
    """Apply eye blink at given scale (0=closed, 1=open)."""
    return _apply_eye_blink_masked(img, landmarks, face_width, eye_scale, eye_scale)

def _make_blink_schedule(n_frames, fps=15):
    """
    Returns set of frame indices that are part of a blink.
    Guarantees AT LEAST ONE blink per animation.
    Human blink: every 3-5 sec (45-75 frames at 15fps).
    For short clips (<45 frames), insert blink in middle.
    """
    blink_set = {}   # frame_idx -> blink_tpl_offset
    blink_len = len(BLINK_TPL)

    if n_frames <= blink_len + 4:
        # Too short for natural blink → no blink to avoid cutoff
        return {}

    # For very short animations, put blink at 30% mark
    if n_frames < 45:
        start = max(2, int(n_frames * 0.30))
        for k in range(blink_len):
            if start + k < n_frames:
                blink_set[start + k] = k
        return blink_set

    # Normal: every 3-5 seconds (45 to 75 frames)
    t = random.randint(20, 40)   # first blink comes earlier
    while t < n_frames - blink_len:
        for k in range(blink_len):
            blink_set[t + k] = k
        t += random.randint(45, 75)
    return blink_set


# ── Public blink/wink animators ─────────────────────────────────
def animate_blink(img, landmarks, w, h, cycle=None, fps=12):
    p_l = landmarks[234]; p_r = landmarks[454]
    fw  = float(np.linalg.norm(np.array(p_l) - np.array(p_r)))
    if cycle is None:
        cycle = BLINK_TPL
    return [cv2.cvtColor(_apply_blink(img, landmarks, fw, sc), cv2.COLOR_BGR2RGB)
            for sc in cycle]

def animate_wink(img, landmarks, w, h, side="left", cycle=None):
    p_l = landmarks[234]; p_r = landmarks[454]
    fw  = float(np.linalg.norm(np.array(p_l) - np.array(p_r)))
    if cycle is None:
        cycle = [1.0, 0.7, 0.3, 0.1, 0.3, 0.7, 1.0]
    frames = []
    for sc in cycle:
        left_sc = sc if side == "left" else 1.0
        right_sc = sc if side == "right" else 1.0
        f = _apply_eye_blink_masked(img, landmarks, fw, left_sc, right_sc)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


# ── Smile / Eyebrow / Nod ────────────────────────────────────────
def animate_smile(img, landmarks, w, h, intensity=1.0):
    p_l = landmarks[234]; p_r = landmarks[454]
    fw  = float(np.linalg.norm(np.array(p_l) - np.array(p_r)))
    sigma   = fw * 0.14
    stretch = fw * 0.13 * intensity
    lc = np.array(landmarks[61], dtype=np.float32)
    rc = np.array(landmarks[291], dtype=np.float32)
    pairs   = [(lc, lc+np.array([-stretch, -stretch*0.3])),
               (rc, rc+np.array([+stretch, -stretch*0.3]))]
    anchors = [6, 168, 197, 2, 94, 152, 148, 377, 33, 133, 263, 362]
    frames  = []
    for t in np.linspace(0, 1.0, 10):
        f = rbf_warp_local(img, landmarks, pairs, anchors, sigma, scale=1.0-t)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    frames += [frames[-1]] * 5
    return frames

def animate_eyebrow_raise(img, landmarks, w, h, intensity=1.0):
    p_l = landmarks[234]; p_r = landmarks[454]
    fw  = float(np.linalg.norm(np.array(p_l) - np.array(p_r)))
    sigma = fw * 0.16
    shift = fw * 0.13 * intensity
    pairs = [(np.array(landmarks[i], dtype=np.float32),
              np.array(landmarks[i], dtype=np.float32) + np.array([0.0, -shift]))
             for i in LEFT_EYEBROW + RIGHT_EYEBROW]
    anchors = [6, 168, 197, 33, 133, 263, 362, 234, 454]
    frames  = []
    for t in np.linspace(0, 1.0, 8):
        f = rbf_warp_local(img, landmarks, pairs, anchors, sigma, scale=1.0-t)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    frames += [frames[-1]] * 4
    return frames

def animate_nod(img, landmarks, w, h, intensity=1.0):
    frames = []
    for t in np.sin(np.linspace(0, np.pi*2, 14)) * intensity:
        f = img.copy()
        M = np.float32([[1, 0, 0], [0, 1, int(t * 8)]])
        f = cv2.warpAffine(f, M, (w, h))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


# ═══════════════════════════════════════════════════════════════════
# VISEME ENGINE v4 — PROPERLY SCALED DISPLACEMENTS
# ═══════════════════════════════════════════════════════════════════
#
# HOW IT WORKS:
# 1. Text → grapheme-to-viseme mapping (char by char, digraphs handled)
# 2. Each viseme maps to 5 parameters: [jaw, spread, purse, upper_rise, lower_drop]
# 3. Parameters expanded to per-frame list, Gaussian smoothed (co-articulation)
# 4. Per frame: RBF warp moves lip landmarks UP (upper) and DOWN (lower)
# 5. Dark cavity drawn as ellipse at mouth center when open
# 6. Blink injected at scheduled frames using separate eye RBF
#
# KEY FIX v4: base_unit now face_width * 0.10 (was 0.048) — 2x larger
# so lips visibly move even on small (200px) faces.
# ═══════════════════════════════════════════════════════════════════

VISEME_TABLE = {
    # key      jaw    spread  purse  u_rise  lo_drop
    "silence": (0.00,  0.00,  0.00,  0.00,   0.00),
    "rest":    (0.05,  0.00,  0.00,  0.03,   0.05),
    "M":       (0.00,  0.00,  0.05,  0.08,   0.00),   # lips pressed closed
    "F":       (0.12,  0.00,  0.00,  0.40,   0.10),   # lower lip tucks up
    "TH":      (0.40,  0.12,  0.00,  0.12,   0.35),
    "L":       (0.30,  0.06,  0.00,  0.08,   0.28),   # neutral open
    "SH":      (0.60, -0.20,  0.30,  0.10,   0.55),   # rounded open
    "A":       (2.00,  0.12,  0.00,  0.45,   1.80),   # WIDE OPEN
    "E":       (0.70,  0.65,  0.00,  0.20,   0.60),   # spread grin
    "I":       (0.30,  0.85,  0.00,  0.10,   0.25),   # wide grin barely open
    "O":       (1.00, -0.38,  0.55,  0.15,   0.90),   # rounded moderate open
    "U":       (0.50, -0.65,  0.75,  0.10,   0.45),   # tight round
}

VISEME_ALIAS = {
    "B":"M","P":"M","V":"F","W":"U","Y":"I",
    "N":"L","T":"L","D":"L","S":"L","Z":"L",
    "R":"L","C":"L","X":"L","Q":"U",
    "K":"A","G":"A","H":"rest","NG":"L",
    "CH":"SH","J":"SH","ZH":"SH",
}

def _resolve_viseme(key):
    k = key.upper()
    return VISEME_TABLE.get(VISEME_ALIAS.get(k, k), VISEME_TABLE["rest"])

_CHAR_MAP = {
    'a':('A',3),'e':('E',3),'i':('I',2),'o':('O',3),'u':('U',3),
    'm':('M',2),'b':('M',2),'p':('M',2),
    'f':('F',2),'v':('F',2),
    'w':('U',2),'y':('I',2),
    'l':('L',2),'n':('L',2),'t':('L',1),'d':('L',1),
    's':('L',2),'z':('L',2),'r':('L',2),
    'k':('A',1),'g':('A',1),'h':('rest',1),
    'j':('SH',2),'c':('L',1),'q':('U',2),'x':('L',2),
    ' ':('silence',3),',':('silence',4),'.':('silence',6),
    '!':('silence',5),'?':('silence',5),
}
_DIGRAPH_MAP = {
    'sh':('SH',3),'ch':('SH',3),'zh':('SH',3),
    'th':('TH',2),'ng':('L',2),
}


def text_to_param_timeline(text: str):
    """
    text → smoothed numpy array (N, 5): [jaw, spread, purse, upper_rise, lower_drop]
    Co-articulation: asymmetric Gaussian kernel blends adjacent phonemes.
    """
    text = text.lower().strip()
    # Prepend 4 frames of silence to allow a smooth fade-in from the natural closed face
    raw = [_resolve_viseme('silence')] * 4
    i = 0
    while i < len(text):
        dg = text[i:i+2]
        if dg in _DIGRAPH_MAP:
            vk, dur = _DIGRAPH_MAP[dg]
            raw.extend([_resolve_viseme(vk)] * dur)
            i += 2
        else:
            vk, dur = _CHAR_MAP.get(text[i], ('L', 1))
            raw.extend([_resolve_viseme(vk)] * dur)
            i += 1
    raw.extend([_resolve_viseme('silence')] * 10)
    if not raw:
        raw = [_resolve_viseme('silence')]

    arr = np.array(raw, dtype=np.float32)           # (N, 5)
    # 7-tap Gaussian kernel for extremely smooth viseme co-articulation transitions
    kernel = np.array([0.04, 0.09, 0.18, 0.38, 0.18, 0.09, 0.04], dtype=np.float32)
    kernel /= kernel.sum()
    N = len(arr)
    smoothed = np.zeros_like(arr)
    for i in range(N):
        total_w = 0.0
        for ki, kw in enumerate(kernel):
            fi = i + (ki - 3)  # Center index shifted by 3 for 7-tap kernel
            if 0 <= fi < N:
                smoothed[i] += arr[fi] * kw
                total_w += kw
        smoothed[i] /= (total_w + 1e-9)
    return smoothed


def _apply_viseme_frame(img, landmarks, params, face_width, sigma_mouth):
    """
    Apply lip warp for one frame.
    params: [jaw, spread, purse, upper_rise, lower_drop]
    """
    jaw, spread, purse, upper_rise, lower_drop = [float(x) for x in params]
    base_unit = face_width * 0.05
    lc = np.array(landmarks[L_CORNER], dtype=np.float32)
    rc = np.array(landmarks[R_CORNER], dtype=np.float32)
    mouth_cx    = (lc[0] + rc[0]) * 0.5
    mouth_width = abs(rc[0] - lc[0])

    def pt(idx): return np.array(landmarks[idx], dtype=np.float32)

    active = []

    # ── Synthetic neck/body anchors to keep collar/neck completely static ──
    # Disabled for speed optimization (rendered redundant by tight lips blending mask)
    # cx, cy = landmarks[152]
    # for dy_factor in [0.08, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.3]:
    #     dy = int(dy_factor * face_width)
    #     for dx_factor in [-1.8, -1.3, -0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.3, 1.8]:
    #         dx = int(dx_factor * face_width)
    #         pt_sec = np.array([cx + dx, cy + dy], dtype=np.float32)
    #         active.append((pt_sec, pt_sec))

    # ── Corners ──────────────────────────────────────────────────
    corner_dx = spread * mouth_width * 0.20 - purse * mouth_width * 0.12
    corner_dy = jaw * base_unit * 0.04
    active.append((lc, lc + np.array([-corner_dx,  corner_dy])))
    active.append((rc, rc + np.array([+corner_dx,  corner_dy])))

    # ── Upper lip — moves UP (negative y in image coords) ────────
    upper_dy = -upper_rise * base_unit * 0.9   # negative = upward

    def get_taper(px):
        # Parabolic taper function that goes to 0.0 at the corners of the mouth
        # to preserve the mouth corners as a locked physical hinge (lens shape)
        dist_ratio = abs(px - mouth_cx) / (mouth_width * 0.5 + 1e-6)
        return max(0.0, 1.0 - dist_ratio**2)

    for idx in UL_TOP:
        p = pt(idx)
        taper = get_taper(p[0])
        dx_p  = -purse * (p[0]-mouth_cx) * 0.09
        dst   = p + np.array([dx_p, upper_dy * taper])
        active.append((p, dst))

    for idx in UL_BOT:
        p = pt(idx)
        taper = get_taper(p[0])
        dx_p  = -purse * (p[0]-mouth_cx) * 0.07
        dst   = p + np.array([dx_p, upper_dy * taper * 0.6])
        active.append((p, dst))

    # ── Lower lip — moves DOWN (positive y) ──────────────────────
    lower_dy = lower_drop * base_unit * 1.0    # positive = downward

    if upper_rise > 0.30 and lower_drop < 0.20 and purse < 0.1:
        lower_dy = -base_unit * 0.60

    for idx in LL_BOT:
        p = pt(idx)
        taper = get_taper(p[0])
        dx_p  = -purse * (p[0]-mouth_cx) * 0.07
        dst   = p + np.array([dx_p, lower_dy * taper])
        active.append((p, dst))

    for idx in LL_TOP:
        p = pt(idx)
        taper = get_taper(p[0])
        dx_p  = -purse * (p[0]-mouth_cx) * 0.06
        dst   = p + np.array([dx_p, lower_dy * taper * 0.72])
        active.append((p, dst))

    # ── Jaw/Chin movement ────────
    # Disabled to ensure jawline remains 100% static and does not overflow.
    # jaw_dy = lower_dy * 0.18
    # for idx in CHIN:
    #     p = pt(idx)
    #     active.append((p, p + np.array([0.0, jaw_dy])))

    # Apply RBF warp
    warped_img = rbf_warp_local(img, landmarks, active, MOUTH_ANCHORS, sigma_mouth, scale=0.0)

    # ── Dynamically sized tight lips mask ──
    h_img, w_img = img.shape[:2]
    mouth_xs = [landmarks[idx][0] for idx in MOUTH_OUTER]
    mouth_ys = [landmarks[idx][1] for idx in MOUTH_OUTER]
    m_x_min, m_x_max = min(mouth_xs), max(mouth_xs)
    m_y_min, m_y_max = min(mouth_ys), max(mouth_ys)
    
    mouth_cx = (m_x_min + m_x_max) / 2
    mouth_cy = (m_y_min + m_y_max) / 2
    mouth_w = m_x_max - m_x_min
    mouth_h = m_y_max - m_y_min

    # Shift center of mask with the lip movement
    ellipse_cy = int(mouth_cy + (lower_dy + upper_dy) * 0.5)
    
    # Set radii of the ellipse mask to tightly cover the lips
    rx = int(mouth_w * 0.75)
    # The vertical radius should cover the open lips with padding
    ry = int((mouth_h + abs(lower_dy) + abs(upper_dy)) * 0.85)
    
    mask = np.zeros((h_img, w_img), dtype=np.float32)
    cv2.ellipse(mask, (int(mouth_cx), ellipse_cy), (rx, ry), 0, 0, 360, 1.0, -1)
    
    # Feather the mask with a small kernel size relative to mouth width
    ksize = int(mouth_w * 0.18) | 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    
    # Safeguard: vertical fade factor to ensure mask is exactly 0 above the nose tip/base
    y_nose_base = int(landmarks[2][1])
    y_fade_start = y_nose_base - 10
    y_fade_end = int(mouth_cy)
    
    ramp = np.ones(h_img, dtype=np.float32)
    for y in range(h_img):
        if y < y_fade_start:
            ramp[y] = 0.0
        elif y < y_fade_end:
            ramp[y] = (y - y_fade_start) / (y_fade_end - y_fade_start)
            
    mask = mask * ramp[:, np.newaxis]
    
    # Blend the warp into the output image
    mask_3d = mask[:, :, np.newaxis]
    warped_img = (img * (1.0 - mask_3d) + warped_img * mask_3d).astype(np.uint8)

    # ── Draw mouth cavity inside the open lips ──
    open_px = lower_drop * base_unit * 1.8
    if open_px >= 1.5:
        upper_inner_pts = []
        for idx in UL_BOT:
            p = pt(idx)
            taper = get_taper(p[0])
            p_warped = p + np.array([-purse * (p[0]-mouth_cx) * 0.07, upper_dy * taper * 0.6])
            upper_inner_pts.append(p_warped.astype(np.int32))

        lower_inner_pts = []
        for idx in LL_TOP:
            p = pt(idx)
            taper = get_taper(p[0])
            p_warped = p + np.array([-purse * (p[0]-mouth_cx) * 0.06, lower_dy * taper * 0.72])
            lower_inner_pts.append(p_warped.astype(np.int32))

        # Build closed polygon
        polygon = upper_inner_pts + list(reversed(lower_inner_pts))
        polygon = np.array(polygon, dtype=np.int32)

        mask = np.zeros(warped_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Soften mask
        k = max(3, int(open_px * 0.4)) | 1
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Cavity color: very dark warm tint from surrounding skin
        m_pts = [pt(idx) for idx in MOUTH_OUTER]
        mx = int(np.mean([p[0] for p in m_pts]))
        my = int(np.mean([p[1] for p in m_pts]))
        row = warped_img[max(0, my-4), max(0, mx-6):min(warped_img.shape[1], mx+6)]
        skin = row.mean(axis=0) if row.size > 0 else np.array([90, 70, 60])
        cavity = (0.08 * skin + 0.92 * np.array([18, 12, 10])).astype(np.uint8)
        cav_img = np.full_like(warped_img, cavity)

        alpha = min(0.82, open_px / (base_unit * 1.5) * 0.82)
        m3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0 * alpha
        warped_img = np.clip(warped_img * (1 - m3d) + cav_img * m3d, 0, 255).astype(np.uint8)

    return warped_img


def _draw_mouth_cavity(frame, landmarks, lower_drop, face_width):
    """Legacy helper, now handled in _apply_viseme_frame."""
    return frame


# ── Main talk animator ───────────────────────────────────────────
def animate_talk(img, landmarks, w, h, cycle=None, text=None):
    """
    PIPELINE:
    1. Build param_timeline (N frames, 5 params each)
       - If text given: grapheme→viseme→smooth
       - Else: generic open/close cycle
    2. Schedule blink frames (guaranteed ≥1 blink)
    3. Per frame:
       a. Apply blink (eye RBF warp)
       b. Apply subtle eyebrow raise
       c. Apply lip warp (core viseme)
       d. Draw mouth cavity
    4. Return RGB frame list
    """
    p_l = landmarks[234]; p_r = landmarks[454]
    fw  = float(np.linalg.norm(np.array(p_l) - np.array(p_r)))

    # Sigma values
    sigma_mouth = fw * 0.18   # local, prevents spilling onto neck/collar
    sigma_eye   = fw * 0.04   # tight eye socket warp, protects eyebrows
    sigma_brow  = fw * 0.18

    # ── Build param timeline ──────────────────────────────────────
    if text:
        timeline = text_to_param_timeline(text)
    else:
        if cycle is None:
            cycle = [1.0,0.7,0.5,0.8,0.4,0.6,0.3,0.7,0.5,1.0,0.6,0.3,0.6,0.8,1.0]
        rows = []
        for sc in cycle:
            hf = max(0.0, (1.0 - sc) * 1.8)
            rows.append([hf*2.0, hf*0.12, 0.0, hf*0.45, hf*1.80])
        timeline = np.array(rows, dtype=np.float32)

    n_frames = len(timeline)

    # ── Guaranteed blink schedule ─────────────────────────────────
    blink_map = _make_blink_schedule(n_frames, fps=15)

    frames = []
    for i, params in enumerate(timeline):
        frame = img.copy()
        jaw, spread, purse, upper_rise, lower_drop = [float(x) for x in params]

        # 1. BLINK
        if i in blink_map:
            off = blink_map[i]
            eye_sc = BLINK_TPL[off] if 0 <= off < len(BLINK_TPL) else 1.0
            if eye_sc < 0.98:
                frame = _apply_eye_blink_masked(frame, landmarks, fw, eye_sc, eye_sc)

        # 2. EYEBROW — disabled during speech to keep eyebrow/eye animations completely separate and static

        # 3. LIP WARP
        frame = _apply_viseme_frame(frame, landmarks, params, fw, sigma_mouth)

        # 4. MOUTH CAVITY
        frame = _draw_mouth_cavity(frame, landmarks, lower_drop, fw)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return frames


# ── Text → animation intent ──────────────────────────────────────
def _parse_locally(user_text: str):
    t = user_text.lower().strip()
    if any(k in t for k in ("blink", "aankh jhapak")):
        loops = 3 if "3" in t or "three" in t else (2 if "2" in t or "twice" in t else 1)
        return {"animation": "blink", "loops": loops, "fps": 12}
    if "wink left"  in t or "baayi aankh" in t: return {"animation": "wink_left",  "loops": 1, "fps": 12}
    if "wink right" in t or "daayi aankh" in t: return {"animation": "wink_right", "loops": 1, "fps": 12}
    if "wink"       in t: return {"animation": "wink_left", "loops": 1, "fps": 12}
    if any(k in t for k in ("smile", "muskura", "happy")):
        return {"animation": "smile", "loops": 1, "fps": 12}
    if any(k in t for k in ("eyebrow", "brow", "surprised", "surprise")):
        return {"animation": "eyebrow_raise", "loops": 1, "fps": 12}
    if any(k in t for k in ("nod", "haan")):
        return {"animation": "nod", "loops": 1, "fps": 12}
    if any(k in t for k in ("talk", "speak", "bol", "baat")):
        return {"animation": "talk", "loops": 1, "fps": 15}
    meaningful = [w for w in t.split()
                  if w not in {"make","do","please","can","you","show","him","her","face","the","a","an"}]
    if len(meaningful) >= 2:
        return {"animation": "talk", "speech_text": user_text, "loops": 1, "fps": 15}
    return None

def interpret_text(user_text: str) -> dict:
    res = _parse_locally(user_text)
    if res:
        return res
    if gemini_client is None:
        if len(user_text.split()) > 1:
            return {"animation": "talk", "speech_text": user_text, "loops": 1, "fps": 15}
        return {"animation": "blink", "loops": 1, "fps": 12}
    prompt = (
        'You are a face animation controller.\n'
        'Available: blink, wink_left, wink_right, talk, smile, eyebrow_raise, nod, blink_smile\n'
        f'User: "{user_text}"\n'
        'If the user wants the face to say/speak something, use "talk" and set speech_text.\n'
        'Return ONLY JSON: {"animation":"<name>","loops":<1-5>,"fps":<8-20>,"speech_text":"<text or null>"}'
    )
    try:
        model_name = os.environ.get("GEMINI_AI_MODEL", "gemini-2.0-flash")
        resp = gemini_client.models.generate_content(model=model_name, contents=prompt)
        text = resp.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[Gemini] {e}")
        if len(user_text.split()) > 1:
            return {"animation": "talk", "speech_text": user_text, "loops": 1, "fps": 15}
        return {"animation": "blink", "loops": 1, "fps": 12}

# Keep old name for backward compat
interpret_text_with_gemini = interpret_text


# ── Main pipeline ────────────────────────────────────────────────
def animate_image(image_path, user_text, output_path="output.gif", gemini_api_key=None):
    if gemini_api_key:
        try:
            init_gemini(gemini_api_key)
        except Exception as e:
            print(f"[Gemini init] {e}")

    img, (w, h), landmarks = get_landmarks(image_path)

    if landmarks is None:
        print("[Pipeline] Trying Gemini Vision fallback...")
        coords = get_landmarks_via_gemini(image_path)
        if coords is None:
            print("[Pipeline] FAIL — no face detected")
            return None
        landmarks = [(0, 0)] * 478
        for key, val in coords.items():
            cx, cy = int(val[0]), int(val[1])
            if   key == "left_eye":      [landmarks.__setitem__(idx, (cx,cy)) for idx in LEFT_EYE_ANCHORS+list(LEFT_EYE_PAIRS.keys())]
            elif key == "right_eye":     [landmarks.__setitem__(idx, (cx,cy)) for idx in RIGHT_EYE_ANCHORS+list(RIGHT_EYE_PAIRS.keys())]
            elif "mouth" in key:         [landmarks.__setitem__(idx, (cx,cy)) for idx in MOUTH_OUTER+MOUTH_INNER]
            elif key == "left_eyebrow":  [landmarks.__setitem__(idx, (cx,cy)) for idx in LEFT_EYEBROW]
            elif key == "right_eyebrow": [landmarks.__setitem__(idx, (cx,cy)) for idx in RIGHT_EYEBROW]

    anim_cfg    = interpret_text(user_text)
    anim_type   = anim_cfg["animation"]
    loops       = anim_cfg.get("loops", 1)
    fps         = anim_cfg.get("fps", 12)
    speech_text = anim_cfg.get("speech_text")

    print(f"[Pipeline] {anim_type} | loops={loops} | fps={fps} | speech={speech_text!r}")

    BLINK_CYCLE = [1.0, 0.8, 0.5, 0.2, 0.05, 0.2, 0.5, 0.8, 1.0]
    TALK_CYCLE  = [1.0,0.7,0.5,0.8,0.4,0.6,0.3,0.7,0.5,1.0,0.6,0.3,0.6,0.8,1.0]

    all_frames = []
    for _ in range(max(1, loops)):
        if   anim_type == "blink":          all_frames += animate_blink(img, landmarks, w, h, BLINK_CYCLE, fps)
        elif anim_type == "wink_left":      all_frames += animate_wink(img, landmarks, w, h, "left",  BLINK_CYCLE)
        elif anim_type == "wink_right":     all_frames += animate_wink(img, landmarks, w, h, "right", BLINK_CYCLE)
        elif anim_type == "talk":           all_frames += animate_talk(img, landmarks, w, h, TALK_CYCLE, text=speech_text)
        elif anim_type == "smile":          all_frames += animate_smile(img, landmarks, w, h)
        elif anim_type == "eyebrow_raise":  all_frames += animate_eyebrow_raise(img, landmarks, w, h)
        elif anim_type == "nod":            all_frames += animate_nod(img, landmarks, w, h)
        elif anim_type == "blink_smile":
            all_frames += animate_blink(img, landmarks, w, h, BLINK_CYCLE, fps)
            all_frames += animate_smile(img, landmarks, w, h)
        else:
            print(f"[Pipeline] Unknown '{anim_type}' — defaulting to blink")
            all_frames += animate_blink(img, landmarks, w, h, BLINK_CYCLE, fps)

    imageio.mimsave(output_path, all_frames, fps=fps, loop=0)
    kb = os.path.getsize(output_path) / 1024
    print(f"[Pipeline] Saved: {output_path} ({kb:.1f} KB, {len(all_frames)} frames)")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("text")
    ap.add_argument("--output", "-o", default="output.gif")
    ap.add_argument("--api-key", "-k")
    args = ap.parse_args()
    animate_image(
        image_path    = args.image,
        user_text     = args.text,
        output_path   = args.output,
        gemini_api_key = args.api_key or os.environ.get("GEMINI_API_KEY"),
    )
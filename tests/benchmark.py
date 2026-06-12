# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np

# Adjust path to import animate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import animate

def run_benchmark():
    image_path = "b24f6f8a40671b67b98221c102cba557.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("Loading MediaPipe FaceMesh landmarks...")
    img, (w, h), landmarks = animate.get_landmarks(image_path)
    if landmarks is None:
        print("No face detected.")
        return

    text_normal = "Hello, nice to meet you! my name is devanand , what about you man , what you name , tell me ?"
    text_double = text_normal + " " + text_normal

    print("\n================ BENCHMARK RUN ================ ")
    print(f"Normal text: '{text_normal}'")
    print(f"Double text: '{text_double}'")
    print("================================================\n")

    results = []

    for label, text in [("Normal (Length: 95 chars)", text_normal), ("2x Double (Length: 191 chars)", text_double)]:
        print(f"--- Benchmarking: {label} ---")
        
        # Get total frames first
        timeline = animate.text_to_param_timeline(text)
        num_frames = len(timeline)
        
        # 1. Benchmark Legacy RBF Warp (Run on 10 frames to avoid long wait)
        print(f"Running Legacy RBF Warp (Precomputation disabled, dry-run 10 of {num_frames} frames)...")
        start_legacy = time.time()
        _ = animate.animate_talk(img, landmarks, w, h, text=text, use_precompute=False, max_frames=10)
        time_legacy_subset = time.time() - start_legacy
        per_frame_legacy = time_legacy_subset / 10.0
        extrapolated_legacy = per_frame_legacy * num_frames
        print(f"Legacy RBF (10 frames) completed in {time_legacy_subset:.3f} seconds ({per_frame_legacy:.4f}s per frame)")

        # 2. Benchmark Precomputed RBF Warp (Run full timeline)
        print(f"Running Optimized Precomputed RBF Warp (Full timeline: {num_frames} frames)...")
        # Warmup
        _ = animate.animate_talk(img, landmarks, w, h, text=text, use_precompute=True, max_frames=10)
        
        start_pre = time.time()
        _ = animate.animate_talk(img, landmarks, w, h, text=text, use_precompute=True)
        time_pre = time.time() - start_pre
        per_frame_pre = time_pre / num_frames
        print(f"Precomputed RBF (Full timeline) completed in {time_pre:.3f} seconds ({per_frame_pre:.4f}s per frame)")

        speedup = per_frame_legacy / (per_frame_pre + 1e-9)
        print(f"=> Speedup: {speedup:.1f}x faster!\n")

        results.append({
            "label": label,
            "frames": num_frames,
            "legacy_extrapolated": extrapolated_legacy,
            "optimized_actual": time_pre,
            "speedup": speedup
        })

    print("=================== SUMMARY RESULTS ===================")
    print("| Text Case | Frame Count | Legacy RBF (Extrapolated) | Optimized RBF (Actual) | Speedup Factor |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['frames']} | {r['legacy_extrapolated']:.2f}s | {r['optimized_actual']:.2f}s | **{r['speedup']:.1f}x** |")
    print("=======================================================")

if __name__ == "__main__":
    run_benchmark()

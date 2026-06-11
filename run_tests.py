# -*- coding: utf-8 -*-
"""
AI Face Animator - Test Runner
==============================
This script runs the speech animation pipeline at full resolution,
and saves the output GIF and visual frame sequence strip inside the 'tests/' directory.
This ensures only the generated outputs (frames) are stored in the test folder.
"""

import os
import cv2
import numpy as np
from PIL import Image
import animate

def execute_animation_test(image_name, text_phrase, output_gif_name, output_strip_name, max_frames=8):
    # Setup directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(project_dir, "tests")
    
    # Ensure tests directory exists
    os.makedirs(tests_dir, exist_ok=True)
    
    image_path = os.path.join(project_dir, image_name)
    gif_path = os.path.join(tests_dir, output_gif_name)
    strip_path = os.path.join(tests_dir, output_strip_name)
    
    print(f"[Test Suite] Target Image: {image_path}")
    print(f"[Test Suite] Speech Phrase: '{text_phrase}'")
    
    # 1. Run speech animation
    try:
        animate.animate_image(
            image_path=image_path,
            user_text=text_phrase,
            output_path=gif_path
        )
    except Exception as e:
        print(f"[Test Suite Error] Animation generation failed: {e}")
        return False
        
    if not os.path.exists(gif_path):
        print(f"[Test Suite Error] Output GIF not found at: {gif_path}")
        return False
        
    print(f"[Test Suite] Speech GIF saved to: {gif_path}")
    
    # 2. Extract and stitch frames horizontally for sequential analysis
    gif = Image.open(gif_path)
    frames = []
    
    try:
        while True:
            # Convert frame to RGB format
            frame_rgb = gif.convert('RGB')
            frames.append(np.array(frame_rgb))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
        
    total_frames = len(frames)
    print(f"[Test Suite] Extracted {total_frames} frames from GIF.")
    
    if total_frames == 0:
        return False
        
    # Pick N evenly spaced frames across the duration of the animation
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    # Stitch frames horizontally (resize each to height 240px for clean layout)
    target_h = 240
    resized_frames = []
    
    for idx, f in enumerate(selected_frames):
        h, w = f.shape[:2]
        target_w = int(w * (target_h / h))
        r_f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Add a visual frame index label (F-1, F-2, ...)
        cv2.putText(r_f, f"F-{idx + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        resized_frames.append(r_f)
        
    # Stack frames horizontally
    strip_img = np.hstack(resized_frames)
    
    # Save the frame strip in the tests/ directory
    Image.fromarray(strip_img).save(strip_path)
    print(f"[Test Suite] Visual frame sequence strip saved to: {strip_path}")
    return True

if __name__ == "__main__":
    execute_animation_test(
        image_name="b24f6f8a40671b67b98221c102cba557.jpg",
        text_phrase="Good morning",
        output_gif_name="test_speech.gif",
        output_strip_name="test_strip_jaw_fixed.png"
    )

import cv2
import numpy as np
from PIL import Image
import os

def generate_strip(gif_path, output_path, max_frames=8):
    if not os.path.exists(gif_path):
        print(f"Error: GIF path {gif_path} does not exist.")
        return False
        
    gif = Image.open(gif_path)
    frames = []
    
    # Extract all frames
    try:
        while True:
            # Convert frame to RGB numpy array
            frame_rgb = gif.convert('RGB')
            frames.append(np.array(frame_rgb))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
        
    total_frames = len(frames)
    print(f"Extracted {total_frames} frames from {gif_path}")
    
    if total_frames == 0:
        return False
        
    # Select max_frames spaced evenly
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    # Resize frames to a uniform smaller size for the strip (e.g., height 240px)
    target_h = 240
    resized_frames = []
    for f in selected_frames:
        h, w = f.shape[:2]
        target_w = int(w * (target_h / h))
        r_f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Draw frame number on each frame
        cv2.putText(r_f, f"F-{resized_frames.__len__() + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        resized_frames.append(r_f)
        
    # Stitch horizontally
    strip = np.hstack(resized_frames)
    
    # Convert back to PIL Image and save
    Image.fromarray(strip).save(output_path)
    print(f"Saved frame sequence strip to {output_path} (width={strip.shape[1]}, height={strip.shape[0]})")
    return True

if __name__ == "__main__":
    generate_strip("test_b24.gif", "frame_strip.png")

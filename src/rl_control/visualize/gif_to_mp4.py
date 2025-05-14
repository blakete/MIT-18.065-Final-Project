#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import imageio
from PIL import Image, ImageSequence

def gif_to_mp4_opencv(input_path: str, output_path: str):
    gif = Image.open(input_path)
    width, height = gif.size
    duration_ms = gif.info.get('duration', 100)   # ms per frame
    fps = 1000.0 / duration_ms

    # Try a couple of fourccs in case one isn’t supported
    for codec in ('mp4v','avc1'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec '{codec}' → {fps:.2f} FPS")
            break
        else:
            writer.release()
            print(f"Codec '{codec}' failed, trying next…")
    else:
        writer = None

    if not writer or not writer.isOpened():
        print("OpenCV VideoWriter failed for all codecs.")
        return False

    for frame in ImageSequence.Iterator(gif):
        frame_rgb = frame.convert('RGB')
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return True

def gif_to_mp4_imageio(input_path: str, output_path: str):
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    # meta['duration'] is seconds per frame
    duration_s = meta.get('duration', 0.1)
    fps = 1.0 / duration_s

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p'
    )
    for frame in reader:
        writer.append_data(frame)
    writer.close()
    print(f"imageio → '{output_path}' @ {fps:.2f} FPS")
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python gif_to_mp4.py input.gif output.mp4")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]

    # First try OpenCV
    if gif_to_mp4_opencv(inp, outp):
        print("✅ OpenCV conversion succeeded.")
        sys.exit(0)

    # Fallback to imageio/ffmpeg
    try:
        import imageio
    except ImportError:
        print("❌ imageio not installed. Install with `pip install imageio`")
        sys.exit(1)

    if gif_to_mp4_imageio(inp, outp):
        print("✅ imageio conversion succeeded.")
        sys.exit(0)
    else:
        print("❌ Both conversions failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

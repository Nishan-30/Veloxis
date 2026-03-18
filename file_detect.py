# ============================================================
#  file_detect.py  –  Analyse a saved video file
#  Run:  python file_detect.py
# ============================================================

import cv2
import os
import sys
import datetime
from detector import VehicleDetector
import config


def run_file_detection(video_path: str):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    print(f"\n[FILE MODE] Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video. Check the file path.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_actual   = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    duration_sec = total_frames / fps_actual
    print(f"[INFO] Frames: {total_frames}  |  FPS: {fps_actual:.1f}  |  Duration: {duration_sec:.1f}s")

    # Update fps in config so speed is accurate
    config.VIDEO_FPS = fps_actual

    label    = os.path.splitext(os.path.basename(video_path))[0]
    detector = VehicleDetector(session_label=label)

    # Optional: set up video writer to save output
    out_path = os.path.join(config.DATA_FOLDER, f"output_{label}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer   = cv2.VideoWriter(out_path, fourcc, fps_actual, (w, h))

    frame_count = 0
    print("[INFO] Processing... Press 'q' to quit early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated, summary = detector.process_frame(frame)
        writer.write(annotated)

        # Progress every 50 frames
        if frame_count % 50 == 0:
            pct = (frame_count / total_frames * 100) if total_frames else 0
            print(f"  Frame {frame_count}/{total_frames} ({pct:.0f}%)  |  "
                  f"Vehicles so far: {summary['total_unique']}", end="\r")

        if config.SHOW_WINDOW:
            # Resize for display
            disp = cv2.resize(annotated, (config.WINDOW_WIDTH,
                                          int(h * config.WINDOW_WIDTH / w)))
            cv2.imshow("Vehicle Counter — File Mode (q to quit)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Stopped by user.")
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    detector.print_summary()
    print(f"[INFO] Annotated video saved: {out_path}")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("\n[FILE DETECTOR]")
        print("Put your video files inside the 'videos/' folder.")
        path = input("Enter video file path (e.g. videos/dhaka_road.mp4): ").strip()
        if not path:
            path = "videos/test.mp4"

    run_file_detection(path)

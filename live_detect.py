# ============================================================
#  live_detect.py  –  Real-time detection from camera
#  Supports: laptop webcam, USB camera, phone via DroidCam
#  Run:  python live_detect.py
# ============================================================

import cv2
import sys
import datetime
from detector import VehicleDetector
import config


def get_camera_source():
    """
    Returns the OpenCV camera source.
    0 = built-in webcam
    1, 2 = USB camera
    "http://..." = DroidCam / IP camera
    """
    print("\n[LIVE MODE] Camera source:")
    print("  1. Laptop webcam (default)")
    print("  2. USB camera")
    print("  3. Phone via DroidCam (WiFi)")
    print("  4. RTSP / IP camera URL")
    choice = input("\nEnter choice (1–4) or press Enter for default: ").strip()

    if choice == "2":
        idx = input("Camera index (usually 1 or 2): ").strip()
        return int(idx) if idx.isdigit() else 1
    elif choice == "3":
        print("\nDroidCam setup:")
        print("  1. Install DroidCam app on your Android phone (free on Play Store)")
        print("  2. Connect phone and laptop to the SAME WiFi")
        print("  3. Open DroidCam app — note the IP address shown")
        ip = input("  Enter DroidCam IP (e.g. 192.168.1.5): ").strip()
        port = input("  Enter port (default 4747): ").strip() or "4747"
        return f"http://{ip}:{port}/video"
    elif choice == "4":
        url = input("Enter full camera URL: ").strip()
        return url
    else:
        return 0   # default: built-in webcam


def run_live_detection(source=0):
    print(f"\n[INFO] Opening camera source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  → For DroidCam: make sure phone and laptop are on the same WiFi.")
        print("  → For webcam: check that no other app is using the camera.")
        return

    # Try to set a good resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {w}×{h}")
    print("[INFO] Press 'q' to stop and save results.\n")

    label    = "live_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    detector = VehicleDetector(session_label=label)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Lost camera feed. Retrying...")
            continue

        annotated, summary = detector.process_frame(frame)

        if config.SHOW_WINDOW:
            disp = cv2.resize(annotated, (config.WINDOW_WIDTH,
                                          int(h * config.WINDOW_WIDTH / w)))
            cv2.imshow("Vehicle Counter — Live Mode (q to quit)", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.print_summary()


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow passing source directly: python live_detect.py 0
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg
    else:
        source = get_camera_source()

    run_live_detection(source)

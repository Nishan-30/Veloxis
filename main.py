# ============================================================
#  main.py  –  Main menu — start everything from here
#  Run:  python main.py
# ============================================================

import os
import sys


BANNER = r"""
  ______ __  __  ___  _____  _________  ___ __________
 /_  __// / / / / _ \/ __  \/ _/ ___/ |/ / //_  __/ _ \
  / /  / /_/ / / , _/ /_/ // // /__ >   <    / / / ___/
 /_/   \____/ /_/|_|\____/___/\___//_/|_|   /_/ /_/

  Bangladesh Road Vehicle Counter — by Nishan (SUST CEE)
  100%% free | YOLO v8 + DeepSORT | Flask dashboard
"""


def menu():
    print(BANNER)
    print("  Choose a mode:\n")
    print("  [1]  Live detection   — webcam or phone camera (real-time)")
    print("  [2]  File detection   — analyse a saved video file")
    print("  [3]  Draw lanes       — click to define roads/zones for counting")
    print("  [4]  Open dashboard   — graphs in your browser")
    print("  [5]  Exit\n")
    choice = input("  Enter 1–4: ").strip()
    return choice


def main():
    while True:
        choice = menu()

        if choice == "1":
            from live_detect import get_camera_source, run_live_detection
            source = get_camera_source()
            run_live_detection(source)

        elif choice == "2":
            from file_detect import run_file_detection
            print("\n[FILE MODE]")
            print("Put your .mp4 / .avi video inside the 'videos/' folder first.")
            path = input("Video path (e.g. videos/dhaka_road.mp4): ").strip()
            if not path:
                print("[ERROR] No path given.")
            else:
                run_file_detection(path)

        elif choice == "3":
            from lane_tool import LaneTool
            print("\n[LANE TOOL]")
            print("Open a video and click to draw lane boundaries.")
            path = input("Video path (e.g. videos/dhaka_road.mp4): ").strip()
            if not path:
                print("[ERROR] No path given.")
            else:
                tool = LaneTool(path)
                tool.run()

        elif choice == "4":
            from dashboard import app
            import config, os
            os.makedirs(config.DATA_FOLDER, exist_ok=True)
            print("\n[DASHBOARD] Open http://localhost:5000 in your browser.")
            print("Press Ctrl+C to stop the dashboard.\n")
            app.run(debug=False, port=5000)

        elif choice == "5":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("[!] Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()

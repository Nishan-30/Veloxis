"""
build_exe.py  —  Creates TrafficCounter.exe for Windows
Run:  python build_exe.py

Output: dist/TrafficCounter/TrafficCounter.exe
"""
import subprocess, sys, os

def main():
    print("=" * 52)
    print("  Building TrafficCounter.exe ...")
    print("=" * 52)

    # Install PyInstaller if not present
    subprocess.check_call([sys.executable, "-m", "pip",
                           "install", "pyinstaller", "PyQt6",
                           "--quiet"])

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name",       "TrafficCounter",
        "--onedir",                         # folder (faster startup than onefile)
        "--windowed",                       # no console window
        "--noconfirm",
        "--clean",

        # Hidden imports PyInstaller often misses
        "--hidden-import", "ultralytics",
        "--hidden-import", "deep_sort_realtime",
        "--hidden-import", "cv2",
        "--hidden-import", "flask",
        "--hidden-import", "matplotlib",
        "--hidden-import", "pandas",

        # Include config + detector files
        "--add-data", f"config.py{os.pathsep}.",
        "--add-data", f"detector.py{os.pathsep}.",
        "--add-data", f"lane_tool.py{os.pathsep}.",
        "--add-data", f"live_detect.py{os.pathsep}.",
        "--add-data", f"file_detect.py{os.pathsep}.",
        "--add-data", f"dashboard.py{os.pathsep}.",

        "app_windows.py",                   # main entry
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        exe_path = os.path.join("dist", "TrafficCounter", "TrafficCounter.exe")
        print("\n" + "="*52)
        print("  BUILD SUCCESSFUL!")
        print(f"  EXE: {exe_path}")
        print("\n  Copy the entire 'dist/TrafficCounter/' folder")
        print("  to any Windows PC — no Python needed!")
        print("="*52)
    else:
        print("\n[ERROR] Build failed. Check messages above.")

if __name__ == "__main__":
    main()

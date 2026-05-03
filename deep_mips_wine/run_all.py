import subprocess
import sys
import time
from pathlib import Path

def main():
    steps = [
        ("01_prepare.py",  "Data Preparation"),
        ("02_train.py",    "Model Training"),
        ("03_convert.py",  "Weight Conversion"),
        ("04_compile.py",  "MIPS Compilation"),
        ("05_verify.py",   "Verification Report"),
    ]

    print("=" * 60)
    print("Deep-MIPS Wine Quality - Full Pipeline")
    print("=" * 60)

    for script, name in steps:
        print(f"\n[STEP] {name}...")
        t_start = time.time()
        result = subprocess.run([sys.executable, script], capture_output=False)
        t_end = time.time()
        if result.returncode != 0:
            print(f"[FAILED] {name} failed. Stopping pipeline.")
            sys.exit(1)
        print(f"[DONE]  {name} ({t_end - t_start:.1f}s)")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print("Next step: open outputs/wine_model.asm in MARS simulator")
    print("Compare MARS output to outputs/verification_report.txt")

if __name__ == "__main__":
    main()

import subprocess
import os

def run_gpen_test():
    source = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
    target = "/Users/kendrick/Documents/FS Source/zBam_noOcclusion.png"
    output = "previewtest/test_gpen_2048.png"
    
    cmd = [
        "./venv/bin/python", "watserface.py", "headless-run",
        "-s", source,
        "-t", target,
        "-o", output,
        "--processors", "face_swapper", "face_enhancer",
        "--face-swapper-model", "simswap_unofficial_512",
        "--face-landmarker-model", "2dfan4",
        "--face-mask-types", "region",
        "--face-enhancer-model", "gpen_bfr_2048", # Trying the 2K enhancer
        "--face-enhancer-blend", "100",
        "--output-image-quality", "100"
    ]
    
    print(f"Running GPEN 2048 test...")
    subprocess.run(cmd, check=True)
    print(f"Saved to {output}")

if __name__ == "__main__":
    run_gpen_test()

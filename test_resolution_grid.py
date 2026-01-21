import cv2
import subprocess
import os

def run_cmd(args):
    cmd = ["./venv/bin/python", "watserface.py", "headless-run"] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def create_grid():
    source = "/Users/kendrick/Downloads/Gemini_Generated_Image_vhmelnvhmelnvhme.png"
    target = "/Users/kendrick/Documents/zBam_noOcclusion.png"
    
    # 1. Inswapper 128
    run_cmd([
        "-s", source, "-t", target, "-o", "previewtest/grid_1_inswapper128.png",
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "--face-mask-types", "box"
    ])
    
    # 2. Hyperswap 256
    run_cmd([
        "-s", source, "-t", target, "-o", "previewtest/grid_2_hyperswap256.png",
        "--processors", "face_swapper",
        "--face-swapper-model", "hyperswap_1a_256",
        "--face-mask-types", "box"
    ])
    
    # 3. SimSwap 512
    run_cmd([
        "-s", source, "-t", target, "-o", "previewtest/grid_3_simswap512.png",
        "--processors", "face_swapper",
        "--face-swapper-model", "simswap_unofficial_512",
        "--face-mask-types", "box"
    ])
    
    # 4. SimSwap 512 + Boost 1024
    run_cmd([
        "-s", source, "-t", target, "-o", "previewtest/grid_4_simswap512_boost.png",
        "--processors", "face_swapper",
        "--face-swapper-model", "simswap_unofficial_512",
        "--face-swapper-pixel-boost", "1024x1024",
        "--face-mask-types", "box"
    ])

    print("Generating Grid Image...")
    img1 = cv2.imread("previewtest/grid_1_inswapper128.png")
    img2 = cv2.imread("previewtest/grid_2_hyperswap256.png")
    img3 = cv2.imread("previewtest/grid_3_simswap512.png")
    img4 = cv2.imread("previewtest/grid_4_simswap512_boost.png")

    # Resize all to match img1 (if target size varies, which it shouldn't)
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    img3 = cv2.resize(img3, (w, h))
    img4 = cv2.resize(img4, (w, h))

    # Add Labels
    cv2.putText(img1, "Inswapper 128", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img2, "Hyperswap 256", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img3, "SimSwap 512", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img4, "SimSwap 512 + Boost", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Stitch
    top = cv2.hconcat([img1, img2])
    bottom = cv2.hconcat([img3, img4])
    final = cv2.vconcat([top, bottom])
    
    cv2.imwrite("previewtest/resolution_grid.png", final)
    print("Saved previewtest/resolution_grid.png")

if __name__ == "__main__":
    create_grid()

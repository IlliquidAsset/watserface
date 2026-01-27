import cv2
import subprocess
import os
import numpy as np

def run_swap(output_name, enhancer, enhancer_blend):
    source = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
    target = "/Users/kendrick/Documents/FS Source/zBam_noOcclusion.png"
    output_path = f"previewtest/{output_name}"
    
    cmd = [
        "./venv/bin/python", "watserface.py", "headless-run",
        "-s", source,
        "-t", target,
        "-o", output_path,
        "--processors", "face_swapper", "face_enhancer",
        "--face-swapper-model", "simswap_unofficial_512",
        "--face-landmarker-model", "2dfan4",
        "--face-mask-types", "region", 
        "--face-mask-regions", "skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip",
        "--face-mask-blur", "0.05", # Tighter blur for feathering without bleeding
        "--face-enhancer-model", enhancer,
        "--face-enhancer-blend", str(enhancer_blend),
        "--output-image-quality", "100"
    ]
    
    print(f"Generating {output_path} with {enhancer}...")
    subprocess.run(cmd, check=True)
    return output_path

def post_process_blend(swap_path, target_path, output_path):
    print(f"Blending original background back into {swap_path}...")
    
    # Load images
    swap = cv2.imread(swap_path)
    target = cv2.imread(target_path)
    
    if swap is None or target is None:
        print("Error loading images for blending.")
        return

    # Ensure same size (swap might be slightly different if padding/crop logic differed, but shouldn't be here)
    if swap.shape != target.shape:
        swap = cv2.resize(swap, (target.shape[1], target.shape[0]))

    # Calculate difference to find the "Swapped Region"
    # This is a hacky way to find the mask if we don't have it directly.
    # Ideally, we'd output the mask from the tool. 
    # Since we can't easily get the internal mask, we'll assume the swapper
    # changed pixels.
    diff = cv2.absdiff(swap, target)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 1, 255, cv2.THRESH_BINARY)
    
    # Dilate slightly to include the blending edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Blur the mask for seamless composition
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Convert to float 0-1
    alpha = mask.astype(float) / 255.0
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    
    # Composite: Swap where changed, Original where not
    # Final = Swap * Alpha + Target * (1 - Alpha)
    final = (swap.astype(float) * alpha) + (target.astype(float) * (1.0 - alpha))
    
    cv2.imwrite(output_path, final.astype(np.uint8))
    print(f"Saved strict blend to {output_path}")

def main():
    target = "/Users/kendrick/Documents/FS Source/zBam_noOcclusion.png"
    
    # Test 1: CodeFormer (Good skin) with reduced blend to fix "glow"
    # Reducing blend mixes in some of the raw SimSwap result which might look more natural?
    # Or actually, let's keep high blend but strictly mask it.
    swap_1 = run_swap("temp_codeformer_strict.png", "codeformer", 100)
    post_process_blend(swap_1, target, "previewtest/test_strict_codeformer.png")

    # Test 2: RestoreFormer++ (Better eyes usually) 
    swap_2 = run_swap("temp_restoreformer_strict.png", "restoreformer_plus_plus", 100)
    post_process_blend(swap_2, target, "previewtest/test_strict_restoreformer.png")

if __name__ == "__main__":
    main()

import subprocess
import os

def run_cmd(output_name, extra_args):
    source = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
    target = "/Users/kendrick/Documents/FS Source/zBam_noOcclusion.png"
    output = f"previewtest/{output_name}"
    
    base_cmd = [
        "./venv/bin/python", "watserface.py", "headless-run",
        "-s", source,
        "-t", target,
        "-o", output,
        "--face-swapper-model", "simswap_unofficial_512",
        "--face-landmarker-model", "2dfan4",
        "--face-mask-types", "region",
        "--output-image-quality", "100"
    ]
    
    full_cmd = base_cmd + extra_args
    print(f"Generating {output}...")
    subprocess.run(full_cmd, check=True)

def main():
    # 1. RestoreFormer++ (Known for better skin texture/realism than GFPGAN)
    run_cmd("opt_1_restoreformer.png", [
        "--processors", "face_swapper", "face_enhancer",
        "--face-enhancer-model", "restoreformer_plus_plus",
        "--face-enhancer-blend", "100"
    ])

    # 2. CodeFormer (Known for maximum sharpness, sometimes 'plastic')
    run_cmd("opt_2_codeformer.png", [
        "--processors", "face_swapper", "face_enhancer",
        "--face-enhancer-model", "codeformer",
        "--face-enhancer-blend", "100"
    ])

    # 3. GFPGAN + Frame Enhancer (To crisp up the context as well)
    run_cmd("opt_3_gfpgan_frame_enhanced.png", [
        "--processors", "face_swapper", "face_enhancer", "frame_enhancer",
        "--face-enhancer-model", "gfpgan_1.4",
        "--face-enhancer-blend", "80",
        "--frame-enhancer-model", "real_esrgan_x4",
        "--frame-enhancer-blend", "50" 
    ])

    print("\nOptimization Batch Complete.")
    print("1. previewtest/opt_1_restoreformer.png")
    print("2. previewtest/opt_2_codeformer.png")
    print("3. previewtest/opt_3_gfpgan_frame_enhanced.png")

if __name__ == "__main__":
    main()

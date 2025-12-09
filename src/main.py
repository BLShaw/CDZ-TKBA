# DEPRECATED

import os
import sys
import subprocess

def main():
    print("=== Multimodal Brain Project Pipeline ===")
    print("This master script executes the pipeline steps in sequence.")
    
    # Define scripts
    scripts = [
        "scripts/1_prepare_data.py",
        "scripts/2_generate_encodings.py",
        "scripts/3_run_TKBA.py",
        "scripts/4_evaluate.py",
        "scripts/5_visualize.py"
    ]
    
    enc_dir = os.path.join('data', 'encodings')
    train_enc_exists = os.path.exists(os.path.join(enc_dir, 'visual_train_encodings.npy'))
    
    for script in scripts:
        script_path = script.replace('/', os.sep)
        
        if "2_generate_encodings.py" in script and train_enc_exists:
            print(f"\n[INFO] Encodings found. Skipping {script}...")
            continue
            
        print(f"\n>>> Running {script}...")
        try:
            result = subprocess.run([sys.executable, script_path], check=True)
            if result.returncode != 0:
                print(f"[ERROR] {script} failed with code {result.returncode}")
                break
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {script}: {e}")
            break
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            break

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()

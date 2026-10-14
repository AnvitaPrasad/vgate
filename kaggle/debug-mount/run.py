import subprocess
print(subprocess.run("ls -la /kaggle/input; find /kaggle/input -maxdepth 4 | head -60; find /kaggle/input -name '*.zip' | head", shell=True, capture_output=True, text=True).stdout)

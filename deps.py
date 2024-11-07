import os
import subprocess
print("\033[34mInstalling xformers...\033[0m")
subprocess.run("pip install xformers==0.0.28.post2", shell=True)

print("\033[34mFixing dependencies...\033[0m")
subprocess.run("pip install prodigyopt==1.0 onnxruntime==1.17.3", shell=True)
subprocess.run("pip install flax==0.8.4 jax==0.4.23 jaxlib==0.4.23 install opencv-python-headless==4.9.0.80", shell=True)
#subprocess.run("", shell=True)
#subprocess.run("pip install huggingface-hub==0.20.3", shell=True)
subprocess.run("pip uninstall -y pygobject", shell=True)

print("\033[34mChecking dependencies...\033[0m")
result = subprocess.run(['pip', 'check'], capture_output=True, text=True); print("\n".join([f"\033[1;33m- {line}\033[0m" for line in result.stdout.splitlines()]) if result.stdout else "No dependency issues found.")

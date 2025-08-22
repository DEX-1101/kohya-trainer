import os
import subprocess
print("\033[34mInstalling xformers...\033[0m")
subprocess.run("pip install xformers==0.0.32.post2", shell=True)

print("\033[34mFixing dependencies...\033[0m")
subprocess.run("pip install prodigyopt==1.0 onnxruntime==1.17.3", shell=True)
subprocess.run("pip install flax==0.8.4 jax==0.4.23 jaxlib==0.4.23 opencv-python-headless", shell=True)
subprocess.run("pip install httpx==0.28.1 numpy==1.26.4 protobuf==5.29.1 open-clip-torch==2.32.0 wandb==0.21.0 diffusers==0.33.1 jedi==0.19.2 huggingface-hub==0.34.4 opencv-python==4.10.0.82", shell=True)
subprocess.run("pip uninstall -y pygobject salesforce-lavis", shell=True)

print("\033[34mChecking dependencies...\033[0m")
result = subprocess.run(['pip', 'check'], capture_output=True, text=True); print("\n".join([f"\033[1;33m- {line}\033[0m" for line in result.stdout.splitlines()]) if result.stdout else "No dependency issues found.")

import os
import subprocess
print("\033[34mInstalling xformers...\033[0m")
subprocess.run("pip install -q xformers==0.0.28.post1", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\033[34mFixing dependencies...\033[0m")
subprocess.run("pip -q install prodigyopt==1.0 onnxruntime==1.17.3", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run("pip install -q flax==0.8.4 jax==0.4.23 jaxlib==0.4.23 opencv-python-headless==4.9.0.80", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os
import subprocess
print("\033[34mInstalling xformers...\033[0m")
subprocess.run("pip install xformers==0.0.28.post2", shell=True)

print("\033[34mFixing dependencies...\033[0m")
subprocess.run("pip install prodigyopt==1.0 onnxruntime==1.17.3", shell=True)
subprocess.run("pip install flax==0.8.4 jax==0.4.23 jaxlib==0.4.23 opencv-python-headless accelerate==0.21.0 huggingface-hub transformers==4.41.0", shell=True)

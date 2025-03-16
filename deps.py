import os
import subprocess
print("\033[34mInstalling xformers...\033[0m")
subprocess.run("pip install -q xformers==0.0.29.post3", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\033[34mFixing dependencies...\033[0m")
subprocess.run("pip -q install prodigyopt==1.0 onnxruntime==1.17.3", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run("pip install -q flax==0.8.4 jax==0.4.23 jaxlib==0.4.23 opencv-python-headless", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run("pip uninstall -y salesforce-lavis pygobject", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\033[34mChecking requirements...\033[0m")
result = subprocess.run(['pip', 'check'], capture_output=True, text=True); print("\n".join([f"\033[1;33m- {line}\033[0m" for line in result.stdout.splitlines()]) if result.stdout else "No dependency issues found.")

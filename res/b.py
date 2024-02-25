#Download SDXL
import os
import re
import json
import glob
import gdown
import requests
import subprocess
from IPython.utils import capture
from urllib.parse import urlparse, unquote
from pathlib import Path
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError

%store -r

os.chdir(root_dir)

# @markdown Place your Huggingface token [here](https://huggingface.co/settings/tokens) to download gated models.

HUGGINGFACE_TOKEN     = "" #@param {type: "string"}
LOAD_DIFFUSERS_MODEL  = True #@param {type: "boolean"}
SDXL_MODEL_URL        = "https://huggingface.co/cagliostrolab/animagine-xl-3.0/resolve/main/animagine-xl-3.0.safetensors" # @param ["gsdf/CounterfeitXL", "Linaqruf/animagine-xl", "stabilityai/stable-diffusion-xl-base-1.0", "PASTE MODEL URL OR GDRIVE PATH HERE"] {allow-input: true}
SDXL_VAE_URL          = "FP16 VAE" # @param ["None", "Original VAE", "FP16 VAE", "PASTE VAE URL OR GDRIVE PATH HERE"] {allow-input: true}

MODEL_URLS = {
    "gsdf/CounterfeitXL"        : "https://huggingface.co/gsdf/CounterfeitXL/resolve/main/CounterfeitXL_%CE%B2.safetensors",
    "Linaqruf/animagine-xl"   : "https://huggingface.co/Linaqruf/animagine-xl/resolve/main/animagine-xl.safetensors",
    "stabilityai/stable-diffusion-xl-base-1.0" : "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
}
VAE_URLS = {
    "None"                    : "",
    "Original VAE"           : "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
    "FP16 VAE"           : "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors"
}

SDXL_MODEL_URL = MODEL_URLS.get(SDXL_MODEL_URL, SDXL_MODEL_URL)
SDXL_VAE_URL = VAE_URLS.get(SDXL_VAE_URL, SDXL_VAE_URL)

def get_filename(url):
    if any(url.endswith(ext) for ext in [".ckpt", ".safetensors", ".pt", ".pth"]):
        return os.path.basename(url)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    if 'content-disposition' in response.headers:
        filename = re.findall('filename="?([^"]+)"?', response.headers['content-disposition'])[0]
    else:
        filename = unquote(os.path.basename(urlparse(url).path))

    return filename

def aria2_download(dir, filename, url):
    user_header = f"Authorization: Bearer {HUGGINGFACE_TOKEN}"
    aria2_args = [
        "aria2c",
        "--console-log-level=error",
        "--summary-interval=10",
        f"--header={user_header}" if "huggingface.co" in url else "",
        "--continue=true",
        "--max-connection-per-server=16",
        "--min-split-size=1M",
        "--split=16",
        f"--dir={dir}",
        f"--out={filename}",
        url
    ]
    subprocess.run(aria2_args)

def download(url, dst):
    print(f"Starting downloading from {url}")
    filename = get_filename(url)
    filepath = os.path.join(dst, filename)

    if "drive.google.com" in url:
        gdown.download(url, filepath, quiet=False)
    else:
        if "huggingface.co" in url and "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        aria2_download(dst, filename, url)

    print(f"Download finished: {filepath}")
    return filepath

def all_folders_present(base_model_url, sub_folders):
    fs = HfFileSystem()
    existing_folders = set(fs.ls(base_model_url, detail=False))

    for folder in sub_folders:
        full_folder_path = f"{base_model_url}/{folder}"
        if full_folder_path not in existing_folders:
            return False
    return True

def get_total_ram_gb():
    with open('/proc/meminfo', 'r') as f:
        for line in f.readlines():
            if "MemTotal" in line:
                return int(line.split()[1]) / (1024**2)  # Convert to GB

def get_gpu_name():
    try:
        return subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits", shell=True).decode('ascii').strip()
    except:
        return None

def main():
    global model_path, vae_path, LOAD_DIFFUSERS_MODEL

    model_path, vae_path = None, None

    required_sub_folders = [
        'scheduler',
        'text_encoder',
        'text_encoder_2',
        'tokenizer',
        'tokenizer_2',
        'unet',
        'vae',
    ]

    download_targets = {
        "model": (SDXL_MODEL_URL, pretrained_model),
        "vae": (SDXL_VAE_URL, vae_dir),
    }

    total_ram = get_total_ram_gb()
    gpu_name = get_gpu_name()

    # Check hardware constraints
    if total_ram < 13 and gpu_name in ["Tesla T4", "Tesla V100"]:
        print("Attempt to load diffusers model instead due to hardware constraints.")
        if not LOAD_DIFFUSERS_MODEL:
            LOAD_DIFFUSERS_MODEL = True

    for target, (url, dst) in download_targets.items():
        if url and not url.startswith(f"PASTE {target.upper()} URL OR GDRIVE PATH HERE"):
            if target == "model" and LOAD_DIFFUSERS_MODEL:
                # Code for checking and handling diffusers model
                if 'huggingface.co' in url:
                    match = re.search(r'huggingface\.co/([^/]+)/([^/]+)', SDXL_MODEL_URL)
                    if match:
                        username = match.group(1)
                        model_name = match.group(2)
                        url = f"{username}/{model_name}"
                if all_folders_present(url, required_sub_folders):
                    print(f"Diffusers model is loaded : {url}")
                    model_path = url
                else:
                    print("Repository doesn't exist or no diffusers model detected.")
                    filepath = download(url, dst)  # Continue with the regular download
                    model_path = filepath
            else:
                filepath = download(url, dst)

                if target == "model":
                    model_path = filepath
                elif target == "vae":
                    vae_path = filepath

            print()

    if model_path:
        print(f"Selected model: {model_path}")

    if vae_path:
        print(f"Selected VAE: {vae_path}")

main()

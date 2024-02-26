import os
import zipfile
import shutil
import time
import requests
import torch
from subprocess import getoutput
from IPython.utils import capture
#from google.colab import drive

repo_dict = {
    "qaneel/kohya-trainer (forked repo, stable, optimized for colab use)" : "https://github.com/qaneel/kohya-trainer",
    "kohya-ss/sd-scripts (original repo, latest update)"                    : "https://github.com/kohya-ss/sd-scripts",
}

repository        = "qaneel/kohya-trainer (forked repo, stable, optimized for colab use)" #@param ["qaneel/kohya-trainer (forked repo, stable, optimized for colab use)", "kohya-ss/sd-scripts (original repo, latest update)"] {allow-input: true}
repo_url          = repo_dict[repository]
branch            = "main"  # @param {type: "string"}
output_to_drive   = False  # @param {type: "boolean"}

def clone_repo(url, dir, branch):
    if not os.path.exists(dir):
       os.system('git clone -b {branch} {url} {dir}')

def mount_drive(dir):
    output_dir      = os.path.join(training_dir, "output")

    if output_to_drive:
        if not os.path.exists(drive_dir):
            drive.mount(os.path.dirname(drive_dir))
        output_dir  = os.path.join(drive_dir, "kohya-trainer/output")

    return output_dir

def setup_directories():
    global output_dir

    output_dir      = mount_drive(drive_dir)

    for dir in [training_dir, config_dir, pretrained_model, vae_dir, repositories_dir, output_dir]:
        os.makedirs(dir, exist_ok=True)

def pastebin_reader(id):
    if "pastebin.com" in id:
        url = id
        if 'raw' not in url:
                url = url.replace('pastebin.com', 'pastebin.com/raw')
    else:
        url = "https://pastebin.com/raw/" + id
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.split('\n')
    return lines

def install_repository():
    global infinite_image_browser_dir, voldy, discordia_archivum_dir

    _, voldy = pastebin_reader("kq6ZmHFU")[:2]

    infinite_image_browser_url  = f"https://github.com/zanllp/{voldy}-infinite-image-browsing.git"
    infinite_image_browser_dir  = os.path.join(repositories_dir, f"infinite-image-browsing")
    infinite_image_browser_deps = os.path.join(infinite_image_browser_dir, "requirements.txt")

    discordia_archivum_url = "https://github.com/Linaqruf/discordia-archivum"
    discordia_archivum_dir = os.path.join(repositories_dir, "discordia-archivum")
    discordia_archivum_deps = os.path.join(discordia_archivum_dir, "requirements.txt")

    clone_repo(infinite_image_browser_url, infinite_image_browser_dir, "main")
    clone_repo(discordia_archivum_url, discordia_archivum_dir, "main")

    !pip install -q --upgrade -r {infinite_image_browser_deps}
    !pip install python-dotenv
    !pip install -q --upgrade -r {discordia_archivum_deps}

def install_dependencies():
    requirements_file = os.path.join(repo_dir, "requirements.txt")
    model_util        = os.path.join(repo_dir, "library/model_util.py")
    gpu_info          = getoutput('nvidia-smi')
    t4_xformers_wheel = "https://github.com/Linaqruf/colab-xformers/releases/download/0.0.20/xformers-0.0.20+1d635e1.d20230519-cp310-cp310-linux_x86_64.whl"

    !apt install -y aria2 lz4
    !wget https://github.com/DEX-1101/kohya-trainer/raw/main/requirements.txt -O /kaggle/working/requirements.txt
    !wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /kaggle/working/libtcmalloc_minimal.so.4
    !pip install -q --upgrade -r {requirements_file}

    !pip install -q xformers==0.0.22.post7

    from accelerate.utils import write_basic_config

    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)

def prepare_environment():
    #os.environ["LD_PRELOAD"] = "/kaggle/working/libtcmalloc_minimal.so.4"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"

def main():
    os.chdir(root_dir)
    clone_repo(repo_url, repo_dir, branch)
    os.chdir(repo_dir)
    setup_directories()
    install_repository()
    install_dependencies()
    prepare_environment()

main()

"""
Package the project (code + data/raw) and upload it to a remote GPU server over SSH/SFTP.

Run from the repository root:
    python deployment_tools/deploy_to_gpu.py

Connection details are read from environment variables (or a .env file in the
repository root, see .env.example):
    GPU_HOST, GPU_PORT, GPU_USER, GPU_PASSWORD, GPU_REMOTE_DIR (optional)
"""

import os
import sys
import subprocess


def install_deps():
    print("[INFO] Installing deployment dependency paramiko...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])


try:
    import paramiko
except ImportError:
    install_deps()
    import paramiko

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; plain environment variables also work


def require_env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"[ERROR] Environment variable {name} is not set. "
                 f"Copy .env.example to .env and fill it in.")
    return value


# =============== Configuration ===============
HOST = require_env("GPU_HOST")
PORT = int(os.environ.get("GPU_PORT", "22"))
USER = require_env("GPU_USER")
PASSWORD = require_env("GPU_PASSWORD")

REMOTE_DIR = os.environ.get("GPU_REMOTE_DIR", "/workspace")
LOCAL_ZIP = "deploy_package.zip"
REMOTE_ZIP = f"{REMOTE_DIR}/deploy_package.zip"

EXCLUDED_DIRS = ['.git', '__pycache__', 'venv', '.venv', 'models', 'plots', '.gemini', 'archive', 'non_cats']
EXCLUDED_FILES = {'.env'}


def create_zip():
    print("[1/5] Packaging local code and dataset (this can take a minute or two, depending on the number of images)...")

    import zipfile
    with zipfile.ZipFile(LOCAL_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Skip large directories and hidden directories that should not be uploaded
            if any(exclude in root for exclude in EXCLUDED_DIRS):
                continue
            for file in files:
                # Skip old packages, locally trained weights, caches and credentials
                if file in EXCLUDED_FILES or file.endswith(('.zip', '.pth', '.pyc')):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
    print(f"      -> Package created: {LOCAL_ZIP} (size: {os.path.getsize(LOCAL_ZIP) / (1024*1024):.2f} MB)")


def deploy():
    print(f"\n[2/5] Connecting to GPU server {HOST}:{PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname=HOST, port=PORT, username=USER, password=PASSWORD, timeout=60, banner_timeout=120, auth_timeout=60)
        print("      -> SSH connection established.")

        print("\n[3/5] Creating the working directory on the GPU server...")
        ssh.exec_command(f"mkdir -p {REMOTE_DIR}")

        print("\n[4/5] Uploading the project package over SFTP (a few hundred MB of images can take several minutes)...")
        sftp = ssh.open_sftp()
        sftp.put(LOCAL_ZIP, REMOTE_ZIP)
        sftp.close()
        print("      -> Upload complete.")

        print("\n[5/5] Unzipping on the GPU server and installing the required Python packages...")
        stdin, stdout, stderr = ssh.exec_command(
            f"cd {REMOTE_DIR} && apt-get update -y && apt-get install -y unzip && unzip -o deploy_package.zip && pip install pandas scikit-learn matplotlib seaborn torch torchvision opencv-python"
        )

        # Stream the unzip / install output live
        for line in iter(stdout.readline, ""):
            print(line, end="")

        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("      -> Environment setup complete.")
        else:
            print("      -> Environment setup finished with warnings; this does not necessarily affect training.")
            err_msg = stderr.read().decode()
            if err_msg:
                print("Error details:", err_msg)

        print("\n=======================================================")
        print("Deployment complete. Code and data have been uploaded and unzipped on the GPU server.")
        print("Start training from a terminal with:")
        print(f"    ssh -p {PORT} {USER}@{HOST}")
        print(f"    cd {REMOTE_DIR}")
        print("    python training_src/preprocess.py")
        print("    python training_src/train.py --model transfer --arch resnet50 --epochs 5")
        print("Or run deployment_tools/run_remote.py to start the full two-phase training in the background.")
        print("=======================================================")

    except Exception as e:
        print(f"[ERROR] Deployment failed: {e}")
    finally:
        ssh.close()
        # Remove the temporary local package
        if os.path.exists(LOCAL_ZIP):
            os.remove(LOCAL_ZIP)


if __name__ == "__main__":
    create_zip()
    deploy()

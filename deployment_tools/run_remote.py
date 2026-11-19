"""
Start the two-phase training run in the background on the remote GPU server.

Run deploy_to_gpu.py first: it uploads the code and data/raw to GPU_REMOTE_DIR.
Training output is written to /root/train.log on the server.

Connection details are read from environment variables (or a .env file in the
repository root, see .env.example):
    GPU_HOST, GPU_PORT, GPU_USER, GPU_PASSWORD, GPU_REMOTE_DIR (optional)
"""

import os
import sys

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


hostname = require_env("GPU_HOST")
port = int(os.environ.get("GPU_PORT", "22"))
username = require_env("GPU_USER")
password = require_env("GPU_PASSWORD")
remote_dir = os.environ.get("GPU_REMOTE_DIR", "/workspace")

# Phase 1: train the classification head with a frozen ResNet50 backbone.
# Phase 2: unfreeze the backbone and fine-tune from the phase-1 weights.
cmd = (
    "nohup bash -c '"
    "[ -f /etc/network_turbo ] && source /etc/network_turbo; "
    f"cd {remote_dir} && "
    "pip install pandas scikit-learn matplotlib seaborn torch torchvision opencv-python && "
    "python training_src/preprocess.py && "
    "echo \"Starting Phase 1\" && "
    "python training_src/train.py --model transfer --arch resnet50 --epochs 10 --no_aug && "
    "echo \"Starting Phase 2\" && "
    "python training_src/train.py --model transfer --arch resnet50 --epochs 30 --fine_tune --weights models/best_transfer.pth"
    "' > /root/train.log 2>&1 &"
)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, port=port, username=username, password=password, timeout=10)
stdin, stdout, stderr = ssh.exec_command(cmd)
exit_status = stdout.channel.recv_exit_status()
print(f"Exit code: {exit_status}")
ssh.close()

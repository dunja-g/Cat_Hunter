import os
import sys
import subprocess
import time

def install_deps():
    print("[INFO] 正在安装部署依赖 paramiko...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paramiko"])

try:
    import paramiko
except ImportError:
    install_deps()
    import paramiko

# =============== 配置区域 ===============
HOST = os.getenv("GPU_HOST", "cpod-1suhf5agph17.podtcp.compshare.cn")
PORT = int(os.getenv("GPU_PORT", 25292))
USER = os.getenv("GPU_USER", "root")
PASSWORD = os.getenv("GPU_PASSWORD", "YOUR_PASSWORD_HERE") # ⚠️ 请勿在代码中硬编码密码


REMOTE_DIR = "/root/cat_project"
LOCAL_ZIP = "deploy_package.zip"
REMOTE_ZIP = f"{REMOTE_DIR}/deploy_package.zip"

def create_zip():
    print("[1/5] 正在打包本地代码和数据集 (这可能需要一两分钟，取决于图片数量)...")
    
    import zipfile
    with zipfile.ZipFile(LOCAL_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # 排除不需要上传的大目录和隐藏目录
            if any(exclude in root for exclude in ['.git', '__pycache__', 'venv', '.venv', 'models', 'plots', '.gemini', 'archive', 'non_cats']):
                continue
            for file in files:
                # 排除历史打包文件、本地训练好的权重、环境缓存
                if file.endswith('.zip') or file.endswith('.pth') or file.endswith('.pyc'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
    print(f"      -> 打包完成: {LOCAL_ZIP} (文件大小: {os.path.getsize(LOCAL_ZIP) / (1024*1024):.2f} MB)")

def deploy():
    print(f"\n[2/5] 正在连接到 GPU 服务器 {HOST}:{PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname=HOST, port=PORT, username=USER, password=PASSWORD, timeout=60, banner_timeout=120, auth_timeout=60)
        print("      -> SSH 连接成功！")
        
        print("\n[3/5] 正在 GPU 上创建工作目录...")
        ssh.exec_command(f"mkdir -p {REMOTE_DIR}")
        
        print("\n[4/5] 正在通过 SFTP 上传项目压缩包 (上传几百 MB 的图片可能需要几分钟，请耐心等待)...")
        sftp = ssh.open_sftp()
        sftp.put(LOCAL_ZIP, REMOTE_ZIP)
        sftp.close()
        print("      -> 上传成功！")
        
        print("\n[5/5] 正在 GPU 上解压并自动安装必要的 Python 依赖...")
        stdin, stdout, stderr = ssh.exec_command(
            f"cd {REMOTE_DIR} && apt-get update -y && apt-get install -y unzip && unzip -o deploy_package.zip && pip install pandas scikit-learn matplotlib seaborn torch torchvision opencv-python"
        )
        
        # 实时打印解压和安装的输出
        for line in iter(stdout.readline, ""):
            print(line, end="")
            
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("      -> 环境配置完成！")
        else:
            print("      -> 环境配置出现一些警告，但这不一定影响训练。")
            err_msg = stderr.read().decode()
            if err_msg:
                print("Error details:", err_msg)
        
        print("\n=======================================================")
        print("🎉 部署成功！你的代码和数据已经全部安全送达 GPU 并在远端解压完毕！")
        print("接下来，你可以使用以下命令直接在终端里启动训练：")
        print(f"    ssh -p {PORT} {USER}@{HOST}")
        print(f"    密码: {PASSWORD}")
        print(f"    cd {REMOTE_DIR}")
        print("    python src/preprocess.py")
        print("    python src/train.py --model transfer --epochs 5")
        print("=======================================================")
        
    except Exception as e:
        print(f"[ERROR] 部署失败: {e}")
    finally:
        ssh.close()
        # 清理本地的临时压缩包
        if os.path.exists(LOCAL_ZIP):
            os.remove(LOCAL_ZIP)

if __name__ == "__main__":
    create_zip()
    deploy()

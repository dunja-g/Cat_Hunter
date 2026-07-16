import os
import subprocess
import shutil

REPO_URL = "https://github.com/dunja-g/Cat_Hunter.git"
REPO_DIR = "Cat_Hunter_Repo"
ZIP_FILE = "deploy_package.zip"

def upload_to_github():
    print("[1/4] 正在克隆共享仓库 dunja-g/Cat_Hunter ...")
    if os.path.exists(REPO_DIR):
        # 如果已经存在旧目录，先清理掉
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    
    print(f"\n[2/4] 正在将 {ZIP_FILE} 复制到仓库中...")
    shutil.copy(ZIP_FILE, os.path.join(REPO_DIR, ZIP_FILE))
    
    # 切换到仓库目录内
    os.chdir(REPO_DIR)
    
    print("\n[3/4] 正在配置 Git LFS (大文件存储)...")
    print("      (注意：因为我们的压缩包有 126MB，超过了 GitHub 100MB 的单文件死线，必须用 LFS 才能传上去)")
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "lfs", "track", "*.zip"], check=True)
        subprocess.run(["git", "add", ".gitattributes"], check=True)
        subprocess.run(["git", "add", ZIP_FILE], check=True)
    except Exception as e:
        print("\n[ERROR] Git LFS 配置失败！请确保你的电脑上安装了 Git LFS。")
        raise e
    
    print("\n[4/4] 正在提交并推送到共享仓库...")
    subprocess.run(["git", "commit", "-m", "Upload compressed 5 category dataset zip"], check=True)
    
    try:
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("\n=======================================")
        print("上传成功！压缩包已推送到 dunja-g/Cat_Hunter")
        print("=======================================")
    except Exception as e:
        print("\n[ERROR] 推送失败！最常见的原因是：你当前的 GitHub 账号没有 dunja-g/Cat_Hunter 这个仓库的“写入/推送权限”。")
        print("请联系 dunja-g 给你加权限，或者你 Fork 一个仓库后再提交。")

if __name__ == "__main__":
    if not os.path.exists(ZIP_FILE):
        print(f"[ERROR] 找不到 {ZIP_FILE}，请确保脚本在 bsaeline model 目录下运行。")
    else:
        upload_to_github()

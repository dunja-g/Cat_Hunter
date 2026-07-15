import os
import subprocess
import sys
import time

def install_and_download():
    print("==============================================")
    print("Pallas Cat Image Downloader (using Bing)")
    print("==============================================")
    
    # 1. 自动安装需要的库
    print("\n[INFO] 正在检查并安装 bing-image-downloader 库...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bing-image-downloader"])
    
    from bing_image_downloader import downloader
    
    # 2. 设置保存路径
    # 我们把它保存在 data/raw 下面，符合我们的项目结构
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 开始批量下载
    print("\n[INFO] 准备从必应 (Bing) 批量下载 300 张兔狲 (Pallas cat) 图片...")
    print("[INFO] 这可能会花费几分钟时间，请耐心等待。下载过程中如果有些链接失效报错是正常的，程序会自动跳过。")
    time.sleep(2)
    
    try:
        downloader.download(
            query="Pallas cat", 
            limit=300,  
            output_dir=output_dir, 
            adult_filter_off=True, 
            force_replace=False, 
            timeout=10,
            verbose=False
        )
    except Exception as e:
        print(f"\n[ERROR] 下载过程中出现错误: {e}")
        
    # 4. 重命名文件夹为 Pallas
    # downloader 会默认创建一个名字等于搜索词 ("Pallas cat") 的文件夹
    old_folder = os.path.join(output_dir, 'Pallas cat')
    new_folder = os.path.join(output_dir, 'Pallas')
    
    if os.path.exists(old_folder):
        if not os.path.exists(new_folder):
            os.rename(old_folder, new_folder)
            print(f"\n[SUCCESS] 所有图片已成功下载并重命名，保存在: {new_folder} 文件夹中！")
        else:
            print(f"\n[SUCCESS] 下载完成！图片保存在: {old_folder} (因为 {new_folder} 已存在)")

if __name__ == "__main__":
    install_and_download()

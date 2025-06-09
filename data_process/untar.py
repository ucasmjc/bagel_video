import tarfile
import os
import glob
import multiprocessing
import sys

def extract_tar(tar_path):
    """
    解压单个 tar 文件到同名目录
    """
    try:
        # 创建目标目录（基于 tar 文件名）
        #output_dir = os.path.splitext(tar_path)[0].replace("/mnt/weka/data_hw","/mnt/weka/data_hw/final")
        output_dir="/mnt/weka/data_hw/final/video_framesn"
        os.makedirs(output_dir, exist_ok=True)
        
        # 解压文件
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=output_dir)
        
        print(f"✅ 成功解压: {tar_path} → {output_dir}")
        return True
    except Exception as e:
        print(f"❌ 解压失败 {tar_path}: {str(e)}", file=sys.stderr)
        return False

if __name__ == '__main__':
    # 获取所有 tar 文件路径
    import json

    data=[
         "/mnt/weka/data_hw/video_frames",
        
    ]
    for tar_dir in data:
        tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))[::-1]
        
        if not tar_files:
            print(f"目录中没有找到 .tar 文件: {tar_dir}")
            sys.exit(1)
        
        print(f"找到 {len(tar_files)} 个 tar 文件，开始解压...")
        
        # 设置进程池（根据 CPU 核心数调整，I/O 密集型可适当增加）
        max_workers = 2
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.map(extract_tar, tar_files)
        
        # 统计结果
        success = sum(results)
        failed = len(results) - success
        print(f"\n解压完成: 成功 {success} 个, 失败 {failed} 个")
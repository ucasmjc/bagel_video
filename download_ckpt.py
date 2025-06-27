from huggingface_hub import snapshot_download
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
snapshot_download(
    repo_id="ucasmjc/unify_video",
    local_dir="./pretrained_ckpts/t2i",
    repo_type="model",
    token="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",
)
directory="./pretrained_ckpts/t2i"
parts=[os.path.join(directory, filename) for filename in os.listdir(directory)]
parts.sort(key=lambda x: x[0])
        
output_path = os.path.join(directory, parts[0].split(".part")[0])
from tqdm import tqdm
# 合并分片
total_size = sum(os.path.getsize(part_path) for part_path in parts)
with open(output_path, 'wb') as outfile:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'合并 ') as pbar:
        for part_path in parts:
            with open(part_path, 'rb') as infile:
                while chunk := infile.read(8192):
                    outfile.write(chunk)
                    pbar.update(len(chunk))
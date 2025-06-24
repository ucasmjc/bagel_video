from huggingface_hub import login, snapshot_download
import os

local_dir="pretrained_ckpts/"
os.makedirs(local_dir, exist_ok=True)
cache_dir = local_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=local_dir+"BAGEL-7B-MoT",
  repo_id="ByteDance-Seed/BAGEL-7B-MoT",
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)


snapshot_download(cache_dir=cache_dir,
  local_dir=local_dir+"BAGEL-7B-MoT",
  repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*VAE.pth"],
)

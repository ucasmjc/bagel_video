from huggingface_hub import login, snapshot_download
import os

local_dir="pretrained_ckpts/"
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id="nvidia/Cosmos-1.0-Tokenizer-CV8x8x8", local_dir=local_dir+"Cosmos-1.0-Tokenizer-CV8x8x8")

cache_dir = local_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=local_dir+"BAGEL-7B-MoT",
  repo_id="ByteDance-Seed/BAGEL-7B-MoT",
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)


import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
vae = AutoencoderKLWan.from_pretrained("/mnt/data/checkpoints/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/", subfolder="vae", torch_dtype=torch.float32)

image = load_image(
    "/mnt/data/mjc/Bagel-main/assets/teaser.webp"
)

aaa=torch.randn(1, 3, 1, 256,256)
import pdb
pdb.set_trace()
latent=vae.encode(aaa).latent_dist.sample() #1*16*1*32*32

bbb=vae.decode(latent)
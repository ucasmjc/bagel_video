git clone https://github.com/ucasmjc/bagel_video.git
#conda envs
cd bagel_video
conda env create -f ./env.yml
conda activate bagel_train
pip install flash_attn==2.5.8 --no-build-isolation

sudo apt-get install -y ffmpeg git-lfs

python download_models.py
cp tokenizer_config.json pretrained_ckpts/BAGEL-7B-MoT


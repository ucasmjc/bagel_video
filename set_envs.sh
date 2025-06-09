git clone https://github.com/ucasmjc/bagel_video.git
#conda envs
cd bagel_video
conda env create -f ./env.yml
conda activate bagel_train
pip install flash_attn==2.5.8 --no-build-isolation
#cosmos
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
cd Cosmos-Tokenizer
sudo apt-get install -y ffmpeg git-lfs
git lfs pull
pip3 install -e .

#下载预训练权重
cd ..
python download_models.py
cp tokenizer_config.json pretrained_ckpts/BAGEL-7B-MoT
#下载T2I adapt权重

import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# 图像预处理管道
def process_image(path):
    image = Image.open(path).convert('RGB')  # 确保RGB格式
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
    ])
    return preprocess(image)

# 创建符合模型要求的输入张量 (1,3,9,512,512)
image_tensor = process_image('/mnt/localdisk/hongwei/Bagel/test.png').unsqueeze(0)  # 添加批次维度 -> [1,3,512,512]
input_tensor = image_tensor.unsqueeze(2).repeat(1, 1, 9, 1, 1)  # 扩展时间维度 -> [1,3,9,512,512]
input_tensor = input_tensor.to('cuda').to(torch.bfloat16)

# 初始化模型
model_name = "Cosmos-1.0-Tokenizer-CV8x8x8"
encoder = CausalVideoTokenizer(checkpoint_enc=f'/mnt/localdisk/hongwei/Bagel/Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit')
decoder = CausalVideoTokenizer(checkpoint_dec=f'/mnt/localdisk/hongwei/Bagel/Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit')
print(encoder._device)
# 编码解码过程
with torch.no_grad():
    latent = encoder.encode(input_tensor)[0]
    print(latent.shape)
    import pdb
    pdb.set_trace()
    reconstructed_tensor = decoder.decode(latent)

# 后处理并保存重建图像
def tensor_to_image(tensor):
    return transforms.Compose([
        transforms.Lambda(lambda x: x[0].cpu().float()),  # 取批次首元素并转CPU
        transforms.Lambda(lambda x: x * 0.5 + 0.5),       # 反归一化到[0,1]
        transforms.Lambda(lambda x: x.clamp(0, 1)),        # 确保数值合法
        transforms.ToPILImage()
    ])(tensor)
print(image_tensor.shape,reconstructed_tensor.shape)
# 取时间维度第一帧对比
original_image = tensor_to_image(image_tensor)
reconstructed_image = tensor_to_image(reconstructed_tensor[:, :, 0, :, :])  # 取首帧

# 保存对比结果
def create_comparison(ori_path, rec_path, output_path="comparison.jpg"):
    # 打开两张图片
    img1 = Image.open(ori_path)
    img2 = Image.open(rec_path)
    
    # 创建新画布 (宽度相加，保持高度一致)
    assert img1.size == img2.size, "图片尺寸不一致"
    w, h = img1.size
    canvas = Image.new('RGB', (w*2, h))
    
    # 拼接图片
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w, 0))
    
    # 添加分隔线
    draw = ImageDraw.Draw(canvas)
    draw.line((w,0,w,h), fill="white", width=2)
    
    canvas.save(output_path)
    return output_path

# 保存原始和重建图片后生成对比图
original_image.save('original.jpg')
reconstructed_image.save('reconstructed.jpg')
create_comparison('original.jpg', 'reconstructed.jpg', 'comparison.jpg')
print("对比图已保存为 comparison.jpg")
from imgutils.detect import detect_faces
import json,os
from tqdm import tqdm
with open("/mnt/localdisk/hongwei/Bagel/image_anime.json") as f:
    data = json.load(f)
data=[item for item in data if item["length"]>90]
root_dir = "/mnt/weka/data_hw/final/anime/"
output=[]
for item in tqdm(data):
    item["bbox"]=[]
    img_path = os.path.join(root_dir, item["video_path"].split("/cpfs01/shared/llm_lol/sunyanan/datasets/AnimeFace/")[1])
    try:
        for idx in range(item["length"]):
            result = detect_faces(os.path.join(img_path,f"{idx}.jpg"))  
            item["bbox"].append(result[0][0])
        output.append(item)
    except Exception as e:
        print("error",e)
        continue
output_path = "/mnt/localdisk/hongwei/Bagel/generated_captions.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
        

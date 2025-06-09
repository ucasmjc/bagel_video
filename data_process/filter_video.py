import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from PIL import Image

def process_single_sub(items, folder,all_dirname):
    """处理单个数据项的线程函数"""
    res=[]
    for item in items:
        # tar_path=item["path"].split("/")[0]
        # item["path"] = os.path.join(folder,tar_path, item["path"])
        frame_number = item['cut'][0]
        crop_coords = item['crop']
        video_path = '/work/'+item ['path']
        filename = video_path.rsplit('/', 1)[-1].split('.mp4')[0]
        isexists=False
        for dirn in all_dirname:
            
            output_path= os.path.join(folder,dirn,filename+f'{frame_number}.jpg')
            if os.path.exists(output_path):
                isexists=True
                item["path"]=output_path
                break
        if isexists:
            res.append(item)
        else:
            continue
    return res

def process_single_file(anno, folder, file_name, output_path,pre_fix,all_dirname, sub_workers=4):
    data_path = os.path.join(anno, file_name)
    output_file = data_path.replace(".pkl", ".json").replace(pre_fix, output_path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        return 0,0
    print(data_path)
    # 读取数据
    try:
        if data_path.endswith(".json"):
            data_out = pd.read_json(data_path)
        elif data_path.endswith(".pkl"):
            data_out = pd.read_pickle(data_path)
    except:
        return 0,0
    if isinstance(data_out, pd.DataFrame):
        data_out=data_out.to_dict("records")

    #import pdb
    #pdb.set_trace()
    part_length=50
    length=len(data_out)//part_length
    
    # 创建子线程池处理单个文件内的数据
    with ThreadPoolExecutor(max_workers=sub_workers) as executor:
        futures = []
        print(folder)
        for sub_id in range(0,length):
            sub_part=data_out[sub_id*part_length:(sub_id+1)*part_length]
            futures.append(executor.submit(process_single_sub, sub_part, folder,all_dirname))
        last_part=data_out[length*part_length:]
        futures.append(executor.submit(process_single_sub, last_part, folder,all_dirname))
        
        # 使用tqdm显示单个文件处理进度
        filtered_data = []
        with tqdm(total=len(futures), desc=f"Processing {file_name}", leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    filtered_data.extend(result)
                pbar.update(1)
    print(len(filtered_data),len(data_out))

    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=None, ensure_ascii=False)
    print("saving to",output_file)
    return len(filtered_data),len(data_out)
    

def filter_data(output_path):
    """双层多线程处理函数"""

    with open("/mnt/weka/data_hw/final/dataset.json", 'r', encoding='utf-8') as f:
        data_out = json.load(f)
    data=[
    ["/mnt/weka/data_hw/img_json/video_Frames", "/mnt/weka/data_hw/final/video_frames"]
    ]
    start=0.5
    all_dirname=os.listdir("/mnt/weka/data_hw/final/video_frames")
    newjson=[]
    pre_fix="/mnt/weka/data_hw/"
    for item in data:
        new_path=item[0]
        for file in os.listdir(new_path):
            if file.endswith(".json"):
                newjson.append([os.path.join(new_path,file),item[1]])
    data=newjson
    data=data[int(len(data)*start):]
    tasks = []
    fnum=0
    onum=0
    for line in data:
        anno, folder = line
        files = [anno]
        for f in files:
            filtered,origin=process_single_file(anno, folder, f, output_path, pre_fix,all_dirname,sub_workers=128)
            fnum=fnum+filtered
            onum=onum+origin
    print(fnum,onum)

if __name__ == "__main__":
    output_json = "/mnt/weka/data_hw/final/"
    filter_data(output_json)
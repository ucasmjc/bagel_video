import json
import lmdb
import os
from tqdm import tqdm  # 进度条（可选）
import multiprocessing
from functools import partial
import pandas as pd
def json_to_lmdb(json_path):
    """Convert JSON file to LMDB with unique keys"""
    # Generate LMDB path from JSON path
    base_name = os.path.splitext(json_path)[0].replace("/work/share/projects/mjc/data","/work/share/projects/mjc/data/lmdb_data")
    lmdb_path = f"{base_name}.lmdb"
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    # Generate key prefix from directory structure
    json_dir = os.path.dirname(json_path)
    parent_dir = os.path.basename(json_dir)
    json_filename = os.path.basename(json_path)
    filename_part = os.path.splitext(json_filename)[0]
    key_prefix = f"{parent_dir}/{filename_part}"

    # Create LMDB environment
    env = lmdb.open(lmdb_path, map_size=1099511627776)  # 1TB
    
    # Read JSON data
    # data_list = pd.read_pickle(json_path)
    # if isinstance(data_list, pd.DataFrame):
    #     data_list=data_list.to_dict("records")
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    
    # Write to LMDB with unique keys
    with env.begin(write=True) as txn:
        for idx, item in enumerate(tqdm(data_list, desc=f"Processing {json_path}")):
            try:
                unique_key = f"{key_prefix}_{idx}".encode()
                value = json.dumps(item).encode()
                txn.put(unique_key, value)
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue
    
    print(f"Saved {len(data_list)} items to {lmdb_path}")
    return len(data_list)

def process_single_json(json_path):
    """Wrapper function for parallel processing"""
    #base_name = os.path.splitext(json_path)[0].replace("/work/share/projects/mjc/data","/work/share/projects/mjc/data/lmdb_data")
    base_name = os.path.splitext(json_path)[0].replace("/work/share1/yqs/uni_dataset/","/work/share/projects/mjc/data/lmdb_data")
    lmdb_path = f"{base_name}.lmdb"
    
    try:
        length=json_to_lmdb(json_path)
        return (json_path, True, "Converted successfully",length)
    except Exception as e:
        print(f"Failed to convert {json_path}: {e}")
        return (json_path, False, str(e))

# 处理所有 JSON 文件

def get_all_keys_optimized(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, map_size=1099511627776)
    keys = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in tqdm(cursor, total=txn.stat()["entries"]):
            keys.append(key.decode())
    return keys

def valid_lmdb_files(lmdb_path, json_path):
    try:
        env = lmdb.open(lmdb_path, readonly=True, map_size=1099511627776)
        with env.begin() as txn:
            stat = env.stat()
            expected_nums=stat['entries']
        if expected_nums==0:
            return False
        return True
    except Exception as e:
        print(f"LMDB Error: {e}")
        return False

    
    
def test_lmdb_integrity( lmdb_path, key_json_path, check_image_exist=False):
    # 加载keys
    try:
        with open(key_json_path, 'r') as f:
            keys = json.load(f)
    except Exception as e:
        print(f"Error loading keys: {e}")
        return False


    # 打开LMDB环境
    env = lmdb.open(lmdb_path, readonly=True, map_size=1099511627776, max_readers=256, lock=False)  # 只读模式无需锁
    
    success = 0
    failed_keys = []
    corrupted_items = []

    with env.begin() as txn:
        # 进度条显示检查进度
        pbar = tqdm(total=len(keys), desc="Validating LMDB")
        
        for key_str in keys:
            try:
                # 转换为bytes类型键
                key = key_str.encode()
                
                # 读取数据
                value = txn.get(key)
                if not value:
                    failed_keys.append( (key_str, "KEY_NOT_FOUND") )
                    pbar.update(1)
                    continue
                
                # 解析JSON数据
                try:
                    item = json.loads(value.decode())
                except Exception as e:
                    corrupted_items.append( (key_str, f"JSON_PARSE_ERROR: {str(e)}") )
                    pbar.update(1)
                    continue
                success += 1
                pbar.update(1)
                
            except Exception as e:
                failed_keys.append( (key_str, f"UNEXPECTED_ERROR: {str(e)}") )
                pbar.update(1)
        
        pbar.close()

    # 打印统计信息
    print(f"\nValidation Report for {lmdb_path}:")
    print(f"Total keys: {len(keys)}")
    print(f"Successfully read: {success} ({success/len(keys)*100:.2f}%)")
    print(f"Failed keys: {len(failed_keys)}")
    print(f"Corrupted items: {len(corrupted_items)}")
    
    # 输出详细错误日志
    if failed_keys:
        print("\nTop 10 Failed Keys:")
        for key, reason in failed_keys[:10]:
            print(f"{key}: {reason}")
    
    if corrupted_items:
        print("\nTop 10 Data Issues:")
        for key, reason in corrupted_items[:10]:
            print(f"{key}: {reason}")
    
    return len(failed_keys) == 0 and len(corrupted_items) == 0

def test_worker(args, check_image_exist=False):
    """
    多进程工作函数
    """
    json_path, img_data_dir = args
    base_name = os.path.splitext(json_path)[0]
    lmdb_path = f"{base_name}.lmdb"
    key_json_path = f"{base_name}_key.json"
    result = test_lmdb_integrity(
        lmdb_path=lmdb_path,
        key_json_path=key_json_path,
        check_image_exist=check_image_exist
    )

    return (lmdb_path, result)

def test_lmdb_func(lmdb_path, key_json_path, check_image_exist=False):
    # 加载keys
    try:
        with open(key_json_path, 'r') as f:
            keys = json.load(f)
    except Exception as e:
        print(f"Error loading keys: {e}")
        return False
    key_length=len(keys)
    # 打开LMDB环境
    env = lmdb.open(lmdb_path, readonly=True, map_size=1099511627776, max_readers=256, lock=False)  # 只读模式无需锁
    with env.begin() as txn:
        stat = env.stat()
        expected_nums=stat['entries']
        if key_length!=expected_nums:
            print("error!",lmdb_path)
            return False
        try:
            key = keys[0].encode()
            value = txn.get(key).decode()
            data_item = json.loads(value)
            return True
        except Exception as e:
            print(len(keys),expected_nums)
            print(f"Error loading lmdb: {e}")
            return False
image_gen_config = [
   ["/mnt/weka/data_hw/final/img_json/recap2/","/mnt/weka/data_hw/final/recap2" ],
]

# ======= 执行逻辑 =======
convert_json_to_lmdb = True
extract_lmdb_keys = True
valid_lmdb = False
test_lmdb = False
check_image_files = False  # 是否验证图片文件实际存在
max_workers = 8 

# aaa=[]
# for i in range(40):
#     aaa.append([f"/work/share/projects/mjc/lmfusion/data/long_video_180/long_video_180_{i}.json","/work/"])
# data["video_gen"]=aaa
# print(data["video_gen"])
# with open(f"/work/share/projects/mjc/lmfusion/train_files/merge_datalmdb.json", 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False)
# sys.exit()

if convert_json_to_lmdb:
    print("\n===== Starting parallel conversion =====")
    json_paths = [os.path.join(entry[0],f) for entry in image_gen_config for f in os.listdir(entry[0]) if ".json" in f and "key.json" not in f]

    pool = multiprocessing.Pool(processes=max_workers)
    results = []
    
    # 使用imap_unordered获取实时进度
    for result in tqdm(pool.imap_unordered(process_single_json, json_paths), 
                      total=len(json_paths), desc="Overall Progress"):
        results.append(result)
    
    pool.close()
    pool.join()
    
    # 打印汇总结果
    success_count = sum(1 for r in results if r[1])
    all_data = sum(r[3] for r in results)
    print(f"\nConversion completed: {success_count}/{len(results)} successful")
    print(all_data)

if extract_lmdb_keys:
    for json_path in json_paths:
        base_name = os.path.splitext(json_path)[0]
        lmdb_path = f"{base_name}.lmdb"
        key_json = f"{base_name}_key.json"
        
        if os.path.exists(key_json):
            with open(key_json, 'r') as f:
                keys = json.load(f)
            if len(keys)>0:
                print(f"{key_json} exists")
                continue
        
        if not os.path.exists(lmdb_path):
            print(f"{lmdb_path} missing")
            continue
        
        try:
            keys = get_all_keys_optimized(lmdb_path)
            with open(key_json, 'w') as f:
                json.dump(keys, f, indent=2)
            print(f"Keys saved to {key_json}")
        except Exception as e:
            print(f"Error extracting keys: {e}")
        

# 导出所有 key 到文件（二进制格式）
# mdb_dump -k -n /mnt/new_cpfs/captions/img_cap/cap_final/cap_merge_final_640/recap2/uni_part0_cap6595998.lmdb  > /mnt/new_cpfs/captions/img_cap/cap_final/cap_merge_final_640/recap2/uni_part0_cap6595998_keys.dump
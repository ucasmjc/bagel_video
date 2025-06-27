---
language:
- en
pretty_name: BLIP3o-60k
size_categories:
- 10K<n<100K
license: apache-2.0
---
This is BLIP3o-60k Text-to-Image instruction tuning dataset distilled from GPT-4o, including the following categories:

1. JourneyDB
2. Human (including MSCOCO with human caption, human gestures, occupations)
3. Dalle3
4. Geneval (no overlap with test set)
5. Common objects
6. Simple text


Here we provide the code guidance to download tar file:
```
from huggingface_hub import snapshot_download
snapshot_download(repo_id='BLIP3o/BLIP3o-60k', repo_type=‘dataset’)
```

And you can use huggingface datasets to read the tar file without unzipping them:

```
from datasets import load_dataset
import glob
data_files = glob.glob('/your/datasets/path/*.tar') 
train_dataset = load_dataset("webdataset", data_files=data_files, cache_dir='/your/cache/directory/', split="train", num_proc=64)
```
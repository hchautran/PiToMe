import os
import subprocess
import os
import lavis
from lavis.common.utils import (
    cleanup_dir,
    get_abs_path,
    get_cache_path
) 
from omegaconf import OmegaConf
from pathlib import Path
lavis_path = '/'.join(lavis.__file__.split('/')[:-1])
config_path = get_abs_path(f"{lavis_path}/configs/datasets/coco/defaults_ret.yaml")



storage_dir = OmegaConf.load(
    config_path
).datasets.coco_retrieval.build_info.images.storage
COCO_PATH = Path(get_cache_path(storage_dir))
print('downloading coco dataset to:', COCO_PATH)

os.makedirs(COCO_PATH, exist_ok=True)

os.chdir(COCO_PATH)

print(COCO_PATH)

download_urls = [
    "http://images.cocodataset.org/zips/train2014.zip",
    "http://images.cocodataset.org/zips/test2014.zip",
    "http://images.cocodataset.org/zips/val2014.zip",
]

for url in download_urls:
    file_name = url.split("/")[-1]
    zip_file = os.path.join(COCO_PATH, file_name)

    subprocess.run(["wget", url])

    subprocess.run(["unzip", zip_file])

    os.remove(zip_file)

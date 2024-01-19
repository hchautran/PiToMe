import os
import subprocess
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
DATA_PATH = os.environ.get('DATA_PATH')

COCO_PATH = os.path.join(DATA_PATH, "coco/images")

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

import os
import subprocess
import os

# Load environment variables from .env file

# Access the environment variable
DATA_PATH = '' 


os.makedirs(DATA_PATH, exist_ok=True)

os.chdir(DATA_PATH)
print('Saving data to', DATA_PATH)

cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
cifar_tar = "cifar-10-python.tar.gz"
subprocess.run(["wget", cifar_url])
subprocess.run(["tar", "-xvf", cifar_tar])
os.remove(cifar_tar)

lra_url = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
lra_tar = "lra_release.gz"
subprocess.run(["wget", lra_url])
subprocess.run(["tar", "-xvf", lra_tar, "lra_release/listops-1000"])
subprocess.run(["tar", "-xvf", lra_tar, "lra_release/lra_release/tsv_data"])
os.remove(lra_tar)

imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
imdb_tar = "aclImdb_v1.tar.gz"
subprocess.run(["wget", imdb_url])
subprocess.run(["tar", "-xvf", imdb_tar])
os.remove(imdb_tar)

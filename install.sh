conda create -n pitome python=3.10
conda activate pitome
pip install -r requirements.txt
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
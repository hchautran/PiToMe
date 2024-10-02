# Accelerate Transformer With Spectrum Preserving Token Merging [(full PDF)](https://arxiv.org/abs/2405.16148)  
![Example Image](/figures/overview.png)
---
### News
- [01/10/2024] Release code for text classification task
- [29/09/2024] Release code for image-text retrieval task
- [25/09/2024] Our paper has been accepted at NeurIPS 2024 as a Poster ([OpenReview](https://openreview.net/forum?id=PPdJPIO3mV&noteId=NUW4EoVirr))
- [29/05/2024] Upload PrePrint on Arxiv

--- 
## Abstract

Increasing the throughput of the Transformer architecture, a foundational component used in numerous state-of-the-art models for vision and language tasks (e.g., GPT, LLaVa), is an important problem in machine learning. One recent and effective strategy is to merge token representations within Transformer models, aiming to reduce computational and memory requirements while maintaining accuracy. Prior works have proposed algorithms based on Bipartite Soft Matching (BSM), which divides tokens into distinct sets and merges the top k similar tokens. However, these methods have significant drawbacks, such as sensitivity to token-splitting strategies and damage to informative tokens in later layers. This paper presents a novel paradigm called PiToMe, which prioritizes the preservation of informative tokens using an additional metric termed the energy score. This score identifies large clusters of similar tokens as high-energy, indicating potential candidates for merging, while smaller (unique and isolated) clusters are considered as low-energy and preserved. Experimental findings demonstrate that PiToMe saved from 40-60\% FLOPs of the base models while exhibiting superior off-the-shelf performance on image classification (0.5\% average performance drop of ViT-MAE-H compared to 2.6\% as baselines), image-text retrieval (0.3\% average performance drop of CLIP on Flickr30k compared to 4.5\% as others), and analogously in visual questions answering with LLaVa-7B. Furthermore, PiToMe is theoretically shown to preserve intrinsic spectral properties of the original token space under mild conditions.

---
## Method
![Example Image](/figures/method.png)

All implementations of PiToMe and baselines can be found in the [algo](algo) folder

---
## Experiments 
### Installation 
First, you need to install the required packaged using the command below:  
```
conda create -n pitome python=3.10
conda activate pitome
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -r requirements.txt
```

### Image-Text Retrieval 

#### Data Preparation

In our paper we evaluate our method on 2 dataset - Flickr30k and MS-COCO. 

**Step 1**: Configure the data storage path to your wanted path in the `default.yml` file. This file is located in the the folder where lavis is installed. you can find it quickly by using this command:
```
print(f"{'/'.join(lavis.__file__.split('/')[:-1])}/configs")

```

Update the `cache_root`  entry to the path that you wanted.


**Step 2**: Download the data
You can download Flickr30k and MSCOCO by using avaiable scripts:
```
python itr/download_coco.py
python itr/download_flickr.py
```


#### Run 

Currently, we are supporting `blip`, `blip2`, `clip`, and `albef` you can try directly compressing these models for off-the-shell performance or retrain them by omitting the `--eval` argument.

```
python -m torch.distributed.run \
    --nproc_per_node=5 main_vl.py \
    --cfg-path scripts/eval_scripts/blip_itr_coco.yml \
    --algo pitome \
    --use_k False \
    --ratio 0.95 \
    --model blip \
    --dataset flickr \
    --eval 
```

You can also evaluate all other baselines with multiple ratio `r` by running:

```
python scripts/eval_itr.sh
```

The results will be printed and saved to the `itr_outputs` directory.

### Image Classification 
Comming soon


### Text Classification 

First, you need to download the data
you can download Flickr30k and MSCOCO:
```
python tc/get_data.py
```

#### Run

Currently, we are supporting `bert`,  and `distilbert` you can try directly compressing these models for off-the-shell performance or retrain them by omitting the `--eval` argument.
```
python main_tc.py \
   --algo pitome \
   --ratio  0.65 \
   --task imdb \
   --model bert-base-uncased \
   --eval 
```

You can also evaluate all other baselines with multiple ratio `r` by running:
```
python scripts/eval_tc.sh
```
The results will be printed and saved to `tc_outputs` directory.

### Visual Question Answering
Coming soon

---
## Citation

```
@misc{https://doi.org/10.48550/arxiv.2405.16148,
  doi = {10.48550/ARXIV.2405.16148},
  url = {https://arxiv.org/abs/2405.16148},
  author = {Tran,  Hoai-Chau and Nguyen,  Duy M. H. and Nguyen,  Duy M. and Nguyen,  Trung-Tin and Le,  Ngan and Xie,  Pengtao and Sonntag,  Daniel and Zou,  James Y. and Nguyen,  Binh T. and Niepert,  Mathias},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Accelerating Transformers with Spectrum-Preserving Token Merging},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
If you have any issues, feel free to contact me at tranhoaichau.00@gmail.com

# Accelerate Transformer With Spectrum Preserving Token Merging [(full PDF)](https://arxiv.org/abs/2405.16148)  
![Example Image](/figures/overview.png)
---
### News
[29/09/2024] Release code for image-text retrieval task
[25/09/2024] Our paper has been accepted at NeurIPS 2024 
[29/05/2024] Upload PrePrint on Arxiv

--- 
### Abstract

Increasing the throughput of the Transformer architecture, a foundational component used in numerous state-of-the-art models for vision and language tasks (e.g., GPT, LLaVa), is an important problem in machine learning. One recent and effective strategy is to merge token representations within Transformer models, aiming to reduce computational and memory requirements while maintaining accuracy. Prior works have proposed algorithms based on Bipartite Soft Matching (BSM), which divides tokens into distinct sets and merges the top k similar tokens. However, these methods have significant drawbacks, such as sensitivity to token-splitting strategies and damage to informative tokens in later layers. This paper presents a novel paradigm called PiToMe, which prioritizes the preservation of informative tokens using an additional metric termed the energy score. This score identifies large clusters of similar tokens as high-energy, indicating potential candidates for merging, while smaller (unique and isolated) clusters are considered as low-energy and preserved. Experimental findings demonstrate that PiToMe saved from 40-60\% FLOPs of the base models while exhibiting superior off-the-shelf performance on image classification (0.5\% average performance drop of ViT-MAE-H compared to 2.6\% as baselines), image-text retrieval (0.3\% average performance drop of CLIP on Flickr30k compared to 4.5\% as others), and analogously in visual questions answering with LLaVa-7B. Furthermore, PiToMe is theoretically shown to preserve intrinsic spectral properties of the original token space under mild conditions.

---
### Method
![Example Image](/figures/method.png)


[Link Text](relative/path/to/file)

---
### Experiments 
##### Installation 

```
conda create -n pitome python=3.10
conda activate pitome
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -r requirements.txt
```
##### Image-Text Retrieval 

##### Image Classification 
##### Text Classification 
##### Visual Question Answering

---
### Citation

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
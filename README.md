<br />
<p align="center">

  <h1 align="center">Accelerating Transformers with Spectrum-Preserving Token Merging</h1>
  <h3 align="center" style="font-size: 32;">NeurIPS, 2024</h3> 

  <p align="center">
    <a href="https://scholar.google.com/citations?user=FZH2vcEAAAAJ&hl=en"><strong>Hoai-Chau Tran*</strong></a>
    ·
    <a href="https://duyhominhnguyen.github.io/"><strong>Duy M. H. Nguyen*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=wmuJBfcAAAAJ&hl=en"><strong>Duy M. Nguyen</strong></a>
    ·
    <a href="https://trung-tinnguyen.github.io/"><strong>Trung-Tin Nguyen</strong></a>
    ·
    <a href="https://uark-aicv.github.io/"><strong>Ngan Le</strong></a>
    ·
    <a href="https://pengtaoxie.github.io/"><strong>Pengtao Xie</strong></a>
    ·
    <a href="https://www.dfki.de/~sonntag/"><strong>Daniel Sonntag</strong></a>
    ·
    <a href="https://www.james-zou.com/"><strong>James Y. Zou</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=dXEb3PMAAAAJ&hl=en"><strong>Binh T. Nguyen</strong></a>
    ·
    <a href="https://www.matlog.net/"><strong>Mathias Niepert</strong></a>
 
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2405.16148'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'></a>
    <a href='https://arxiv.org/abs/2405.16148'><img src='https://img.shields.io/badge/arXiv-2409.18964-b31b1b.svg'  alt='Arxiv'></a>
    <a href='' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
    <a href='' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'></a>
  </p>

</p>
<br />

This repository provides a PyTorch implementation of the paper [Accelerating Transformers with Spectrum-Preserving Token Merging](https://arxiv.org/abs/2405.16148), accepted at NeurIPS 2024. In this work, we introduce a new algorithm called pitome, designed to compress Vision Transformers (ViT) across various applications through token merging. After each layer, tokens are progressively merged, resulting in a remaining r percentage of tokens, as illustrated in the figure below.

![Example Image](/figures/overview.png)

News 
---
- Code for VQA with LLaVA 1.5 is under refractoring. Comming soon. 
- [27/10/2024] Release code for image classification task
- [01/10/2024] Release code for text classification task
- [29/09/2024] Release code for image-text retrieval task
- [25/09/2024] Our paper has been accepted at NeurIPS 2024 as a Poster ([OpenReview](https://openreview.net/forum?id=PPdJPIO3mV&noteId=NUW4EoVirr))
- [29/05/2024] Upload PrePrint on Arxiv

Abstract
--- 


Increasing the throughput of the Transformer architecture, a foundational component used in numerous state-of-the-art models for vision and language tasks (e.g., GPT, LLaVa), is an important problem in machine learning. One recent and effective strategy is to merge token representations within Transformer models, aiming to reduce computational and memory requirements while maintaining accuracy. Prior works have proposed algorithms based on Bipartite Soft Matching (BSM), which divides tokens into distinct sets and merges the top k similar tokens. However, these methods have significant drawbacks, such as sensitivity to token-splitting strategies and damage to informative tokens in later layers. This paper presents a novel paradigm called `PiToMe`, which prioritizes the preservation of informative tokens using an additional metric termed the energy score. This score identifies large clusters of similar tokens as high-energy, indicating potential candidates for merging, while smaller (unique and isolated) clusters are considered as low-energy and preserved. Experimental findings demonstrate that PiToMe saved from 40-60\% FLOPs of the base models while exhibiting superior off-the-shelf performance on image classification (0.5\% average performance drop of ViT-MAE-H compared to 2.6\% as baselines), image-text retrieval (0.3\% average performance drop of CLIP on Flickr30k compared to 4.5\% as others), and analogously in visual questions answering with LLaVa-7B. Furthermore, PiToMe is theoretically shown to preserve intrinsic spectral properties of the original token space under mild conditions.

Method
---
![Example Image](/figures/method.png)

All implementations of PiToMe and baselines can be found in the [algo](algo) folder

Installation 
---

First, you need to clone this repository
```
git clone https://github.com/hchautran/PiToMe.git
cd PiToMe
```
Next, you need to install the required packages using the commands below:  
```
conda create -n pitome python=3.10
conda activate pitome
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -r requirements.txt
```


Image-Text Retrieval 
---

### Data Preparation

In our paper we evaluate our method on 2 dataset - [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and [MS-COCO](https://cocodataset.org/). 

**Step 1**: Configure the data storage path in the `default.yml` file and change this to your preferred path. This file is located in the the folder where lavis is installed. you can find it quickly by using this command:
```
import lavis;print(f"{'/'.join(lavis.__file__.split('/')[:-1])}/configs");
```

Update the `cache_root`  entry to the path that you wanted.


**Step 2**: Download the data
You can download Flickr30k and MSCOCO by using avaiable scripts:
```
python itr/download_coco.py
python itr/download_flickr.py
```


### Run 

Currently, we are supporting `blip`, `blip2`, `clip`, and `albef` you can try directly compressing these models for off-the-shell performance or retrain them by omitting the `--eval` argument. 


```
python -m torch.distributed.run \
    --nproc_per_node=5 main_itr.py \
    --cfg-path scripts/eval_scripts/blip_itr_coco.yml \
    --algo pitome \
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
### Using `pitome` with ITR models
Currently, only checkpoints from [LAVIS](https://github.com/salesforce/LAVIS) are supported. You can directly download and directly apply `pitome` to pretrained weights

```py
from lavis.models import load_model_and_preprocess
from algo import pitome

# Load a pretrained model, can be blip/albef/blip2 .
model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "coco", is_eval=False)
# Patch the blip's visual encoder with PiToMe.
pitome.patch.blip(model.visual_encoder)
# Set the number of ratio of remaining token per layer. See paper for details.
model.visual_encoder.ratio = 0.9 
```
In the future, we are planning support checkpoints from HuggingFace.

Image Classification 
---
We are currently supporting the `DeiT` and `MAE` models for image classification tasks. You can try directly compressing these models for off-the-shell performance or retraining them by omitting the `--eval` argument.

In this task all experiment are conducted on [ImageNet1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset, which is a subset of ImageNet that contain 1000 classes. By default, all data and model checkpoints will be downloaded and saved into the folder specified by `DATA_PATH` variable located in `tasks/ic/utils.py`. You can change this to the path you wanted.

``` sh
python main_ic.py \
   --batch-size 256 \ 
   --model ${ARCH}-${MODEL_SIZE}-${INPUT_SIZE}  \ 
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --eval
```

You can also evaluate all models with all baselines using multiple ratio `r` by running:
``` sh
python scripts/eval_scripts/eval_ic_all.sh
```
The results will be printed and saved to `outputs/ic_outputs` directory.

### Using `pitome` with ViT models for image classification
```py
from timm.models import create_model
from algo import pitome

# Load a pretrained model, can be any vit / deit model.
model = create_model("deit_base_patch16_224", pretrained=True)
# Patch the ViT model with ToMe.
pitome.patch.deit(model)
# pitome.patch.mae(model)
# Set the ratio of remain token  per layer. See paper for details.
model.ratio = 0.95 
```

Text Classification 
---
We support `bert` and `distilbert` for text classification tasks. You can try directly compressing these models for off-the-shell performance or retrain them by omitting the `--eval` argument.


In this task, all experiments are conducted on the following datasets:  [IMBb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [sst2](stanfordnlp/sst2) and [Rotten Tomatoes](cornell-movie-review-data/rotten_tomatoes) 

By default, all data and model checkpoints are downloaded and saved to the folder specified by the `DATA_PATH` variable in `tasks/tc/config.py`. You can modify this variable to specify a different path as needed.
```sh
python main_tc.py \
   --algo $ALGO \
   --ratio $RATIO \
   --task $TASK \
   --model $MODEL \
   --eval 
```

You can also evaluate all models with all baselines using multiple ratio `r` by running:
```sh
python scripts/eval_scripts/eval_tc_all.sh
```
The results will be printed and saved to `outputs/tc_outputs` directory.

### Using `pitome` with text classification models

```py
from algo import pitome
from transformers import AutoModelForSequenceClassification

# Load a pretrained model, can be bert or distilbert .
model_ckt = 'JiaqiLee/imdb-finetuned-bert-base-uncased'
# model_ckt = 'bert-base-uncased'
# model_ckt = 'distilbert-base-uncased'
model =  AutoModelForSequenceClassification.from_pretrained(model_ckt)

# Patch the bert encoder with PiToMe.
pitome.patch.bert(model.bert.encoder)
# pitome.patch.distilbert(model.distilbert.transformer)

# Set the number of ratio of remaining token per layer. See paper for details.
model.bert.encoder.ratio = 0.65 
# model.distilbert.transformer.ratio = self.ratio 
```

Visual Question Answering
---
Coming soon

Visualization
---


Comming soon

Citation
---


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


Acknowledgement
---

Thanks [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) for providing open-source code. This repository is built based on the original ToMe structure.
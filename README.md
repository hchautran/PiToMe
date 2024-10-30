<br />
<p align="center">

  <h1 align="center">Accelerating Transformers with Spectrum-Preserving Token Merging</h1>
  <h3 align="center" style="font-size: 40;">NeurIPS, 2024</h3> 

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
- [ ] Code for VQA with LLaVA 1.5 is under refractoring. Comming soon. 
- [x] **[27/10/2024]** Release code for image classification task
- [x] **[01/10/2024]** Release code for text classification task
- [x] **[29/09/2024]** Release code for image-text retrieval task
- [x] **[25/09/2024]** Our paper has been accepted at NeurIPS 2024 as a Poster 🎉 🎉 🎉
- [x] **[29/05/2024]** Upload PrePrint on Arxiv

Abstract
--- 


Increasing the throughput of the Transformer architecture, a foundational component used in numerous state-of-the-art models for vision and language tasks (e.g., GPT, LLaVa), is an important problem in machine learning. One recent and effective strategy is to merge token representations within Transformer models, aiming to reduce computational and memory requirements while maintaining accuracy. Prior works have proposed algorithms based on Bipartite Soft Matching (BSM), which divides tokens into distinct sets and merges the top k similar tokens. However, these methods have significant drawbacks, such as sensitivity to token-splitting strategies and damage to informative tokens in later layers. This paper presents a novel paradigm called `PiToMe`, which prioritizes the preservation of informative tokens using an additional metric termed the energy score. This score identifies large clusters of similar tokens as high-energy, indicating potential candidates for merging, while smaller (unique and isolated) clusters are considered as low-energy and preserved. Experimental findings demonstrate that PiToMe saved from 40-60\% FLOPs of the base models while exhibiting superior off-the-shelf performance on image classification (0.5\% average performance drop of ViT-MAE-H compared to 2.6\% as baselines), image-text retrieval (0.3\% average performance drop of CLIP on Flickr30k compared to 4.5\% as others), and analogously in visual questions answering with LLaVa-7B. Furthermore, PiToMe is theoretically shown to preserve intrinsic spectral properties of the original token space under mild conditions.

## Table of Contents

- [News](#news)
- [Abstract](#abstract)
- [Table of Contents](#table-of-contents)
- [Method](#method)
- [Installation](#installation)
- [Image-Text Retrieval](#image-text-retrieval)
  - [Using `pitome` with ITR models](#using-pitome-with-itr-models)
  - [Run](#run)
- [Image Classification](#image-classification)
  - [Using `pitome` with ViT models for image classification](#using-pitome-with-vit-models-for-image-classification)
  - [Run](#run-1)
- [Text Classification](#text-classification)
  - [Using `pitome` with text classification models](#using-pitome-with-text-classification-models)
  - [Run](#run-2)
- [Notebook](#notebook)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

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


### Run 

In our paper we evaluate our method on 2 dataset - [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and [MS-COCO](https://cocodataset.org/). 

**Step 1**: Configure the data storage path in the `default.yml` file and change this to your preferred path. This file is located in the the folder where lavis is installed. you can find it quickly by using this command:
```
import lavis;print(f"{'/'.join(lavis.__file__.split('/')[:-1])}/configs");
```

Update the `cache_root`  entry to the path that you wanted.


**Step 2**: Download the data
You can download Flickr30k and MSCOCO by using avaiable scripts:
```sh
python itr/download_coco.py
python itr/download_flickr.py
```



Currently, we are supporting `blip`, `blip2`, `clip`, and `albef` you can try directly compressing these models for off-the-shell performance by running this command:

```sh
python -m torch.distributed.run \
    --nproc_per_node=5 main_itr.py \
    --cfg-path scripts/eval_scripts/blip_itr_coco.yml \
    --algo pitome \
    --ratio 0.95 \
    --model blip \
    --dataset flickr \
    --eval 
```
Or retrain these model using this command:

```sh
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --main_process_port 29500 main_ic.py \
   --batch-size $BATCH_SIZE \
   --model ${ARCH}_${SIZE}_patch16_${INPUT_SIZE}  \
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --input-size ${INPUT_SIZE} \
   --epoch $EPOCH  \
   --lr 0.00001

```

You can also evaluate/train all other baselines with multiple ratio `r` by running:

```sh
python scripts/eval_scripts/eval_itr_all.sh #off-the-shell evaluate 
python scripts/train_scripts/train_itr_all.sh #retrain
```

The results will be printed and saved to the `itr_outputs` directory. 


Image Classification 
---
### Using `pitome` with ViT models for image classification
We are currently supporting the `DeiT` and `MAE` models for image classification tasks.
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

### Run
In this task all experiment are conducted on [ImageNet1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset, which is a subset of ImageNet that contain 1000 classes. By default, all data and model checkpoints will be downloaded and saved into the folder specified by `DATA_PATH` variable located in `tasks/ic/utils.py`. You can change this to the path you wanted.

You can try directly compressing these models for off-the-shell performance
``` sh
python main_ic.py \
   --batch-size 256 \ 
   --model ${ARCH}-${MODEL_SIZE}-${INPUT_SIZE}  \ 
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --eval
```
Or retraining them by running this command: 
``` sh
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --main_process_port 29500 main_ic.py \
   --batch-size $BATCH_SIZE \
   --model ${ARCH}-${MODEL_SIZE}-${INPUT_SIZE}  \ 
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --epoch $EPOCH  \
   --lr 0.00001

```

You can also evaluate/train all models with all baselines using multiple ratio `r` by running:
``` sh
python scripts/eval_scripts/eval_ic_all.sh #off-the-shell evaluate
python scripts/train_scripts/train_ic_all.sh #retrain model
```
The results will be printed and saved to `outputs/ic_outputs` directory.


Text Classification 
---


### Using `pitome` with text classification models
We support `bert` and `distilbert` for text classification tasks. 
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

### Run
In this task, all experiments are conducted on the following datasets:  [IMBb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [sst2](stanfordnlp/sst2) and [Rotten Tomatoes](cornell-movie-review-data/rotten_tomatoes). 
By default, all data and model checkpoints are downloaded and saved to the folder specified by the `DATA_PATH` variable in `tasks/tc/config.py`. You can modify this variable to specify a different path as needed.

You can directly can evaluate off-the-shell perfomance by running:
```sh
python main_tc.py \
   --algo $ALGO \
   --ratio $RATIO \
   --task $TASK \
   --model $MODEL \
   --eval 
```
Or retrain the model by running:
```sh
CUDA_VISIBLE_DEVICES=$5 python -m accelerate.commands.launch main_tc.py \
   --model $MODEL \
   --algo $ALGO \
   --ratio $RATIO \
   --task $TASK 
```

You can also evaluate all models with all baselines using multiple ratio `r` by running:
```sh
python scripts/eval_scripts/eval_tc_all.sh #off-the-shell performance
python scripts/train_scripts/train_tc_all.sh #retrain
```
The results will be printed and saved to `outputs/tc_outputs` directory.


Notebook
---
You can refer to the [notebooks](notebooks) folder for example usages.


Citation
---


```
@article{tran2024accelerating,
  title={Accelerating Transformers with Spectrum-Preserving Token Merging},
  author={Tran, Hoai-Chau and Nguyen, Duy MH and Nguyen, Duy M and Nguyen, Trung-Tin and Le, Ngan and Xie, Pengtao and Sonntag, Daniel and Zou, James Y and Nguyen, Binh T and Niepert, Mathias},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

If you have any issues, feel free to contact us at tranhoaichau.00@gmail.com or Ho_Minh_Duy.Nguyen@dfki.de


Acknowledgement
---

Thanks [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) for providing open-source code. This repository is built based on the original ToMe structure.

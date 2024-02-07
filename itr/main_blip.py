from model.compressedModel import CompressedLAVISLIPWithQueue 
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from trainer import MyTrainer as Blip2Trainer 
from utils.data_utils import  get_loaders
from lavis.models import load_model_and_preprocess
from config import parser
from config import LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO, COCO, FLICKR, DATA_PATH
from dotenv import load_dotenv
import os
import torch

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
# DATA_PATH = os.environ.get('DATA_PATH')
torch.hub.set_dir(f'{DATA_PATH}/.vision_ckts')
COCO_PATH = f"{DATA_PATH}/coco/images"
FLICKR_PATH = f"{DATA_PATH}/flickr30k/flickr30k_images"

if __name__ == "__main__":
    config = parser.parse_args()
    for dataset in [COCO]:
        config.dataset = dataset

        # tokenizer = model.tokenizer
        if "flickr" in config.dataset:
            config.model_ckt = LAVIS_BLIP_BASE_FLICKR
            model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "flickr", is_eval=False)
            dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
        else:
            model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "coco", is_eval=False)
            config.model_ckt = LAVIS_BLIP_BASE_COCO 
            dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)


        def inner_training_loop(batch_size):
            config.batch_size = batch_size
            train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
                config.batch_size, 
                dataset,
                vis_processor=vis_processors['eval'],
                txt_processor=txt_processors['eval'],
                tokenizer=model.tokenizer,
                eval_batch_size=20
            )

            queue_model = CompressedLAVISLIPWithQueue(config, model)
            trainer = LavisTrainer(
                model=queue_model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                txt2img=test_txt2img,
                img2txt=test_img2txt
            )
            print(trainer.evaluate(use_1k=False))
            # print(trainer.evaluate('val'))
            # trainer.train()


        config.epochs = 2 
        config.enable_log = False
        config.use_margin_loss = False 
        config.r = 0.925

        for distil in [False]:
            config.distil = distil 
            for compress_method in [
                # 'PiToMe', 
                # 'ToMe',
                # 'dct', 
                'none',
            ]:
                config.compress_method = compress_method
                inner_training_loop(config.batch_size)

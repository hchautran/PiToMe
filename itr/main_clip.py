import torch
from transformers import (
    CLIPProcessor,
)
from lavis.datasets.builders import load_dataset
from itr.model.compressedModel import CompressedHFWithQueue
from transformers import CLIPProcessor, BlipProcessor
from itr.utils.data_utils import get_loaders 
from itr.trainer_queue import MyTrainer 


if __name__ == "__main__":
    from itr.config import parser
    from itr.config import COCO_PATH, FLICKR_PATH, CLIP_LARGE_PATCH_14, CLIP_BASE_PATCH_16, BLIP_BASE_FLICKR, FLICKR, COCO
    config = parser.parse_args()
    for dataset in [COCO]:
        config.dataset = dataset
        for model_ckt in [
            CLIP_LARGE_PATCH_14,
            # CLIP_BASE_PATCH_16,
        ]:
            config.model_ckt = model_ckt
            if "blip" in config.model_ckt:
                print("Getting BLIP processor...")
                processor = BlipProcessor.from_pretrained(
                    config.model_ckt, cache_dir=config.cache_dir
                )
            else:
                print("Getting CLIP processor...")
                processor = CLIPProcessor.from_pretrained(
                    config.model_ckt, cache_dir=config.cache_dir
                )

            if "flickr" in config.dataset:
                dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
            else:
                dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)



            def inner_training_loop(batch_size):
                config.batch_size = batch_size

                train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
                    config.batch_size, 
                    dataset,
                    vis_processor=processor,
                    txt_processor=None,
                    tokenizer=processor,
                    eval_batch_size=50
                )

                queue_model = CompressedHFWithQueue(config) 
                trainer = MyTrainer(
                    model=queue_model,
                    config=config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    txt2img=test_txt2img,
                    img2txt=test_img2txt
                )
                print(trainer.evaluate('test'))
                # print(trainer.evaluate('val'))
                # trainer.train()

            config.epochs = 2
            config.enable_log = False 
            config.use_margin_loss = False 
            config.r=0.925

            for compress_method in [
                'none', 
                # 'PiToMe', 
                # 'ToMe',
                # 'dct', 
            ]:
            # for compress_method in ['mean']:
                config.compress_method = compress_method
                for distil in [False]:
                    config.distil = distil 
                    inner_training_loop(config.batch_size)


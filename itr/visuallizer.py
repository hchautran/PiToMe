from model.compressedModel import CompressedLAVISLIPWithQueue 
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from trainer import MyTrainer as Blip2Trainer 
from utils.data_utils import  get_loaders
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor
from model.compressedModel import CompressedHFWithQueue
from config import parser
from model.compressedModel import CompressedLAVISBLIP2WithQueue 
from config import (
    COCO_PATH, 
    FLICKR_PATH, 
    CLIP_LARGE_PATCH_14, 
    CLIP_BASE_PATCH_16, 
    BLIP_BASE_FLICKR, 
    BLIP_BASE_COCO, 
    BLIP2,
    FLICKR, 
    LAVIS_BLIP_BASE_FLICKR, 
    LAVIS_BLIP_BASE_COCO, 
    COCO, 
    FLICKR, 
)
import wandb
 
  
class BLIPRunner():
    def __init__(self, config, algorithms="PiToMe"):
       # tokenizer = model.tokenizer
        config.enable_log = False
        config.compress_method = algorithms 
        if "flickr" in config.dataset:
            config.model_ckt = LAVIS_BLIP_BASE_FLICKR
            model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "flickr", is_eval=False)
            dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
        else:
            model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "coco", is_eval=False)
            config.model_ckt = LAVIS_BLIP_BASE_COCO 
            dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)


        train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
            config.batch_size, 
            dataset,
            vis_processor=vis_processors['eval'],
            txt_processor=txt_processors['eval'],
            tokenizer=model.tokenizer,
            eval_batch_size=50
        )

        queue_model = CompressedLAVISLIPWithQueue(config, model)
        self.trainer = LavisTrainer(
            model=queue_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            txt2img=test_txt2img,
            img2txt=test_img2txt
        )

    def run(self):
        return self.trainer.evaluate()
       
class CLIPRunner():
    def __init__(self, config, algorithms="PiToMe"):
        print("Getting CLIP processor...")
        config.enable_log=False
        config.compress_method = algorithms 
        processor = CLIPProcessor.from_pretrained(
            config.model_ckt, cache_dir=config.cache_dir
        )

        if "flickr" in config.dataset:
            dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
        else:
            dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)


        train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
            config.batch_size, 
            dataset,
            vis_processor=processor,
            txt_processor=None,
            tokenizer=processor,
            eval_batch_size=50
        )

        # model = HypGraphCLIPWithQueue(config) if "clip" in config.model_ckt else HypGraphBLIPWithQueue(config)
        queue_model = CompressedHFWithQueue(config) 
        # model = BLIPWithQueue(config) 
        self.trainer = LavisTrainer(
            model=queue_model,
            config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                txt2img=test_txt2img,
                img2txt=test_img2txt
            )
      

    def run(self):
        return self.trainer.evaluate()

     
class BLIP2Runner():
    def __init__(self, config, algorithms="PiToMe"):
        print("Getting blip2 processor...")
        config.enable_log=False
        config.model_ckt = 'blip2'
        config.compress_method = algorithms 
        model, vis_processors, txt_processors = load_model_and_preprocess("blip2", "coco", is_eval=False)
        # tokenizer = model.tokenizer
        if "flickr" in config.dataset:
            config.model_ckt = LAVIS_BLIP_BASE_FLICKR
            dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
        else:
            config.model_ckt = LAVIS_BLIP_BASE_COCO 
            dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)

        train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
            20, 
            dataset,
            vis_processor=vis_processors['eval'],
            txt_processor=txt_processors['eval'],
            tokenizer=model.tokenizer,
            eval_batch_size=50
        )
        blip2_model = CompressedLAVISBLIP2WithQueue(config, model)

        # model = HypGraphCLIPWithQueue(config) if "clip" in config.model_ckt else HypGraphBLIPWithQueue(config)
        self.trainer = Blip2Trainer(
            model=blip2_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            img2txt=test_img2txt,
            txt2img=test_txt2img
        )
    def run(self):
        return self.trainer.evaluate()


if __name__ == '__main__':
    config = parser.parse_args()
    config.dataset = FLICKR
    for model_ckt in [
        # CLIP_BASE_PATCH_16,
        # CLIP_LARGE_PATCH_14,
        # BLIP_BASE_COCO,
        BLIP2,
    ]:
        config.model_ckt = model_ckt
        for algo in [
            # 'none', 
            'PiToMe', 
            'ToMe',
            'dct', 
            'none', 
        ]:
            # wandb.init(
            #     name=f'{model_ckt}_{algo}', 
            #     config={
            #         "model":model_ckt,
            #         "algorithms": algo,

            #     }, 
            #     reinit=True, 
            #     project='itr-off-the-shell'
            # )
            for r in [0.9, 0.925, 0.95, 0.975]:
                config.r = r
                if 'clip' in model_ckt:
                    visualizer = CLIPRunner(config, algorithms=algo)
                elif 'blip2' in model_ckt: 
                    visualizer = BLIP2Runner(config, algorithms=algo)
                elif 'blip' in model_ckt: 
                    visualizer = BLIPRunner(config, algorithms=algo)

                metrics = visualizer.run()
                print(metrics)
                # wandb.log({
                #     "r@all": metrics['test/r_all'],
                #     "remain memory": metrics['eval memory'],
                #     "gflops": metrics['gflops']
                # })

        
        
        


from model.compressedModel import CompressedLAVISLIPWithQueue 
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from trainer import MyTrainer as Blip2Trainer 
from utils.data_utils import  get_loaders
from lavis.models import load_model_and_preprocess
from transformers import CLIPTokenizerFast, CLIPImageProcessor, CLIPModel
from model.compressedModel import CompressedHFWithQueue
from config import parser
from model.compressedModel import CompressedLAVISBLIP2WithQueue 
import pandas as pd
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
    def __init__(self, config, model, train_loader, val_loader, test_loader, algorithms="PiToMe" ):
       # tokenizer = model.tokenizer
        config.enable_log = False
        config.compress_method = algorithms 



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
    def __init__(self, config, model,train_loader, val_loader, test_loader, algorithms="PiToMe" ):
        print("Getting CLIP processor...")
        config.enable_log=False
        config.compress_method = algorithms 
        model = CompressedHFWithQueue(config, model) 
        self.trainer = LavisTrainer(
            model=model,
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
    def __init__(self, config, model,train_loader, val_loader, test_loader, algorithms="PiToMe" ):
        print("Getting blip2 processor...")
        config.enable_log=False
        config.model_ckt = 'blip2'
        config.compress_method = algorithms 


        blip2_model = CompressedLAVISBLIP2WithQueue(config, model)

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

df = pd.DataFrame()
model_dict = {
    CLIP_BASE_PATCH_16:'CLIP_B',
    CLIP_LARGE_PATCH_14:'CLIP_L',
    BLIP_BASE_COCO:'BLIP',
    BLIP_BASE_FLICKR:'BLIP',
    BLIP2:'BLIP2',
}
if __name__ == '__main__':
    config = parser.parse_args()
    config.dataset = FLICKR 

    if "flickr" in config.dataset:
        dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
    else:
        dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)

    for model_ckt in [
        CLIP_BASE_PATCH_16,
        CLIP_LARGE_PATCH_14,
        BLIP_BASE_COCO,
        BLIP2,
    ]:
        model = None
        train_loader = None
        val_loader= None
        test_loader= None
        config.model_ckt = model_ckt
        if 'clip' in model_ckt:
            processor = CLIPImageProcessor.from_pretrained(
                config.model_ckt, cache_dir=config.cache_dir
            )
            tokenizer= CLIPTokenizerFast.from_pretrained(
                config.model_ckt, cache_dir=config.cache_dir
            )
            model = CLIPModel.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
            train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
                config.batch_size, 
                dataset,
                vis_processor=processor,
                txt_processor=None,
                tokenizer=tokenizer,
                eval_batch_size=50
            )
        elif 'blip2' in model_ckt: 
            model, vis_processors, txt_processors = load_model_and_preprocess("blip2", 'coco', is_eval=False, device='cuda')
            train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
                20, 
                dataset,
                vis_processor=vis_processors['eval'],
                txt_processor=txt_processors['eval'],
                tokenizer=model.tokenizer,
                eval_batch_size=50
            )
        elif 'blip' in model_ckt: 
            model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", config.dataset, is_eval=False, device='cuda')
            train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
                20, 
                dataset,
                vis_processor=vis_processors['eval'],
                txt_processor=txt_processors['eval'],
                tokenizer=model.tokenizer,
                eval_batch_size=50
            )
        for algo in [
            'PiToMe', 
            'ToMe',
            'dct', 
            'baseline', 
        ]:
     
            ratios = [1.0] if algo == 'baseline' else [
                0.9, 0.925, 0.95, 0.975
            ] 
            for r in ratios:
                config.r = r
                if 'clip' in model_ckt:
                    visualizer = CLIPRunner(config, algorithms=algo, model=model, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader)
                elif 'blip2' in model_ckt: 
                    visualizer = BLIP2Runner(config, algorithms=algo, model=model, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader)
                elif 'blip' in model_ckt: 
                    visualizer = BLIPRunner(config, algorithms=algo, model=model, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader)

                metrics = visualizer.run()
                metrics['algo'] = algo 
                metrics['model'] = model_dict[model_ckt]
                df = pd.concat([df, pd.DataFrame(metrics,index=[0])]) 

                print(metrics)

    df.to_csv('flickr_ots.csv')

        
        
        


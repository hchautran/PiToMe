import wandb
from accelerate import Accelerator
from utils.retrivial_utils import report_metrics 
from tqdm.auto import tqdm
import torch
from config import CLIP_BASE_PATCH_16, CLIP_BASE_PATCH_32, CLIP_LARGE_PATCH_14, BLIP_BASE_FLICKR, BLIP_BASE_COCO, LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO
import time

names = {
   CLIP_BASE_PATCH_32: 'clip_base_32', 
   CLIP_BASE_PATCH_16: 'clip_base_16',
   CLIP_LARGE_PATCH_14: 'clip_large_14', 
   BLIP_BASE_FLICKR: 'hf_blip_base', 
   BLIP_BASE_COCO: 'hf_blip_base', 
   LAVIS_BLIP_BASE_FLICKR: 'lv_blip_base_FLICKR',
   LAVIS_BLIP_BASE_COCO: 'lv_blip_base_COCO' 
}

class MyTrainer:
    def __init__(
        self, config, model, train_loader, val_loader, test_loader, img2txt, txt2img
    ):
        self.config = config
        self.model_ckt = config.model_ckt
        self.grad_clip = config.grad_clip
        self.min_epochs = config.min_epochs
        self.eval_freq = config.eval_freq
        self.log_freq = config.log_freq
        self.patience = config.patience
        self.save_dir = config.save_dir
        self.epochs = config.epochs
        self.cache_dir = config.cache_dir
        self.momentum = config.momentum
        self.img2txt= img2txt
        self.txt2img = txt2img 
        self.device = torch.device(
            f"cuda:{config.cuda}"
            if torch.cuda.is_available() 
            else "cpu"
        )
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device_placement=False
        )
        self.enable_log = self.config.enable_log
        self.current_epoch = 0
        self.model = self.accelerator.prepare(model).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.1, patience=3, min_lr=1e-6
        )
        
        (
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, train_loader, val_loader, test_loader, self.scheduler
        )
        self.name = f'{names[config.model_ckt]}_{config.compress_method}_{config.r}_{config.vision_trainable_blocks}_{config.text_trainable_blocks}'
        print("RUNNING:", self.name)

        if self.enable_log:
            wandb.init(name=self.name, config=vars(self.config), reinit=True, project=config.dataset)
        print("trainable parameters:", self.model.num_parameters())
        self.log({"trainable parameters": self.model.num_parameters()})

    def log(self, stat):
        if self.enable_log:
            wandb.log(stat)

    def train(self):
        # loop over the dataset multiple times
        best_r_all = 0.0
        waiting = 0
        for epoch in range(self.epochs):
            current_step = 0
            with self.accelerator.accumulate(self.model):
                self.model.train()
                self.current_epoch = epoch
                running_loss = 0.0
                print('train loader length:', len(self.train_loader))
                for data in tqdm(self.train_loader):
                    if data['pixel_values'].shape[0] < self.config.batch_size: break
                    self.accelerator.free_memory()
                    self.optimizer.zero_grad()
                    current_step += 1

                    loss, stats = self.model(
                        input_ids=data["input_ids"].to(self.device),
                        attention_mask=data["attention_mask"].to(self.device),
                        pixel_values=data["pixel_values"].to(self.device),
                        image_id=data['img_id'].to(self.device),
                        epoch=epoch,
                        iters=current_step,
                        num_iters_per_epoch=len(self.train_loader),
                    )

                    self.accelerator.backward(loss)
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    self.optimizer.step()

                    running_loss += loss.item()

                    if (current_step + 1) % self.log_freq == 0:
                        self.log(stats)
                        print(stats)
                        print("Loss: {}".format(loss.item()))
                    if self.eval_freq != -1 and (current_step + 1) % self.eval_freq == 0:

                        test_metrics = self.evaluate(mode='test')
                        self.log(test_metrics)
                        print(test_metrics)

                        self.scheduler.step(test_metrics["test/r_all"])
                        if best_r_all < test_metrics["test/r_all"]:
                            best_r_all = test_metrics["test/r_all"]
                            self.log({"best r all": test_metrics["test/r_all"]})
                            self.log({"best r i2t": test_metrics["test/r_i2t"]})
                            self.log({"best r t2i": test_metrics["test/r_t2i"]})
                            print("best r all", best_r_all)

                        self.model.train()

                    
        print("Finished Training")

    def rerank(self, sims_matrix, vit_feats, text_feats, num_images, num_texts, k=20):
        score_matrix_i2t = torch.full(
            (num_images, num_texts), -100.0
        ).to(self.model.device)

        score_matrix_t2i = torch.full(
            (num_texts, num_images), -100.0
        ).to(self.model.device)

        with torch.no_grad():
            progress = tqdm(range(len(sims_matrix)))
            print('reranking text to image ...')
            for i, sims in enumerate(sims_matrix):
                topk_sim, topk_idx = sims.topk(k=k, dim=0)
                image_inputs = vit_feats[i].repeat(k, 1, 1).to(self.model.device)
                score = self.model.model.compute_itm(
                    vision_latents=image_inputs,
                    text_latents=text_feats[topk_idx]
                ).float()
                score = score[:, 1]
                score_matrix_i2t[i, topk_idx] = topk_sim + score 
                progress.update(1)

            sims_matrix = sims_matrix.t()
            progress = tqdm(range(len(sims_matrix)))

            print('reranking image to text...')
            for i, sims in enumerate(sims_matrix):
                topk_sim, topk_idx = sims.topk(k=k, dim=0)
                image_inputs = vit_feats[topk_idx.cpu()].to(self.model.device)
                score = self.model.model.compute_itm(
                    vision_latents=image_inputs,
                    text_latents=text_feats[i].repeat(k, 1, 1)
                ).float()
                score = score[:, 1]
                score_matrix_t2i[i, topk_idx] = topk_sim + score 
                progress.update(1)

        return score_matrix_i2t.cpu(), score_matrix_t2i.cpu()
            

    def evaluate(self, mode="test" ,use_1k=False):
        from torch.utils.data import DataLoader
        print("Evaluating current epoch", self.current_epoch)
        
        self.model.eval()

        dataset = self.val_loader if mode == "val" else self.test_loader
        if not isinstance(dataset, DataLoader):
            loader = self.accelerator.prepare(DataLoader(dataset, shuffle=False))
        else:
            loader = self.accelerator.prepare(dataset)

        all_text_embeds = []
        all_vision_embeds = []
        memory_used = 0
        total_flop = 0
        start = time.time()

        with torch.no_grad():
            for data in tqdm(loader):
                text_embeds, _ = self.model.get_text_features(
                    input_ids=data["input_ids"].to(self.device), attention_mask=data["attention_mask"].to(self.device)
                )
                vision_embeds, _, flop ,eval_memory, _ = self.model.get_vision_features(
                    pixel_values=data["pixel_values"].to(self.device), return_source=False
                )
          
                all_text_embeds.append(text_embeds.cpu())
                all_vision_embeds.append(vision_embeds.cpu())
                memory_used += eval_memory
                total_flop = flop 

            total_time = time.time() - start
           

            all_text_embeds = torch.concat(all_text_embeds, 0)
            all_vision_embeds = torch.concat(all_vision_embeds, 0)

            sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds)
            metrics = report_metrics(
                scores_t2i=sims_t2i.cpu().detach(), 
                scores_i2t=sims_t2i.T.cpu().detach(), 
                img2txt=self.img2txt, 
                txt2img=self.txt2img, 
                mode=f'{mode}'
            )
            metrics["eval memory"] = memory_used/len(loader)
            metrics["gflops"] = total_flop/1e9
            metrics["sample/sec"] = 50 * len(loader)/total_time
            self.accelerator.free_memory()
          

        return metrics

    def save(self):
        torch.save(self.model, f"{self.save_dir}/{self.name}.pth")

    def load(self):
        self.model = torch.load(f"{self.save_dir}/{self.name}.pth")


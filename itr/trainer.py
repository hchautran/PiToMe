import wandb
from accelerate import Accelerator
from utils.retrivial_utils import report_metrics 
from tqdm.auto import tqdm
import torch
from config import EUCLID, POINCARE, LORENTZ
from config import CLIP_BASE_PATCH_16, CLIP_BASE_PATCH_32, CLIP_LARGE_PATCH_14, BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO
import torch.nn.functional as F
import time

names = {
   CLIP_BASE_PATCH_32: 'clip_base_32', 
   CLIP_BASE_PATCH_16: 'clip_base_16',
   CLIP_LARGE_PATCH_14: 'clip_large_14', 
   BLIP_BASE_FLICKR: 'hf_blip_base', 
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
        self.img2txt = img2txt
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

        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
        )
       

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.1, patience=1
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
        self.name = f'blip2_{config.compress_method}_{config.r}_{config.vision_trainable_blocks}_{config.text_trainable_blocks}'
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

                        itc_test_metrics = self.evaluate(mode='test')
                        itc_val_metrics = self.evaluate(mode='val')
                        self.log(itc_test_metrics)
                        self.log(itc_val_metrics)
                        # self.log(itm_test_metrics)
                        # self.log(itm_val_metrics)
                        print(itc_test_metrics)
                        print(itc_val_metrics)

                        self.scheduler.step(itc_test_metrics["test/r_all"])
                        if best_r_all < itc_test_metrics["test/r_all"]:
                            best_r_all = itc_test_metrics["test/r_all"]
                            self.log({"best r all": itc_test_metrics["test/r_all"]})
                            self.log({"best r i2t": itc_test_metrics["test/r_i2t"]})
                            self.log({"best r t2i": itc_test_metrics["test/r_t2i"]})
                            print("best r all", best_r_all)

                        self.model.train()

                    
        print("Finished Training")

    def rerank(self, sims_matrix, vit_feats, text_ids, text_atts, num_images, num_texts, k=20):
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
                score = self.model.compute_itm(
                    vit_feats=image_inputs,
                    input_ids=text_ids[topk_idx],
                    attention_mask=text_atts[topk_idx],
                ).float()
                score_matrix_i2t[i, topk_idx] = score + topk_sim
                progress.update(1)

            sims_matrix = sims_matrix.t()
            progress = tqdm(range(len(sims_matrix)))

            for i, sims in enumerate(sims_matrix):
                print('reranking image to text...')
                topk_sim, topk_idx = sims.topk(k=k, dim=0)
                image_inputs = vit_feats[topk_idx.cpu()].to(self.model.device)
                score = self.model.compute_itm(
                    vit_feats=image_inputs,
                    input_ids=text_ids[i].repeat(k, 1),
                    attention_mask=text_atts[i].repeat(k, 1),
                ).float()
                score_matrix_t2i[i, topk_idx] = score + topk_sim
                progress.update(1)

        return score_matrix_i2t.cpu(), score_matrix_t2i.cpu()
            

    def evaluate(self, mode='test'):
        from torch.utils.data import DataLoader
        dataset = self.val_loader if mode == "val" else self.test_loader
        # texts = dataset.text
        # image = dataset.image
        # n_texts, n_images = len(texts), len(image)

        if not isinstance(dataset, DataLoader):
            loader = self.accelerator.prepare(DataLoader(dataset, shuffle=False))
        else:
            loader = self.accelerator.prepare(dataset)
        text_ids = []
        text_embeds = []
        text_atts = []
        vit_feats = []
        image_embeds = []
        max_len=35
        memory_used = 0
        total_flop = 0

        with torch.no_grad():
            for data in tqdm(loader):
                text_feat, _ = self.model.get_text_features(
                    input_ids=data["input_ids"].to(self.device), attention_mask=data["attention_mask"].to(self.device)
                )
                image_feat, vit_feat, flop ,eval_memory, _  = self.model.get_vision_features(
                    pixel_values=data["pixel_values"].to(self.device)
                )
                # cur_len = data['input_ids'].shape[-1]
                # input_ids = F.pad(data['input_ids'][0], (0, max_len - cur_len), "constant", 0)
                # attention_mask = F.pad(data['attention_mask'][0], (0, max_len - cur_len), "constant", 0) 
                # text_ids.append(input_ids.cpu())
                # text_atts.append(attention_mask.cpu())
                # vit_feats.append(vit_feat.cpu())
                image_embeds.append(image_feat.cpu())
                text_embeds.append(text_feat.cpu())
                memory_used += eval_memory
                total_flop += flop 


        text_embeds = torch.cat(text_embeds, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)
        # text_ids = torch.cat(text_ids, dim=0)
        # text_atts = torch.cat(text_atts, dim=0)
        # vit_feats = torch.cat(vit_feats, dim=0)

        sims_matrix = []
        print(image_embeds.shape)
        print(text_embeds.shape)
        for image_embed in image_embeds:
            sim_q2t = image_embed @ text_embeds.T
            sim_i2t, _ = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0)
        itc_metrics = report_metrics(
            scores_t2i=sims_matrix.cpu().detach().T, 
            scores_i2t=sims_matrix.cpu().detach(), 
            img2txt=self.img2txt, 
            txt2img=self.txt2img, 
            mode=mode 
            )
        print(itc_metrics)


        # score_matrix_i2t, score_matrix_t2i = self.rerank(
        #     sims_matrix=sims_matrix, 
        #     vit_feats=vit_feats, 
        #     text_ids=text_ids, 
        #     text_atts=text_atts, 
        #     num_images=n_images,
        #     num_texts=n_texts
        # )
        # itm_metrics = report_metrics(scores_t2i=score_matrix_t2i, scores_i2t=score_matrix_i2t, img2txt=dataset.img2txt, txt2img=dataset.txt2img, mode=f'{mode}_itm')
        # print(itm_metrics)

        itc_metrics["epoch"] = self.current_epoch
        itc_metrics["eval memory"] = memory_used/len(loader)
        itc_metrics["gflops"] = total_flop/len(loader)
        # itm_metrics["epoch"] = self.current_epoch
        
        # return itc_metrics, itm_metrics
        self.accelerator.clear()
        self.model.to('cpu')
        return itc_metrics
        

    def save(self):
        torch.save(self.model, f"{self.save_dir}/{self.name}.pth")

    def load(self):
        self.model = torch.load(f"{self.save_dir}/{self.name}.pth")


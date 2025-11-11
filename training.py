import pytorch_lightning as pl
import torch
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

class LitEfficientLoFTR(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
        self.model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")

        if freeze_backbone:
            for n, p in self.model.named_parameters():
                if "backbone" in n or "encoder" in n:  # 依据实际子模块名做筛选
                    p.requires_grad = False

    def forward(self, images):
        # images: list[PIL.Image] 或已张量化后的字典；这里保持与推理一致
        inputs = self.processor(images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)

    def compute_loss(self, outputs, batch):
        """
        你需要据自己的监督信号来实现：
        - 若 batch 提供 gt_matches / gt_keypoints / 相机姿态，可定义：
          * 匹配二分类/BCE（正负样本）；
          * 基于几何一致性的软损失（epipolar/重投影）；
          * 级联 coarse/fine 的监督（仿照 LoFTR 论文）。
        """
        # 伪代码示例：
        # logits = outputs.get("match_scores")  # 依据真实键名
        # target = batch["gt_match_matrix"]     # [N, ...] 0/1
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 假设 batch["images"] 是两个图像的列表/对；并包含监督标签
        outputs = self.forward(batch["images"])
        loss = self.compute_loss(outputs, batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch["images"])
        loss = self.compute_loss(outputs, batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (decay if p.requires_grad and p.dim() >= 2 else no_decay).append(p)
        optim_groups = [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.lr)
        # 可选调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# 用法（示例）：
# trainer = pl.Trainer(max_epochs=10, precision="16-mixed", devices=1, accelerator="gpu")
# lit = LitEfficientLoFTR(lr=1e-4, freeze_backbone=True)
# trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

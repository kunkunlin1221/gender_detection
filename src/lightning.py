import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

import lightning as L
from src.dataset import GenderDataset
from src.model import GenderModel


class LightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GenderModel()
        self.ce_loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="binary")
        self.f1_score = F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self.forward(imgs)
        loss = self.ce_loss(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self.forward(imgs)
        loss = self.ce_loss(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = self.accuracy(preds, labels)
        f1 = self.f1_score(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": acc, "val_f1_score": f1}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001, weight_decay=0.0001)


class LightningDataModule(L.LightningDataModule):
    def __init__(self, train_txt, val_txt, batch_size=32):
        super().__init__()
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = GenderDataset(self.train_txt)
        self.val_dataset = GenderDataset(self.val_txt)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=2, batch_size=self.batch_size, shuffle=False)

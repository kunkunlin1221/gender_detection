from fire import Fire

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.lightning import LightningDataModule, LightningModule

L.seed_everything(42)  # For reproducibility


def main(train_txt, val_txt, batch_size=16):
    model = LightningModule()
    data_module = LightningDataModule(train_txt, val_txt, batch_size)

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                monitor="val_f1_score",
                mode="max",
                save_top_k=1,
                filename="{epoch}-{val_f1_score:.2f}-{val_accuracy:.2f}",
            )
        ],
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    Fire(main)

import pathlib
import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from models.lightning_crop import Crop
from models.crop import CropModel

from loader.crop_dataset import get_crop_dataset_loader

if __name__ == "__main__":
    train_loader = get_crop_dataset_loader("data", 8)
    val_loader = get_crop_dataset_loader("data", 1, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='trained_models/crop/',
        filename='crop-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = L.Trainer(callbacks=[checkpoint_callback])

    model = CropModel((17, 12, 25, 25), 4)
    lightning_model = Crop(model, 0.001)

    trainer.fit(lightning_model, train_loader, val_loader)

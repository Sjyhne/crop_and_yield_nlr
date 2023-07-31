import pathlib
import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from models.lightning_crop import Crop
from models.crop_model import CropModel
from models.lightning_yield import Yield
from models.yield_model import YieldModel


from loader.crop_dataset import get_crop_dataset_loader, get_yield_dataset_loader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="crop")
    args = parser.parse_args()

    if args.model == "crop":
        train_loader = get_crop_dataset_loader("data/crop/train", 8)
        val_loader = get_crop_dataset_loader("data/crop/test", 1, shuffle=False)
        model = CropModel((17, 12, 25, 25), 4)
        lightning_model = Crop(model, 0.0001)
    elif args.model == "yield":
        train_loader = get_yield_dataset_loader("data/yield/train", 8)
        val_loader = get_yield_dataset_loader("data/yield/test", 1, shuffle=False)
        model = YieldModel(17, 12, None)
        lightning_model = Yield(model, 0.0001)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'trained_models/{args.model}/',
        filename=args.model + '-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = L.Trainer(callbacks=[checkpoint_callback], accelerator='gpu', devices=[0], max_epochs=100)


    trainer.fit(lightning_model, train_loader, val_loader)

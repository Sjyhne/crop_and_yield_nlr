import pathlib
import argparse
import torch

import lightning as L
import pandas as pd
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
        # train_loader = get_crop_dataset_loader("data/crop/train", 8)
        # val_loader = get_crop_dataset_loader("data/crop/test", 1, shuffle=False)
        pred_loader = get_crop_dataset_loader("data/crop", 1, shuffle=False, pred=True)
        model = CropModel((17, 12, 25, 25), 4)
        model = Crop.load_from_checkpoint("trained_models/crop/crop-epoch=09-val_loss=0.67-val_acc=0.732.ckpt", model=model, lr=0.0001)
        model.eval()
    elif args.model == "yield":
        train_loader = get_yield_dataset_loader("data/yield/train", 8)
        val_loader = get_yield_dataset_loader("data/yield/test", 1, shuffle=False)
        model = YieldModel(17, 12, None)
        lightning_model = Yield(model, 0.0001)
        lightning_model = lightning_model.load_from_checkpoint(
            "trained_models/crop/yield-epoch=99-val_loss=20177.91.ckpt")
        lightning_model.eval()

    column_names = ["orgnr", "år", "teigid", "crop_pred"]

    # create an empty DataFrame
    df = pd.DataFrame(columns=column_names)

    for batch in pred_loader:
        images, _, info = batch
        pred = model(images)
        pred = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1).cpu().numpy()

        orgnr = info["orgnr"]
        year = info["år"]
        teigid = info["teigid"]

        new_row = {
            "orgnr": orgnr,
            "år": year,
            "teigid": teigid,
            "crop_pred": pred
        }

        df = df._append(new_row, ignore_index=True)


    df.to_csv("data/crop/pred.csv", index=False)



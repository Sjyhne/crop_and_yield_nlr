import torch
import pathlib
import csv

import pandas as pd
import h5py


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.datapath = pathlib.Path(datapath)
        self.labels = self.load_labels()
        self.images = self.load_images()

    def load_labels(self):
        csvs = list(self.datapath.glob("*.csv"))
        label_file = csvs[0]
        csv = pd.read_csv(label_file)
        label_dict = {}
        for i, row in csv.iterrows():
            if row["Orgnr"] not in label_dict:
                label_dict[row["Orgnr"]] = {}
            else:
                label_dict[row["Orgnr"]][row["År"]] = row["Kornsort"]

        return label_dict

    def load_images(self):
        h5s = list(self.datapath.glob("*.h5"))
        image_file = h5s[0]
        with h5py.File(image_file.__str__(), "r") as f:
            h5images = f['images']
            keys = list(h5images.keys())
            for key in keys:
                års = list(h5images[key].keys())
                for år in års:
                    teigids = list(h5images[key][år].keys())
                    for teigid in teigids:
                        periods = list(h5images[key][år][teigid].keys())
                        for period in periods:
                            l = h5images[key][år][teigid][period]
                            print(len(l))
                            exit(":")

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return 1


if __name__ == "__main__":
    ds = CropDataset("data")

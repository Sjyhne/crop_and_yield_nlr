import torch
import pathlib
import h5py

import pandas as pd
import numpy as np


KORN_TO_LABEL = {
    "Rug": 0,
    "Hvete": 1,
    "Bygg": 2,
    "Havre": 3
}


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.datapath = pathlib.Path(datapath)
        self.labels = self.load_labels()
        self.images = self.get_image_dict()

        self.samples = sorted(list(self.images.keys()))
        self.samples = self.samples[:50]


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

    def get_image(self, orgnr, year, teigid, period):
        label = f"images/{orgnr}/{year}/{teigid}/{period}"
        with h5py.File(self.file_name, "r") as file:
            img = file[label][()] / 100_000_000
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            return np.asarray(img * 255, dtype=np.uint8)

    def get_image_dict(self):
        h5s = list(self.datapath.glob("*.h5"))
        image_file = h5s[0]

        self.file_name = image_file.__str__()

        image_dict = {}
        self.teig_to_org = {}

        with h5py.File(image_file.__str__(), "r") as f:
            h5images = f['images']
            orgnrs = list(h5images.keys())
            for key in orgnrs:

                intkey = int(key)

                if intkey not in self.labels:
                    continue

                års = list(h5images[key].keys())
                for år in års:

                    intår = int(år)
                    if intår not in self.labels[intkey]:
                        continue

                    if år not in image_dict:
                        image_dict[år] = {}

                    teigids = list(h5images[key][år].keys())
                    for teigid in teigids:

                        self.teig_to_org[teigid] = key

                        if teigid not in image_dict[år]:
                            image_dict[år][teigid] = []


                        periods = list(h5images[key][år][teigid].keys())
                        if len(periods) != 17:
                            del image_dict[år][teigid]
                            continue
                        if len(image_dict[år][teigid]) != 17:
                            for period in periods:
                                image_dict[år][teigid].append(period)

        new_image_dict = {}


        for år in image_dict.keys():
            for teigid in image_dict[år].keys():
                unique_key = f"{år}_{teigid}"
                if unique_key not in new_image_dict:
                    new_image_dict[unique_key] = []
                for period in image_dict[år][teigid]:
                    new_image_dict[unique_key].append(period)

        return new_image_dict

    def __getitem__(self, idx):
        image_sample = self.samples[idx]
        år, teigid = image_sample.split("_")
        orgnr = self.teig_to_org[teigid]
        periods = self.images[image_sample]

        images = np.zeros((len(periods), 25, 25, 12))
        for i, period in enumerate(periods):
            img = self.get_image(orgnr, år, teigid, period)
            images[i] = img

        images = torch.tensor(images, dtype=torch.float32) / 255
        label = self.labels[int(orgnr)][int(år)]
        label = KORN_TO_LABEL[label]

        info = {
            "orgnr": orgnr,
            "år": år,
            "teigid": teigid,
            "periods": periods
        }

        images = images.permute(0, 3, 1, 2)

        return images, label, info

    def __len__(self):
        return len(self.samples)

def get_crop_dataset_loader(datapath, batch_size, shuffle=True):
    ds = CropDataset(datapath)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    ds = CropDataset("data")
    print(len(ds))
    images, label, info = ds[0]
    print(images.shape)
    print(label)
    print(info)

    loader = get_crop_dataset_loader("data", 8)

    for batch in loader:
        images, label, info = batch
        print(images.shape)
        print(label)
        print(info)
        break